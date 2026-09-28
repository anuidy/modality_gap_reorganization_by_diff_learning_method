import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.audit_groups import inspect_gradients, parameter_group
from training.audit_config import AuditConfig
from training.backends import TrainingBackend, TrainingStepResult, _relation_step
from training.gradient_audit import _audit_device, audit_backend, audit_cases, run_gradient_audit, state_hashes


class TinyClip(TrainingBackend):
    model_name = "clip"

    def __init__(self, detach_text=False, mutate_forward=False, raise_forward=False):
        super().__init__(torch.device("cpu"))
        self.model = nn.Module()
        self.model.visual = nn.Linear(3, 3, bias=False)
        self.model.visual.proj = nn.Parameter(torch.eye(3))
        self.model.transformer = nn.Linear(3, 3, bias=False)
        self.model.text_projection = nn.Parameter(torch.eye(3))
        self.model.logit_scale = nn.Parameter(torch.tensor(0.5))
        self.detach_text = detach_text
        self.mutate_forward = mutate_forward
        self.raise_forward = raise_forward

    def prepare_batch(self, batch, augmentation_seed):
        return batch

    def forward(self, batch, branch, optimizer_step):
        if self.raise_forward:
            raise RuntimeError("injected forward failure")
        if self.mutate_forward:
            with torch.no_grad():
                self.model.logit_scale.add_(0.1)
        image = self.model.visual(batch[0]) @ self.model.visual.proj
        text = self.model.transformer(batch[1]) @ self.model.text_projection
        if self.detach_text:
            text = text.detach()
        return _relation_step(tuple("abcdef"), image, text, self.model.logit_scale.exp(), branch, optimizer_step, None)


class TinyAlbef(TrainingBackend):
    model_name = "albef"

    def __init__(self):
        super().__init__(torch.device("cpu"))
        from training.albef_accumulation import DeferredAlbefStateUpdates
        self._deferred_state_updates = DeferredAlbefStateUpdates()
        m = self.model = nn.Module()
        m.visual_encoder = nn.Linear(2, 2)
        m.vision_proj = nn.Linear(2, 2)
        m.text_proj = nn.Linear(2, 2)
        m.itm_head = nn.Linear(2, 2)
        m.temp = nn.Parameter(torch.tensor(0.1))
        m.text_encoder = nn.Module()
        m.text_encoder.config = SimpleNamespace(fusion_layer=1)
        m.text_encoder.bert = nn.Module()
        m.text_encoder.bert.embeddings = nn.Linear(2, 2)
        m.text_encoder.bert.encoder = nn.Module()
        m.text_encoder.bert.encoder.layer = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        m.text_encoder.bert.encoder.layer[1].crossattention = nn.Linear(2, 2)
        m.text_encoder.cls = nn.Linear(2, 2)
        for name in ("visual_encoder", "vision_proj", "text_encoder", "text_proj"):
            setattr(m, name + "_m", copy.deepcopy(getattr(m, name)).requires_grad_(False))
        for name in ("image_queue", "text_queue", "queue_ptr"):
            m.register_buffer(name, torch.zeros(1))
        self.momentum_calls = 0
        self.enqueue_calls = 0

        def momentum():
            self.momentum_calls += 1
            with torch.no_grad():
                m.visual_encoder_m.weight.add_(0.01)

        def enqueue(*args):
            self.enqueue_calls += 1
            m.queue_ptr.add_(1)

        m._momentum_update = momentum
        m._dequeue_and_enqueue = enqueue

    def begin_optimizer_step(self, micro_batches):
        self._deferred_state_updates.begin(self.model, micro_batches)

    def prepare_batch(self, batch, augmentation_seed):
        return batch

    def forward(self, batch, branch, optimizer_step):
        m = self.model
        with self._deferred_state_updates.intercept_forward():
            m._momentum_update()
            image = m.visual_encoder(batch)
            text = m.text_encoder.bert.encoder.layer[0](m.text_encoder.bert.embeddings(batch))
            itc = (m.vision_proj(image).sum() + m.text_proj(text).sum()) / m.temp
            m._dequeue_and_enqueue(image.detach(), text.detach())
            if branch == "itc_only":
                return TrainingStepResult(itc, {"loss": itc, "loss/ITC": itc}, None)
            layer = m.text_encoder.bert.encoder.layer[1]
            fused = layer(text) + layer.crossattention(image)
            itm, mlm = m.itm_head(fused).sum(), m.text_encoder.cls(fused).sum()
            total = itc + itm + mlm
            return TrainingStepResult(total, {"loss": total, "loss/ITC": itc, "loss/ITM": itm, "loss/MLM": mlm}, None)


class GradientAuditTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.batch = (torch.randn(6, 3), torch.randn(6, 3))

    def test_mixed_relations_restore_parameters_rng_and_modes_without_optimizer(self):
        backend = TinyClip().eval()
        before = state_hashes(backend)
        rng = torch.get_rng_state().clone()
        with patch("torch.optim.AdamW.step", side_effect=AssertionError("optimizer forbidden")):
            reports = audit_backend(backend, self.batch, branch="mixed_3m_fn_off", seed=4, precision="fp32")
        self.assertEqual([r["case"] for r in reports], ["mixed_3m_fn_off"])
        self.assertTrue(all(r["status"] == "pass" for r in reports), reports)
        self.assertEqual(before, state_hashes(backend))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(backend.training)
        self.assertTrue(all(p.grad is None for p in backend.parameters()))

    def test_detached_text_cannot_pass_using_stale_gradients(self):
        backend = TinyClip(detach_text=True)
        for p in backend.parameters():
            p.grad = torch.ones_like(p)
        result = audit_backend(backend, self.batch, branch="standard", seed=4, precision="fp32")[0]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["gradients"]["groups"]["text_encoder"]["counts"]["none"], 1)

    def test_zero_and_nonfinite_gradients_fail(self):
        backend = TinyClip()
        for p in backend.parameters():
            p.grad = torch.ones_like(p)
        backend.model.logit_scale.grad.zero_()
        self.assertEqual(inspect_gradients(backend, "clip", "total")["status"], "fail")
        backend.model.logit_scale.grad.fill_(float("nan"))
        result = inspect_gradients(backend, "clip", "total")
        self.assertEqual(result["groups"]["temperature"]["counts"]["nonfinite"], 1)

    def test_inactive_zero_tensor_and_unfrozen_momentum_fail(self):
        backend = TinyAlbef()
        backend.model.itm_head.weight.grad = torch.zeros_like(backend.model.itm_head.weight)
        backend.model.visual_encoder_m.weight.requires_grad_(True)
        result = inspect_gradients(backend, "albef", "ITC")
        self.assertEqual(result["groups"]["itm_head"]["status"], "fail")
        self.assertEqual(result["groups"]["momentum/visual_encoder_m"]["status"], "fail")

    def test_unclassified_and_absent_required_group_fail(self):
        backend = TinyClip()
        backend.model.surprise = nn.Parameter(torch.ones(1))
        del backend.model.text_projection
        result = inspect_gradients(backend, "clip", "total")
        self.assertEqual(result["groups"]["unclassified"]["status"], "fail")
        self.assertEqual(result["groups"]["text_projection"]["parameters"], [])

    def test_forward_mutation_is_detected_and_restored(self):
        backend = TinyClip(mutate_forward=True)
        before = state_hashes(backend)
        report = audit_backend(backend, self.batch, branch="standard", seed=4, precision="fp32")[0]
        self.assertEqual(report["unexpected_forward_changes"], ["model.logit_scale"])
        self.assertEqual(report["status"], "fail")
        self.assertEqual(state_hashes(backend), before)

    def test_exception_is_reported_and_cleanup_runs(self):
        backend = TinyClip(raise_forward=True).eval()
        before = state_hashes(backend)
        report = audit_backend(backend, self.batch, branch="standard", seed=4, precision="fp32")[0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertEqual(state_hashes(backend), before)
        self.assertFalse(backend.training)

    def test_backward_state_mutation_is_detected(self):
        backend = TinyClip()
        backend.register_buffer("diagnostic_counter", torch.zeros(()))

        def mutate(grad):
            backend.diagnostic_counter.add_(1)
            return grad

        hook = backend.model.logit_scale.register_hook(mutate)
        before = state_hashes(backend)
        report = audit_backend(backend, self.batch, branch="standard", seed=4, precision="fp32")[0]
        hook.remove()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["backward_state_changes"], ["diagnostic_counter"])
        self.assertEqual(state_hashes(backend), before)

    def test_nonfinite_loss_is_reported_without_backward(self):
        backend = TinyClip()
        with torch.no_grad():
            backend.model.logit_scale.fill_(1000)
        report = audit_backend(backend, self.batch, branch="standard", seed=4, precision="fp32")[0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "FloatingPointError")
        self.assertNotIn("backward_seconds", report)
        json.dumps(report, allow_nan=False)

    def test_albef_components_and_state_isolation(self):
        backend = TinyAlbef()
        before = state_hashes(backend)
        reports = audit_backend(backend, torch.randn(4, 2), branch="full_albef", seed=4, precision="fp32")
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(r["status"] == "pass" for r in reports), reports)
        self.assertEqual(backend.momentum_calls, 4)
        self.assertEqual(backend.enqueue_calls, 0)
        self.assertFalse(backend._deferred_state_updates.active)
        self.assertEqual(before, state_hashes(backend))
        self.assertEqual(reports[2]["gradients"]["groups"]["temperature"]["expectation"], "inactive")
        self.assertTrue(all(r["losses"] == reports[0]["losses"] for r in reports))

    def test_albef_itc_inactive_heads_and_frozen_encoders(self):
        backend = TinyAlbef()
        report = audit_backend(backend, torch.randn(4, 2), branch="itc_only", seed=4, precision="fp32")[0]
        self.assertEqual(report["status"], "pass", report)
        groups = report["gradients"]["groups"]
        self.assertEqual(groups["mlm_head"]["expectation"], "inactive")
        self.assertEqual(groups["momentum/text_encoder_m"]["expectation"], "frozen")

    def test_albef_exception_discards_pending_queue_features(self):
        backend = TinyAlbef()
        before = state_hashes(backend)
        original = backend.forward

        def failing(*args):
            original(*args)
            raise RuntimeError("failure after queue collection")

        backend.forward = failing
        report = audit_backend(backend, torch.randn(4, 2), branch="itc_only", seed=4, precision="fp32")[0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(backend.enqueue_calls, 0)
        self.assertFalse(backend._deferred_state_updates.active)
        self.assertEqual(state_hashes(backend), before)

    def test_shared_mlm_embedding_alias_not_counted_twice(self):
        backend = TinyAlbef()
        backend.model.text_encoder.cls.weight = backend.model.text_encoder.bert.embeddings.weight
        report = audit_backend(backend, torch.randn(4, 2), branch="itc_only", seed=4, precision="fp32")[0]
        self.assertEqual(report["status"], "pass", report)
        params = report["gradients"]["groups"]["text_embeddings"]["parameters"]
        self.assertEqual(len(params[0]["aliases"]), 2)

    def test_architecture_specific_names(self):
        self.assertEqual(parameter_group("beit3", "model.beit3.encoder.layers.0.self_attn.k_proj.A.weight"), "image_attention")
        self.assertEqual(parameter_group("beit3", "model.beit3.encoder.layers.0.ffn.B.fc1.weight"), "text_ffn")
        self.assertEqual(parameter_group("beit3", "model.beit3.vision_embed.mask_token"), "unused_mask_token")
        self.assertEqual(parameter_group("vista", "model.model_visual.visual.norm.weight"), "unused_eva_output")
        self.assertEqual(parameter_group("vista", "model.bge_pooler.dense.weight"), "unused_pooler")
        self.assertEqual(parameter_group("albef", "model.text_encoder.bert.encoder.layer.6.crossattention.self.key.weight"), "cross_attention")

    def test_case_matrix(self):
        self.assertEqual(len(audit_cases("standard")), 1)
        self.assertEqual(len(audit_cases("mixed_3m_fn_off")), 1)
        self.assertEqual([x[2] for x in audit_cases("full_albef")], ["total", "ITC", "ITM", "MLM"])

    def test_cuda_default_resolves_visible_device_without_implicit_index(self):
        with patch("torch.cuda.current_device", return_value=2), patch("torch.cuda.set_device") as select:
            self.assertEqual(_audit_device("cuda"), torch.device("cuda:2"))
            select.assert_called_once_with(torch.device("cuda:2"))
        with patch("torch.cuda.current_device") as current, patch("torch.cuda.set_device") as select:
            self.assertEqual(_audit_device("cuda:0"), torch.device("cuda:0"))
            current.assert_not_called()
            select.assert_called_once_with(torch.device("cuda:0"))

    def test_tiny_native_beit3_parameter_coverage_and_backward(self):
        from torchscale.architecture.config import EncoderConfig
        from torchscale.model.BEiT3 import BEiT3

        class TinyBeit3(TinyClip):
            model_name = "beit3"

            def __init__(self):
                TrainingBackend.__init__(self, torch.device("cpu"))
                self.model = nn.Module()
                args = EncoderConfig(img_size=16, patch_size=8, vocab_size=32, multiway=True,
                                     encoder_embed_dim=8, encoder_attention_heads=2, encoder_ffn_embed_dim=16,
                                     encoder_layers=2, normalize_output=True, no_output_layer=True)
                self.model.beit3 = BEiT3(args)
                self.model.vision_head = nn.Linear(8, 8, bias=False)
                self.model.language_head = nn.Linear(8, 8, bias=False)
                self.model.logit_scale = nn.Parameter(torch.tensor(0.5))

            def forward(self, batch, branch, optimizer_step):
                image = self.model.beit3(visual_tokens=batch[0])["encoder_out"][:, 0]
                text = self.model.beit3(textual_tokens=batch[1])["encoder_out"][:, 0]
                return _relation_step(tuple("abcdef"), self.model.vision_head(image),
                                      self.model.language_head(text), self.model.logit_scale.exp(),
                                      branch, optimizer_step, None)

        backend = TinyBeit3()
        batch = (torch.randn(6, 3, 16, 16), torch.randint(0, 32, (6, 5)))
        reports = audit_backend(backend, batch, branch="mixed_3m_fn_off", seed=4, precision="fp32")
        self.assertTrue(all(r["status"] == "pass" for r in reports), reports)

    def test_vista_native_joint_path_counts_include_image_reuse(self):
        class VistaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_visual = nn.Module()
                self.model_visual.visual = nn.Linear(3, 3)
                self.model_visual.logit_scale = nn.Parameter(torch.ones(()))
                self.visual_proj = nn.Linear(3, 3)
                self.bge_embeddings = nn.Linear(3, 3)
                self.bge_encoder = nn.Linear(3, 3)
                self.bge_pooler = nn.Linear(3, 3)
                self.temperature = 0.02

            def encode_image(self, image):
                return self.encode_mm(image, torch.zeros_like(image))

            def encode_text(self, text):
                return self.bge_encoder(self.bge_embeddings(text))

            def encode_mm(self, image, text):
                return self.bge_encoder(self.visual_proj(self.model_visual.visual(image)) + self.bge_embeddings(text))

        from training.backends import VistaTrainingBackend
        backend = VistaTrainingBackend.__new__(VistaTrainingBackend)
        TrainingBackend.__init__(backend, torch.device("cpu"))
        backend.model = VistaModel()
        batch = SimpleNamespace(images=self.batch[0], text_tokens=self.batch[1], semantic_ids=tuple("abcdef"))
        reports = audit_backend(backend, batch, branch="mixed_3m_fn_off", seed=4, precision="fp32")
        self.assertTrue(all(r["status"] == "pass" for r in reports), reports)
        self.assertEqual([r["path_calls"]["encode_mm"] for r in reports], [2])
        self.assertNotIn("encode_mm", backend.model.__dict__)

    def test_runner_loads_real_batch_and_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for index in range(4):
                Image.new("RGB", (4, 4), color=(index, 0, 0)).save(root / f"{index}.png")
                records.append({"sample_id": str(index), "semantic_id": str(index),
                                "image": f"{index}.png", "text": f"sample {index}"})
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            config = AuditConfig("clip_standard", "clip", "standard", root / "unused.pt", {},
                                 SimpleNamespace(train_manifest=manifest, image_root=root),
                                 4, 4, "fp32", {}, {}, {})
            backend = TinyClip()
            backend.prepare_batch = lambda raw, seed: self.batch
            with patch("training.gradient_audit.create_training_backend", return_value=backend):
                report = run_gradient_audit(config, "cpu")
            self.assertEqual(report["status"], "pass", report)
            self.assertEqual(set(report["batch"]["sample_ids"]), set("0123"))
            self.assertTrue(report["source_sha256"])
            json.dumps(report, allow_nan=False)
            backend.checkpoint_load_report["missing_keys"] = ["text_projection"]
            with patch("training.gradient_audit.create_training_backend", return_value=backend):
                rejected = run_gradient_audit(config, "cpu")
            self.assertEqual(rejected["status"], "error")
            self.assertEqual(rejected["cases"], [])
            self.assertEqual(rejected["checkpoint_load"]["missing_keys"], ["text_projection"])


if __name__ == "__main__":
    unittest.main()
