import importlib
import inspect
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_adapters.albef import ALBEF_SOURCE_ROOT, _official_albef_class  # noqa: E402


OFFICIAL_MODELS_INIT = (ALBEF_SOURCE_ROOT / "models" / "__init__.py").resolve()
OFFICIAL_PRETRAIN = (ALBEF_SOURCE_ROOT / "models" / "model_pretrain.py").resolve()


class AlbefOfficialImportTest(unittest.TestCase):
    def test_inspect_getfile_locates_official_pretrain_module(self):
        model_class = _official_albef_class()
        source = Path(inspect.getfile(model_class)).resolve()
        self.assertEqual(source, OFFICIAL_PRETRAIN)
        self.assertEqual(Path(sys.modules["models"].__file__).resolve(), OFFICIAL_MODELS_INIT)

    def test_reload_does_not_replace_project_model_adapters(self):
        import model_adapters

        adapter_file = Path(model_adapters.__file__).resolve()
        first = _official_albef_class()
        second = _official_albef_class()
        self.assertIs(first, second)
        self.assertEqual(Path(model_adapters.__file__).resolve(), adapter_file)
        self.assertTrue(adapter_file.as_posix().endswith("/src/model_adapters/__init__.py"))
        self.assertIs(sys.modules["model_adapters"], model_adapters)

    def test_refuses_to_overwrite_a_different_models_package(self):
        previous = {name: sys.modules[name] for name in list(sys.modules) if name == "models" or name.startswith("models.")}
        fake = types.ModuleType("models")
        fake.__file__ = str(Path("/tmp/other-models/__init__.py"))
        sys.modules["models"] = fake
        try:
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                _official_albef_class()
        finally:
            for name in list(sys.modules):
                if name == "models" or name.startswith("models."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)

    def test_four_adapter_modules_import_without_the_official_models_name(self):
        import model_adapters.albef as albef
        import model_adapters.beit3 as beit3
        import model_adapters.clip_openai as clip_openai
        import model_adapters.vista as vista

        self.assertTrue(hasattr(albef, "AlbefM0Adapter"))
        self.assertTrue(hasattr(beit3, "Beit3BaseItcAdapter"))
        self.assertTrue(hasattr(clip_openai, "OpenAIClipViTL14Adapter"))
        self.assertTrue(hasattr(vista, "VistaStage1Adapter"))
        self.assertNotEqual(Path(albef.__file__).resolve(), OFFICIAL_MODELS_INIT)
        importlib.import_module("model_adapters.factory")


if __name__ == "__main__":
    unittest.main()
