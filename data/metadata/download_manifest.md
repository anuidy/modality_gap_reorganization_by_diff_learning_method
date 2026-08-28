# M0 Baseline Download Manifest

**Status:** Download deferred because the network is unstable.  
**Purpose:** Obtain the four pretrained model candidates and the two fixed probe sources needed to compute the pre-fine-tuning (`M0`) six-metric baseline.

The sizes below were measured from the remote `Content-Length` headers on 2026-08-25. They exclude source code, Python packages, and extracted-file overhead.

## Important checkpoint decisions

Do not download an alternative checkpoint without recording the decision in this file.

| Model | Current candidate | Decision status |
|---|---|---|
| CLIP-SF | OpenAI `ViT-L/14` | Pending: the research plan also permits a LAION initialization. |
| VISTA | `BGE_EVA_Token_S1.pth` from `JUNJIE99/VISTA`, plus BGE-base text backbone | Selected: verified Stage-1 checkpoint. |
| ALBEF | Official `ALBEF.pth` trained on 14M pairs | Pending: this is a generic pretrained checkpoint, not an MS COCO/Flickr retrieval-finetuned checkpoint. |
| BEiT-3 | Official `beit3_base_itc_patch16_224.pth` | Selected in the research plan. |

## Model weights

| Model | File | Download URL | Destination | Size |
|---|---|---|---|---:|
| CLIP-SF | `ViT-L-14.pt` | <https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt> | `data/raw/models/clip_sf/ViT-L-14.pt` | 0.87 GiB |
| VISTA | `BGE_EVA_Token_S1.pth` (Stage-1) | <https://huggingface.co/JUNJIE99/VISTA/resolve/main/BGE_EVA_Token_S1.pth?download=true> | `data/raw/models/vista/BGE_EVA_Token_S1.pth` | 0.37 GiB |
| VISTA | `model.safetensors` (BGE-base text backbone) | <https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors?download=true> | `data/raw/models/vista/bge-base-en-v1.5/model.safetensors` | 0.41 GiB |
| VISTA | tokenizer/config companion files | <https://huggingface.co/BAAI/bge-base-en-v1.5/tree/main> | `data/raw/models/vista/bge-base-en-v1.5/` | small; download the repository snapshot with the weight |
| ALBEF | `ALBEF.pth` (14M pretrained) | <https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth> | `data/raw/models/albef/ALBEF_14M.pth` | 3.25 GiB |
| ALBEF | `bert-base-uncased` text backbone | <https://huggingface.co/bert-base-uncased/tree/main> | `data/raw/models/albef/bert-base-uncased/` | 0.41 GiB |
| BEiT-3 | `beit3_base_itc_patch16_224.pth` | <https://github.com/addf400/files/releases/download/beit3/beit3_base_itc_patch16_224.pth> | `data/raw/models/beit3/beit3_base_itc_patch16_224.pth` | 0.42 GiB |
| BEiT-3 | `beit3.spm` tokenizer | <https://github.com/addf400/files/releases/download/beit3/beit3.spm> | `data/raw/models/beit3/beit3.spm` | negligible |

**Model-weight subtotal:** approximately **5.72 GiB**.

## External probe: MS COCO 2017 validation

Use all 5,000 validation images. Create and save a deterministic manifest that chooses exactly one caption per image before embedding extraction.

| File | Download URL | Destination | Size |
|---|---|---|---:|
| `val2017.zip` | <http://images.cocodataset.org/zips/val2017.zip> | `data/raw/coco_2017_val/archives/val2017.zip` | 0.76 GiB |
| `annotations_trainval2017.zip` | <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> | `data/raw/coco_2017_val/archives/annotations_trainval2017.zip` | 0.24 GiB |

After extraction, use `annotations/captions_val2017.json` and save the selected 5K pairs in `data/splits/coco_2017_val_probe_v1.*`.

**COCO download subtotal:** approximately **1.00 GiB**.

## In-domain probe: LCS-558K / LLaVA-Pretrain

Select and save the fixed 10K probe sample IDs before any training. The published image source is a single full archive; it cannot be budgeted as a 10K-only download without a separate indexed mirror.

| File | Download URL | Destination | Size |
|---|---|---|---:|
| `blip_laion_cc_sbu_558k.json` | <https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true> | `data/raw/lcs_558k/blip_laion_cc_sbu_558k.json` | 0.17 GiB |
| `images.zip` | <https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true> | `data/raw/lcs_558k/archives/images.zip` | 25.48 GiB |

After extraction, save the fixed 10K pairs in `data/splits/lcs_558k_in_domain_probe_v1.*` and record the source dataset revision in its manifest.

**LCS download subtotal:** approximately **25.65 GiB**.

## Download totals and order

| Scope | Expected download |
|---|---:|
| Four model candidates only | 5.72 GiB |
| Models + COCO external M0 pilot | 6.72 GiB |
| Models + COCO + complete LCS archive | 32.36 GiB |

Recommended order once the network is available:

1. Resolve the three pending checkpoint decisions above.
2. Download model weights and the COCO probe; validate the adapter and six-metric pipeline on M0.
3. Download the LCS annotation, create a versioned 10K split manifest, then obtain and extract the LCS image archive.
4. After every download, record file size and SHA-256 in `data/metadata/checksums.sha256` before use.

## Source code references

- CLIP: <https://github.com/openai/CLIP>
- VISTA / Visualized-BGE: <https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge>
- ALBEF: <https://github.com/salesforce/ALBEF>
- BEiT-3: <https://github.com/microsoft/unilm/tree/master/beit3>
- LLaVA data documentation: <https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md>
