"""Pre-compute frozen CLIP text embeddings (CPU, run once)

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 10 — Pre-compute CLIP Text Embeddings (Run Once, ~3 min)

Loads `openai/clip-vit-base-patch32` on **CPU** (some MPS builds have incomplete CLIP attention support) and encodes every patient's radiology report into a 512-dimensional embedding vector.

**Why pre-compute?**
Running CLIP inside the training loop would add ~200 ms per batch on CPU, tripling training time. Pre-computing once and caching results reduces per-batch overhead to a single array lookup.

**Cache behaviour:**
- First run: encodes all 369 patients (~3 min) and saves to `outputs/clip_text_embeddings.pt`.
- Subsequent runs: loads the cache in under 1 second.

The resulting `emb_dict` maps `patient_id → torch.Tensor[512]`.
"""

# ── Step A: Pre-compute CLIP text embeddings (run ONCE, ~3 min on CPU) ───────
# We deliberately run CLIP on CPU here — some CLIP attention ops are not yet
# fully supported on MPS, and the one-time cost is paid only on the first run.
import os, sys, gc
import numpy as np, torch
from pathlib import Path

PROJECT_DIR = Path(".").resolve()
DATA_DIR    = Path(os.environ.get("BRATS_DATASET_DIR", str(PROJECT_DIR/"dataset")))
OUT_DIR     = Path(os.environ.get("BRATS_OUT_DIR",     str(PROJECT_DIR/"outputs")))
CKPT_DIR    = Path(os.environ.get("BRATS_CKPT_DIR",    str(PROJECT_DIR/"checkpoints")))
for _d in [OUT_DIR, CKPT_DIR]: _d.mkdir(parents=True, exist_ok=True)
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("BRATS_DATASET_DIR", str(DATA_DIR))

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)

# CLIP runs on CPU for compatibility — MPS CLIP support is incomplete in some PyTorch versions
clip_device = torch.device("cpu")
print(f"CLIP device : {clip_device}  (main device: {get_device()})")

EMB_PATH = OUT_DIR / "clip_text_embeddings.pt"

if EMB_PATH.exists():
    emb_dict = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded cached embeddings: {len(emb_dict)} patients")
else:
    from data_loader_utils import build_patient_split_samples
    from transformers import CLIPModel, CLIPTokenizer
    CLIP_NAME = "openai/clip-vit-base-patch32"

    train_s, val_s, test_s = build_patient_split_samples()
    patient_texts = {}
    for pid, _, _, _, text in train_s + val_s + test_s:
        patient_texts[pid] = text
    print(f"Encoding {len(patient_texts)} texts on CPU (~3 min) ...")

    clip = CLIPModel.from_pretrained(CLIP_NAME).to(clip_device).eval()
    tok  = CLIPTokenizer.from_pretrained(CLIP_NAME)
    emb_dict = {}
    with torch.no_grad():
        for pid, text in patient_texts.items():
            t   = tok([text], padding="max_length", max_length=77,
                      truncation=True, return_tensors="pt")
            out = clip.text_model(input_ids=t["input_ids"],
                                  attention_mask=t["attention_mask"])
            emb = clip.text_projection(out.pooler_output).squeeze(0)
            emb_dict[pid] = emb

    del clip; gc.collect()
    torch.save(emb_dict, EMB_PATH)
    print(f"Saved -> {EMB_PATH}")

print(f"Embedding shape: {next(iter(emb_dict.values())).shape}")
empty_cache(get_device())

