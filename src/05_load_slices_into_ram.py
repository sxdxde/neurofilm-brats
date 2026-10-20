"""Load all slices into RAM for fast MPS training

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 11 — Load All Slices into RAM (In-Memory Dataset)

Loads every NIfTI volume from disk and flattens it into six numpy arrays held entirely in RAM:

| Array | Shape | Contents |
|---|---|---|
| `X_tr / X_va / X_te` | `[N, H, W]` float32 | Normalised FLAIR slices |
| `Y_tr / Y_va / Y_te` | `[N, H, W]` uint8 | Binary tumour masks |
| `E_tr / E_va / E_te` | `[N, 512]` float32 | CLIP text embeddings |

**Why load into RAM?**
On MPS, disk I/O during training is the dominant bottleneck. Loading everything upfront eliminates it entirely. Total footprint is ~5–6 GB on 16 GB unified memory — within budget including model weights and OS.

Set `INCLUDE_BLANK=False` to skip tumour-free slices (~60% of total) if memory is tight.
"""

# ── Step B: Extract all slices into RAM ──────────────────────────────────────
# Memory budget on 16 GB unified memory (M1 Pro):
#   Train ~33K slices x (img 64KB + mask 64KB + emb 2KB) = ~4.3 GB
#   Val   + Test ~11K slices                              = ~1.4 GB
#   OS + browser + Python overhead                        = ~3-4 GB
#   Model + optimizer                                     = ~0.5 GB
#   Available headroom                                    = ~5 GB  [OK]
#
# If you hit memory pressure, set INCLUDE_BLANK=False to skip empty slices
# (reduces train set by ~60%) or reduce IMG_SIZE to 64.
import numpy as np, torch, gc, os, sys
from pathlib import Path
from tqdm.notebook import tqdm

PROJECT_DIR = Path(".").resolve()
DATA_DIR    = Path(os.environ.get("BRATS_DATASET_DIR", str(PROJECT_DIR/"dataset")))
OUT_DIR     = Path(os.environ.get("BRATS_OUT_DIR",     str(PROJECT_DIR/"outputs")))
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("BRATS_DATASET_DIR", str(DATA_DIR))

from data_loader_utils import build_patient_split_samples, BraTSMultimodalDataset

INCLUDE_BLANK = True   # set False to load only tumour-containing slices (~40% of total)

def _load_eff(self, path):
    if path not in self._volume_cache:
        a = np.load(path)
        self._volume_cache[path] = a.astype(np.uint8 if a.ndim==4 else np.float32)
    return self._volume_cache[path]
BraTSMultimodalDataset._load_volume = _load_eff

def build_split(samples, name):
    ds  = BraTSMultimodalDataset(samples, include_blank=INCLUDE_BLANK)
    n   = len(ds)
    X   = np.empty((n, 128, 128), dtype=np.float32)
    Y   = np.empty((n, 128, 128), dtype=np.float32)
    E   = np.empty((n, 512),      dtype=np.float32)
    PID = []
    for i, (img, mask, _, meta) in enumerate(tqdm(ds, desc=name, leave=False)):
        X[i]   = img[0].numpy()
        Y[i]   = mask[0].numpy()
        pid    = int(meta["patient_id"].split("_")[-1])
        E[i]   = emb_dict.get(pid, torch.zeros(512)).numpy()
        PID.append(pid)
    del ds; gc.collect()
    print(f"{name}: {n} slices  X={X.nbytes/1e9:.2f} GB  Y={Y.nbytes/1e9:.3f} GB")
    return X, Y, E, PID

# Load emb_dict from cache if not already in memory (e.g. when Step A was skipped)
if "emb_dict" not in dir():
    EMB_PATH = OUT_DIR / "clip_text_embeddings.pt"
    assert EMB_PATH.exists(), f"Run the CLIP embeddings cell first — {EMB_PATH} not found"
    emb_dict = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded cached embeddings: {len(emb_dict)} patients")

train_s, val_s, test_s = build_patient_split_samples()
X_tr, Y_tr, E_tr, PID_tr = build_split(train_s, "train")
X_va, Y_va, E_va, PID_va = build_split(val_s,   "val  ")
X_te, Y_te, E_te, PID_te = build_split(test_s,  "test ")

total_gb = sum(a.nbytes for a in [X_tr,Y_tr,E_tr,X_va,Y_va,E_va,X_te,Y_te,E_te])/1e9
print(f"\nAll slices in RAM. Total: {total_gb:.2f} GB")
print(f"Splits -- train:{len(X_tr)}  val:{len(X_va)}  test:{len(X_te)}")

import psutil; vm=psutil.virtual_memory()
print(f"System RAM: {vm.used/1e9:.1f} used / {vm.total/1e9:.1f} total GB")
if vm.available < 3e9:
    print("WARNING: less than 3 GB RAM available. Consider setting INCLUDE_BLANK=False.")

