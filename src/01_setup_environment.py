"""Environment setup, device selection (MPS), seeds

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 1 — Environment Setup and Utility Helpers

This cell bootstraps the entire notebook session. It must be the **first cell run**.

**What it does:**
- Detects whether the notebook is running on **Kaggle** (`/kaggle/input` exists) or locally on Mac.
- Sets `PROJECT_DIR`, `DATA_DIR`, `OUT_DIR` (outputs), `CKPT_DIR` (model checkpoints), and `FIG_DIR` (saved figures) so every downstream cell uses consistent paths.
- Defines `get_device()` which returns `mps` (Apple Silicon GPU) → `cuda` → `cpu` in order of preference.
- Defines `mem_info()` for printing current RAM / VRAM usage — helpful for monitoring memory pressure on 16 GB unified memory.
- Adds the project directory to `sys.path` so local modules (`data_loader_utils`) can be imported.

All subsequent cells depend on the variables and helpers defined here.

## Cell 5 — Reproducibility Seeds and Device Selection

Sets global random seeds to **42** for `random`, `numpy`, and `torch`, ensuring:
- Train / val / test splits are identical across runs.
- Weight initialisation is deterministic.
- Data augmentation order is reproducible.

Also calls `get_device()` and prints the active hardware backend and current RAM usage. On Apple Silicon this will show `mps`; on Kaggle it shows `cuda`.
"""

# ── M1 Mac environment setup ─────────────────────────────────────────────────
import os, sys
from pathlib import Path

PROJECT_DIR = Path(".").resolve()   # run from the project folder
DATA_DIR    = PROJECT_DIR / "dataset"
OUT_DIR     = PROJECT_DIR / "outputs"
CKPT_DIR    = PROJECT_DIR / "checkpoints"
FIG_DIR     = OUT_DIR / "figures"

for _d in [DATA_DIR, OUT_DIR, CKPT_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ["BRATS_DATASET_DIR"] = str(DATA_DIR)
os.environ["BRATS_OUT_DIR"]     = str(OUT_DIR)
os.environ["BRATS_CKPT_DIR"]    = str(CKPT_DIR)

# ── MPS device ────────────────────────────────────────────────────────────────
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def empty_cache(dev):
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()

def mem_info(dev):
    if dev.type == "mps":
        # MPS does not expose allocated memory directly
        import psutil
        vm = psutil.virtual_memory()
        return f"RAM used {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB"
    elif dev.type == "cuda":
        return f"GPU {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated"
    return "CPU mode"

device = get_device()
print(f"Platform    : M1 Mac (Apple Silicon)")
print(f"Device      : {device}")
print(f"PyTorch     : {torch.__version__}")
print(f"MPS built   : {torch.backends.mps.is_built()}")
print(f"MPS avail   : {torch.backends.mps.is_available()}")
print(f"PROJECT_DIR : {PROJECT_DIR}")
print(f"DATA_DIR    : {DATA_DIR}  ({'OK' if DATA_DIR.exists() else 'MISSING'})")


# ── Seeds + MPS device ───────────────────────────────────────────────────────
import random, numpy as np, torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# cudnn not applicable on MPS — skip torch.backends.cudnn settings

device = get_device()   # defined in Cell 1
print(f"Active device : {device}")
print(f"Memory        : {mem_info(device)}")

FIG_DIR = __import__("pathlib").Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

