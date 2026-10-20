"""Dataset path validation, structure scan and summary

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 3 — Dataset Path Validation

A quick guard that confirms the expected dataset directory exists before any heavy data loading begins.

**Checks performed:**
- Verifies `DATA_DIR` (e.g., `./dataset/`) is present on disk.
- Verifies the `MICCAI_BraTS2020_TrainingData` sub-folder exists inside it.
- Prints a clear ✓ / ✗ status so you know immediately if the data path is wrong.

If this cell prints a warning, fix `DATA_DIR` in Cell 1 before proceeding. Running data-loading cells without valid data causes cryptic NIfTI read errors far downstream.

## Cell 4 — Dataset Structure Scan and Summary

Recursively walks `DATA_DIR` and builds a complete inventory of the dataset.

**What it produces:**
- Count of FLAIR volumes, segmentation masks, and radiology text files found.
- A patient-level **matching report**: flags any patient that has an image but no text file (or vice versa).
- Saves `outputs/dataset_summary.json` with all counts so later cells can reference it.

**Why this matters:** The model uses one text description per patient across all 155 axial slices. Any patient-text mismatch would silently propagate wrong labels through training.

## Cell 6 — Load Data Loader Module

Dynamically imports `data_loader_utils` — a helper module that provides:
- `build_patient_split_samples()` — splits 369 patients into 70% train / 15% val / 15% test **at the patient level** (not slice level), preventing data leakage.
- `BraTSMultimodalDataset` — a PyTorch `Dataset` that loads NIfTI slices on demand, normalises them (per-volume z-score), resizes to 128×128, and pairs each slice with its patient's text embedding.

The dynamic `sys.path` insertion allows the same notebook cell to find the module whether running locally or on Kaggle without any code changes.

## Cell 7 — Display Dataset Summary

Reads and pretty-prints the `dataset_summary.json` saved by Cell 4.

Reported fields:
- **Total patients**: expected 369.
- **FLAIR / mask / text counts**: should all match.
- **Mismatches**: any patient ID without a paired text file.

This is a quick human-readable confirmation of data integrity before model training.
"""

# ── Validate local dataset ────────────────────────────────────────────────────
# No download needed — data should already be in ./dataset/
import os
from pathlib import Path

DATA_DIR  = Path(os.environ.get("BRATS_DATASET_DIR", "dataset"))
FLAIR_DIR = DATA_DIR / "FLAIR_BRATS2020_split"
TEXT_DIR  = DATA_DIR / "TextBRats" / "TextBraTSData"

def print_tree(root, prefix="", depth=3, max_entries=5):
    root = Path(root)
    if not root.exists(): print(f"{prefix}[MISSING] {root}"); return
    entries = sorted(root.iterdir())
    for i, e in enumerate(entries[:max_entries]):
        last = (i == min(len(entries), max_entries)-1)
        print(f"{prefix}{'`-- ' if last else '+-- '}{e.name}{'/' if e.is_dir() else ''}")
        if e.is_dir() and depth > 1:
            print_tree(e, prefix+("    " if last else "|   "), depth-1, max_entries)
    if len(entries) > max_entries:
        print(f"{prefix}    ... ({len(entries)-max_entries} more)")

print(f"Dataset root: {DATA_DIR}")
print_tree(DATA_DIR, depth=2)

print(f"\nFLAIR split : {'OK' if FLAIR_DIR.exists() else 'MISSING'}")
print(f"Text data   : {'OK' if TEXT_DIR.exists()  else 'MISSING'}")

if not FLAIR_DIR.exists() or not TEXT_DIR.exists():
    print("\nExpected layout:")
    print("  dataset/FLAIR_BRATS2020_split/train/images/image_0.npy ...")
    print("  dataset/TextBRats/TextBraTSData/BraTS20_Training_001/...")
    raise FileNotFoundError("Dataset not found. Check DATA_DIR path above.")


# ── Validate dataset structure ────────────────────────────────────────────────
import os
from pathlib import Path

def print_tree(root, prefix="", depth=3, max_entries=8):
    root = Path(root)
    if not root.exists():
        print(f"{prefix}[MISSING] {root}")
        return
    entries = sorted(root.iterdir())
    shown = entries[:max_entries]
    for i, entry in enumerate(shown):
        is_last = (i == len(shown) - 1) and len(entries) <= max_entries
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir() and depth > 1:
            ext = "    " if is_last else "│   "
            print_tree(entry, prefix + ext, depth - 1, max_entries)
    if len(entries) > max_entries:
        print(f"{prefix}    ... ({len(entries) - max_entries} more entries)")

data_root = Path(os.environ.get("BRATS_DATASET_DIR", str(PROJECT_DIR / "dataset")))
print(f"Dataset root: {data_root}\n")
print_tree(data_root, depth=3)
print()

flair_dir = data_root / "FLAIR_BRATS2020_split"
text_dir  = data_root / "TextBRats" / "TextBraTSData"

print(f"FLAIR split dir : {'✓ OK' if flair_dir.exists() else '✗ MISSING'}  ({flair_dir})")
print(f"Text data dir   : {'✓ OK' if text_dir.exists() else '✗ MISSING'}  ({text_dir})")

# If text dir not found at expected path, search for it
if not text_dir.exists():
    candidates = list(data_root.rglob("*TextBraTS*")) + list(data_root.rglob("*flair_text*"))
    if candidates:
        print("\nFound possible text data at:")
        for c in candidates[:5]:
            print(f"  {c}")
        print("\nUpdate TEXT_DIR in 02_data_loader.py if the path differs.")

# File counts
if flair_dir.exists():
    train_imgs = list((flair_dir / "train" / "images").glob("*.npy"))
    val_imgs   = list((flair_dir / "val"   / "images").glob("*.npy"))
    print(f"\nTrain volumes : {len(train_imgs)}")
    print(f"Val   volumes : {len(val_imgs)}")

if text_dir.exists():
    txt_files = list(text_dir.rglob("*.txt"))
    print(f"Text reports  : {len(txt_files)}")

if flair_dir.exists() and text_dir.exists():
    print("\nDataset structure looks correct — ready to proceed.")
else:
    print("\nWARNING: Some paths are missing. Check the tree above and update "
          "FLAIR_DIR / TEXT_DIR in 02_data_loader.py if the zip extracted "
          "to a different folder name.")


import sys, importlib.util
for _p in ['/kaggle/working',
           '/kaggle/input/datasets/sudarshannn2005/brats-multimodal-code',
           str(PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import importlib.util

def run_script(path):
    """Execute a .py file as __main__ and return its module object."""
    spec = importlib.util.spec_from_file_location("__main__", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Resolve script location — working dir first, then code input dataset
_script = PROJECT_DIR / "01_explore_dataset.py"
if not _script.exists():
    _script = Path("/kaggle/input/datasets/sudarshannn2005/brats-multimodal-code/01_explore_dataset.py")

run_script(_script)


# Print the saved summary JSON
import json

summary_path = PROJECT_DIR / "outputs" / "dataset_summary.json"
with open(summary_path) as f:
    summary = json.load(f)

print(json.dumps(summary, indent=2))
