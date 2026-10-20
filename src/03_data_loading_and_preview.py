"""Patient-level split, dataset class, sample slice preview

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

---
## Step 2 — Data Loading and Patient–Text Matching

The `BraTSMultimodalDataset` class loads FLAIR slices on-the-fly from `.npy` volumes, applies per-volume z-score normalisation, collapses 4-channel one-hot masks to binary, and attaches the per-patient radiology report.

**Key design decisions:**
- Train / val / test split at the **patient level** (no data leakage across slices).
- The pre-existing `train/` folder forms the train set; the `val/` folder is split 50/50 into val and test at patient level with `seed=42`.
- Missing text files fall back to a placeholder string.

## Cell 9 — Visualise Sample Training Slices

Renders a 2 × 5 grid of randomly selected training slices to visually confirm:
- FLAIR MRI slices look like valid axial brain scans (greyscale, correct orientation and intensity).
- Ground-truth tumour contours (green outlines) are correctly aligned with the MRI — confirming patient–mask pairing is working.
- Per-volume z-score normalisation hasn't introduced visible artefacts.

Each subplot title shows the patient ID and axial slice index (0–154).
"""

import sys, importlib.util
for _p in ['/kaggle/working',
           '/kaggle/input/datasets/sudarshannn2005/brats-multimodal-code',
           str(PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data_loader_utils import (
    BraTSMultimodalDataset,
    build_patient_split_samples,
    get_dataloaders,
    get_text_for_patient,
    PLACEHOLDER_TEXT,
)

# Build sample lists
train_s, val_s, test_s = build_patient_split_samples(seed=SEED)

print(f"Train slices : {len(train_s):>7,}")
print(f"Val   slices : {len(val_s):>7,}")
print(f"Test  slices : {len(test_s):>7,}")
print(f"Total slices : {len(train_s)+len(val_s)+len(test_s):>7,}")

# Quick sanity check on one item
sample_ds = BraTSMultimodalDataset(train_s[:5], include_blank=True)
img, mask, text, meta = sample_ds[0]
print(f"\nSample item  : {meta}")
print(f"Image shape  : {img.shape}  dtype: {img.dtype}")
print(f"Mask shape   : {mask.shape}  unique: {mask.unique().tolist()}")
print(f"Text (120ch) : {text[:120]!r}")


import matplotlib.pyplot as plt
# %matplotlib inline  # notebook magic

# Visualise a handful of training slices to confirm loading is correct
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

preview_ds = BraTSMultimodalDataset(train_s, include_blank=True)
indices = np.linspace(0, len(preview_ds) - 1, 10, dtype=int)

for ax, idx in zip(axes, indices):
    img_t, mask_t, _, meta = preview_ds[idx]
    img_np  = img_t[0].numpy()
    mask_np = mask_t[0].numpy()

    img_disp = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    ax.imshow(img_disp, cmap="gray")
    if mask_np.sum() > 0:
        ax.contour(mask_np, levels=[0.5], colors="lime", linewidths=0.8)
    ax.set_title(f"{meta['patient_id'].split('_')[-1]}  sl{meta['slice_index']}", fontsize=8)
    ax.axis("off")

fig.suptitle("Sample FLAIR slices with tumor contours (green)", fontsize=12)
fig.tight_layout()
plt.show()
