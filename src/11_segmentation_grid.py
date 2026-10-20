"""Segmentation results grid

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 18 — 6.2 Segmentation Results Grid

**Deliverable 6.2 — Segmentation Results**

Selects 10 random test slices that contain tumour tissue (mask area > 50 pixels) and renders a **4-panel visualisation** for each:

| Panel | Content |
|---|---|
| **Input** | Normalised FLAIR MRI slice (greyscale) |
| **Ground Truth** | Green tumour contour overlaid on MRI |
| **Prediction** | Red predicted tumour mask overlaid on MRI |
| **Overlay** | Green = GT only · Red = Pred only · Yellow = overlap |

The overlay panel is the most diagnostic: yellow pixels show correct predictions, red shows false positives, green shows missed tumour (false negatives).

Saved to `outputs/figures/fig_segmentation_grid.png`.
"""

# ── Segmentation grid — 10 random test samples ────────────────────────────────
import matplotlib.pyplot as plt, numpy as np, torch, os
from pathlib import Path

ON_KAGGLE = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(".").resolve()
OUT_DIR = Path("/kaggle/working/outputs") if ON_KAGGLE else PROJECT_DIR/"outputs"
CKPT_DIR = Path("/kaggle/working/checkpoints") if ON_KAGGLE else PROJECT_DIR/"checkpoints"
FIG_DIR = OUT_DIR/"figures"; FIG_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FiLMUNet(512).to(device)
ckpt = torch.load(CKPT_DIR/"best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.eval()

# Pick 10 slices with non-trivial masks
has_tumor = np.where(Y_te.sum(axis=(1,2)) > 50)[0]
idxs = np.random.RandomState(42).choice(has_tumor, min(10, len(has_tumor)), replace=False)

fig, axes = plt.subplots(len(idxs), 4, figsize=(14, 3*len(idxs)))
if len(idxs)==1: axes = axes[np.newaxis]
with torch.no_grad():
    for row, idx in enumerate(idxs):
        img = torch.from_numpy(X_te[idx]).unsqueeze(0).unsqueeze(0).to(device)
        emb = torch.from_numpy(E_te[idx]).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
            pred = (torch.sigmoid(model(img, emb)) > 0.5).float()
        img_np  = X_te[idx]; gt_np = Y_te[idx]; pr_np = pred[0,0].cpu().numpy()
        disp    = (img_np-img_np.min())/(img_np.max()-img_np.min()+1e-8)
        for col, (im, title) in enumerate([
            (disp, "Input FLAIR"),
            (disp, "Ground Truth"),
            (disp, "Prediction"),
            (disp, "Overlay")]):
            axes[row,col].imshow(im, cmap="gray"); axes[row,col].axis("off")
            if col==1: axes[row,col].contourf(gt_np,[0.5,1.5],colors=["#00ff00"],alpha=0.4)
            if col==2: axes[row,col].contourf(pr_np,[0.5,1.5],colors=["#ff0000"],alpha=0.4)
            if col==3:
                axes[row,col].contourf(gt_np,[0.5,1.5],colors=["#00ff00"],alpha=0.3)
                axes[row,col].contourf(pr_np,[0.5,1.5],colors=["#ff0000"],alpha=0.3)
            if row==0: axes[row,col].set_title(title)
plt.suptitle("Test Segmentation Results (green=GT  red=Pred)", fontsize=12)
plt.tight_layout(); plt.savefig(FIG_DIR/"fig_segmentation_grid.png",dpi=150,bbox_inches="tight")
plt.show()
del model; torch.cuda.empty_cache()

