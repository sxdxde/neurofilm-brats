"""Training curves

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 17 — 6.1 Training Performance Curves (Saved to File)

**Deliverable 6.1 — Training Performance**

Generates three publication-quality figures comparing the multimodal and image-only models across all training epochs:

1. **Dice Score vs Epoch** — primary segmentation metric trajectory for both models.
2. **IoU vs Epoch** — Jaccard index trajectory, a stricter version of Dice.
3. **Loss vs Epoch** — combined training loss (Dice + BCE + Focal) showing convergence.

Each figure shows both train and validation curves, with a legend distinguishing multimodal (blue) from image-only (orange).

Saved to `outputs/figures/fig_training_curves.png` and individual files `fig1_dice_vs_epoch.png`, `fig2_iou_vs_epoch.png`, `fig3_loss_vs_epoch.png`.
"""

# ── Training curves ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os
from pathlib import Path
# %matplotlib inline  # notebook magic

ON_KAGGLE = os.path.exists("/kaggle/input")
OUT_DIR   = Path("/kaggle/working/outputs") if ON_KAGGLE else Path(".").resolve()/"outputs"
FIG_DIR   = OUT_DIR/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_log(name):
    p = OUT_DIR/name
    return pd.read_csv(p) if p.exists() else None

df_mm = load_log("training_log.csv")
df_io = load_log("training_log_image_only.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for df, label, style in [(df_mm,"Multimodal","-"), (df_io,"Image-Only","--")]:
    if df is None: continue
    axes[0].plot(df.epoch, df.val_dice,  style, label=label)
    axes[1].plot(df.epoch, df.val_iou,   style, label=label)
    axes[2].plot(df.epoch, df.train_loss,style, label=f"{label} train")
    axes[2].plot(df.epoch, df.val_loss,  style, alpha=0.5, label=f"{label} val")
for ax, title, ylabel in zip(axes, ["Dice","IoU","Loss"], ["Dice","IoU","Loss"]):
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle("Training Curves — Multimodal vs Image-Only", fontsize=13)
plt.tight_layout(); plt.savefig(FIG_DIR/"fig_training_curves.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved fig_training_curves.png")

