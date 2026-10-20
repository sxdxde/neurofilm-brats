"""Dice distribution and failure cases

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 20 — 6.6 Error Analysis: Dice Distribution and Failure Cases

**Deliverable 6.6 — Error Analysis**

### Part A — Dice Score Distribution (`fig_dice_distribution.png`)
Histogram of per-slice Dice scores across the entire test set with threshold lines at 0.5 (acceptable) and 0.8 (good). Lets you read off what fraction of predictions fall into poor / acceptable / excellent bands.

### Part B — Failure Cases (`fig_failure_cases.png`)
Identifies the **10 worst-performing test slices** (lowest Dice among tumour-containing slices) and renders the same 4-panel grid as in Cell 18.

Common failure patterns to look for:
- Very small or diffuse tumour regions (< 20 pixels).
- Boundary slices at the top/bottom of the tumour volume.
- Slices where FLAIR contrast between tumour and surrounding tissue is low.

These failure cases provide qualitative evidence for the discussion section of the report.
"""

# ── Dice distribution + failure cases ────────────────────────────────────────
import matplotlib.pyplot as plt, numpy as np, torch, os
from pathlib import Path

ON_KAGGLE = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(".").resolve()
OUT_DIR  = Path("/kaggle/working/outputs")  if ON_KAGGLE else PROJECT_DIR/"outputs"
CKPT_DIR = Path("/kaggle/working/checkpoints") if ON_KAGGLE else PROJECT_DIR/"checkpoints"
FIG_DIR  = OUT_DIR/"figures"; FIG_DIR.mkdir(exist_ok=True)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "mm_results" in dir():
    dices = np.array(mm_results["dice"])
    fig, ax = plt.subplots(figsize=(9,4))
    ax.hist(dices, bins=50, edgecolor="white", color="steelblue", alpha=0.85)
    ax.axvline(0.5, color="orange", ls="--", label="0.5 threshold")
    ax.axvline(0.8, color="green",  ls="--", label="0.8 threshold")
    ax.set_xlabel("Dice Score"); ax.set_ylabel("Slice count")
    ax.set_title("Per-slice Dice Distribution (test set)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR/"fig_dice_distribution.png",dpi=150,bbox_inches="tight")
    plt.show()

    # Failure cases — 10 worst slices with tumor present
    has_tumor = np.where(Y_te.sum(axis=(1,2)) > 50)[0]
    tumor_dices = dices[has_tumor]
    worst_local = np.argsort(tumor_dices)[:10]
    worst_idx   = has_tumor[worst_local]

    model = FiLMUNet(512).to(device)
    ckpt  = torch.load(CKPT_DIR/"best_model.pth",map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    fig, axes = plt.subplots(10, 4, figsize=(14, 30))
    with torch.no_grad():
        for row, idx in enumerate(worst_idx):
            img = torch.from_numpy(X_te[idx]).unsqueeze(0).unsqueeze(0).to(device)
            emb = torch.from_numpy(E_te[idx]).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                pred = (torch.sigmoid(model(img,emb))>0.5).float()
            disp = X_te[idx]; disp=(disp-disp.min())/(disp.max()-disp.min()+1e-8)
            gt_np=Y_te[idx]; pr_np=pred[0,0].cpu().numpy()
            for col,(im,title) in enumerate([(disp,"Input"),(disp,"GT"),(disp,"Pred"),(disp,"Overlay")]):
                axes[row,col].imshow(im,cmap="gray"); axes[row,col].axis("off")
                if col==1: axes[row,col].contourf(gt_np,[0.5,1.5],colors=["#00ff00"],alpha=0.4)
                if col==2: axes[row,col].contourf(pr_np,[0.5,1.5],colors=["#ff0000"],alpha=0.4)
                if col==3:
                    axes[row,col].contourf(gt_np,[0.5,1.5],colors=["#00ff00"],alpha=0.3)
                    axes[row,col].contourf(pr_np,[0.5,1.5],colors=["#ff0000"],alpha=0.3)
                if row==0: axes[row,col].set_title(title)
            axes[row,0].set_ylabel(f"Dice={dices[idx]:.3f}", fontsize=8)
    plt.suptitle("Top 10 Failure Cases",fontsize=12); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig_failure_cases.png",dpi=120,bbox_inches="tight"); plt.show()
    del model; torch.cuda.empty_cache()

