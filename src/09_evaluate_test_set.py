"""Training log preview and test-set evaluation

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 15 — Training Curves (Quick Inline Preview)

Reads `training_log.csv` and `training_log_image_only.csv` and renders a quick 3-panel inline plot (Dice, IoU, Loss vs Epoch) for both models.

This is a fast visual health check during or immediately after training — useful for spotting:
- Loss divergence or NaN epochs.
- Overfitting (train Dice rising while val Dice falls).
- Early-stop triggers (where the curves flatten).

For the publication-quality saved figures, see the visualisation cells below.

## Cell 16 — Evaluation on the Held-Out Test Set

Evaluates both trained models on the **test split only** (never seen during training or validation).

### Metrics computed per slice, then aggregated (mean ± std)

| Metric | What it measures |
|---|---|
| **Dice coefficient** | Overlap between predicted and ground-truth mask; primary metric |
| **IoU (Jaccard index)** | Intersection / Union; stricter than Dice |
| **Hausdorff distance** | Maximum boundary error in pixels; penalises outlier predictions |
| **Precision** | Of all predicted tumour pixels, how many are correct |
| **Recall** | Of all true tumour pixels, how many were found |

Results are saved to `outputs/results_table.csv` — one row per test slice plus aggregate summary rows.

The evaluation loads directly from checkpoints so it can be re-run without retraining.
"""

import pandas as pd
from pathlib import Path
OUT_DIR = Path("/kaggle/working/outputs") if __import__("os").path.exists("/kaggle/input") else Path(".").resolve()/"outputs"
for label, name in [("Multimodal","training_log.csv"),("Image-Only","training_log_image_only.csv")]:
    p = OUT_DIR/name
    if p.exists():
        df=pd.read_csv(p); print(f"{label} — last 5 epochs:"); display(df.tail())
    else: print(f"{label}: not found yet")


# ── Inline evaluation on test set ────────────────────────────────────────────
import os, sys, gc, csv
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm as nb_tqdm
from scipy.ndimage import label as ndlabel
from skimage.metrics import hausdorff_distance as skimage_hd

ON_KAGGLE   = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working")             if ON_KAGGLE else Path(".").resolve()
OUT_DIR     = Path("/kaggle/working/outputs")     if ON_KAGGLE else PROJECT_DIR/"outputs"
CKPT_DIR    = Path("/kaggle/working/checkpoints") if ON_KAGGLE else PROJECT_DIR/"checkpoints"
for _p in [str(PROJECT_DIR), "/kaggle/input/datasets/sudarshannn2005/brats-multimodal-code"]:
    if _p not in sys.path: sys.path.insert(0, _p)

device = get_device()   # MPS / CPU

te_ds = TensorDataset(torch.from_numpy(X_te).unsqueeze(1),
                      torch.from_numpy(Y_te).unsqueeze(1),
                      torch.from_numpy(E_te))
te_dl = DataLoader(te_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

def eval_model(ckpt_path, label, use_text=True):
    model = FiLMUNet(512).to(device)
    if ckpt_path.exists():
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c["model_state"])
        print(f"Loaded {label} checkpoint (epoch {c['epoch']}, dice={c['val_dice']:.4f})")
    else:
        print(f"WARNING: {ckpt_path} not found — using random weights")
    model.eval()

    dices, ious, hds, precs, recs = [], [], [], [], []
    ZERO = torch.zeros(8, 512, device=device)
    s=1e-6
    with torch.no_grad():
        for imgs, masks, embs in nb_tqdm(te_dl, desc=label, leave=False):
            imgs=imgs.to(device); masks=masks.to(device)
            emb_in = embs.to(device) if use_text else ZERO[:imgs.size(0)]
            # No AMP on MPS
            lg = model(imgs, emb_in)
            preds = (torch.sigmoid(lg) > 0.5).float()
            for i in range(len(imgs)):
                p = preds[i,0].cpu().numpy(); m = masks[i,0].cpu().numpy()
                inter=(p*m).sum(); union=p.sum()+m.sum()
                dices.append((2*inter+s)/(union+s))
                ious.append((inter+s)/(union-inter+s))
                precs.append((inter+s)/(p.sum()+s))
                recs.append((inter+s)/(m.sum()+s))
                if p.sum()==0 and m.sum()==0: hds.append(0.0)
                elif p.sum()==0 or m.sum()==0: hds.append(np.inf)
                else: hds.append(float(skimage_hd(p.astype(bool),m.astype(bool))))
    del model; gc.collect(); empty_cache(device)

    finite_hd = [h for h in hds if np.isfinite(h)]
    results = {"dice": dices, "iou": ious, "hausdorff": hds,
               "precision": precs, "recall": recs}
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    print(f"  Dice      : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  IoU       : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"  Hausdorff : {np.mean(finite_hd):.4f} ± {np.std(finite_hd):.4f}")
    print(f"  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    d=np.array(dices)
    print(f"  Dice>0.8  : {(d>0.8).mean()*100:.1f}%  |  0.5-0.8: {((d>=0.5)&(d<=0.8)).mean()*100:.1f}%  |  <0.5: {(d<0.5).mean()*100:.1f}%")
    return results

mm_results = eval_model(CKPT_DIR/"best_model.pth",           "Multimodal",  use_text=True)
io_results = eval_model(CKPT_DIR/"best_model_image_only.pth", "Image-Only",  use_text=False)

# Save results CSV
import csv as _csv
with open(OUT_DIR/"results_table.csv","w",newline="") as f:
    w=_csv.DictWriter(f,["model","dice","iou","hausdorff","precision","recall"]); w.writeheader()
    for label,res in [("multimodal",mm_results),("image_only",io_results)]:
        for i in range(len(res["dice"])):
            w.writerow({"model":label,"dice":res["dice"][i],"iou":res["iou"][i],
                        "hausdorff":res["hausdorff"][i],"precision":res["precision"][i],
                        "recall":res["recall"][i]})
print(f"\nSaved results_table.csv")

