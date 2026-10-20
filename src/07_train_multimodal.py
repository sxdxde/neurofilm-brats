"""Train multimodal model (FiLM + CLIP)

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 13 — Train Multimodal Model (FiLM U-Net + CLIP Text Guidance)

Trains the FiLM-conditioned U-Net using pre-computed CLIP text embeddings as the conditioning signal.

### Loss function (implemented from scratch)
```
L = 0.5 × DiceLoss + 0.3 × BCEWithLogitsLoss + 0.2 × FocalLoss(α=0.25, γ=2)
```
- **Dice loss** directly optimises the segmentation metric of interest.
- **BCE** provides stable gradients at all prediction confidence levels.
- **Focal loss** down-weights easy background pixels and focuses learning on hard tumour boundaries.

### Training configuration
| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW (weight decay 1e-4) |
| Learning rate | 1e-4 |
| Batch size | 8 (MPS memory budget) |
| Epochs | 50 max |
| Early stopping patience | 10 epochs |
| Scheduler | CosineAnnealingLR |
| Gradient clip | max norm 1.0 |
| AMP/GradScaler | **Disabled** — MPS does not support it |

### Checkpoint logic
If `checkpoints/best_model.pth` already exists **and its architecture keys match**, training is skipped and the checkpoint is loaded. Delete the checkpoint file to force retraining.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTIMODAL TRAINING  (M1 Mac / MPS edition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Key M1 adaptations vs Kaggle version:
#   - device     : MPS (Apple Silicon GPU)
#   - BATCH      : 8  (16 GB unified RAM, conservative)
#   - AMP/Scaler : DISABLED  (MPS does not support GradScaler)
#   - DataParallel: DISABLED  (single unified GPU)
#   - num_workers : 0  (avoid macOS multiprocessing issues)
#   - cudnn flags : REMOVED   (not applicable on MPS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os, gc, csv, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

PROJECT_DIR = Path(".").resolve()
OUT_DIR     = Path(os.environ.get("BRATS_OUT_DIR",  str(PROJECT_DIR/"outputs")))
CKPT_DIR    = Path(os.environ.get("BRATS_CKPT_DIR", str(PROJECT_DIR/"checkpoints")))
for _d in [OUT_DIR, CKPT_DIR]: _d.mkdir(parents=True, exist_ok=True)

SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)

device = get_device()   # MPS / CPU — defined in Cell 1
print(f"Device : {device}  |  {mem_info(device)}")

# ── Hyper-parameters (Mac-tuned) ──────────────────────────────────────────────
BATCH     = 8       # reduced from 32: 16 GB unified memory is shared CPU+GPU
EPOCHS    = 50      # set to 20 for a quick trial on Mac
LR        = 1e-4
PATIENCE  = 10
MAX_NORM  = 1.0
LOG_EVERY = 50      # print more frequently (smaller batches)

# ── DataLoaders ────────────────────────────────────────────────────────────────
tr_ds = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1),
                      torch.from_numpy(Y_tr).unsqueeze(1),
                      torch.from_numpy(E_tr))
va_ds = TensorDataset(torch.from_numpy(X_va).unsqueeze(1),
                      torch.from_numpy(Y_va).unsqueeze(1),
                      torch.from_numpy(E_va))
# num_workers=0 required on macOS (fork issues with MPS context)
# pin_memory=False — not useful for MPS (unified memory)
tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
print(f"Train: {len(tr_dl)} batches  Val: {len(va_dl)} batches  (batch={BATCH})")

# ── Loss + metrics ─────────────────────────────────────────────────────────────
def dice_loss(lg, t, s=1e-6):
    p=torch.sigmoid(lg); i=(p*t).sum((1,2,3)); u=p.sum((1,2,3))+t.sum((1,2,3))
    return 1-((2*i+s)/(u+s)).mean()
def focal(lg, t, a=0.25, g=2.0):
    bce=nn.functional.binary_cross_entropy_with_logits(lg,t,reduction="none")
    return (a*(1-torch.exp(-bce))**g*bce).mean()
def combo(lg, t):
    return 0.5*dice_loss(lg,t)+0.3*nn.functional.binary_cross_entropy_with_logits(lg,t)+0.2*focal(lg,t)
def dice_sc(lg, t, s=1e-6):
    p=(torch.sigmoid(lg)>0.5).float(); i=(p*t).sum((1,2,3)); u=p.sum((1,2,3))+t.sum((1,2,3))
    return ((2*i+s)/(u+s)).mean().item()
def iou_sc(lg, t, s=1e-6):
    p=(torch.sigmoid(lg)>0.5).float(); i=(p*t).sum((1,2,3)); u=p.sum((1,2,3))+t.sum((1,2,3))-i
    return ((i+s)/(u+s)).mean().item()

# ── Model (no DataParallel on Mac) ────────────────────────────────────────────
mm_model = FiLMUNet(text_dim=512).to(device)
# Single MPS device — DataParallel not used
opt = optim.AdamW(mm_model.parameters(), lr=LR, weight_decay=1e-4)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
# AMP/GradScaler DISABLED: MPS does not support GradScaler
# Training in full float32 — MPS is efficient enough without AMP

ckpt_best   = CKPT_DIR / "best_model.pth"
ckpt_latest = CKPT_DIR / "latest_model.pth"
log_path    = OUT_DIR  / "training_log.csv"
best=0.0; no_imp=0; start=1

def _save_ckpt(path, epoch, vd):
    torch.save({
        "epoch":          epoch,
        "model_state":    mm_model.state_dict(),
        "optimizer_state":opt.state_dict(),
        "scheduler_state":sch.state_dict(),
        "val_dice":        vd,
        "no_imp":          no_imp,
    }, path)

resume_path = ckpt_latest if ckpt_latest.exists() else (ckpt_best if ckpt_best.exists() else None)
if resume_path:
    c = torch.load(resume_path, map_location=device, weights_only=False)
    mm_model.load_state_dict(c["model_state"])
    opt.load_state_dict(c["optimizer_state"])
    sch.load_state_dict(c["scheduler_state"])
    best   = c["val_dice"]
    no_imp = c.get("no_imp", 0)
    start  = c["epoch"] + 1
    print(f"Resumed ep {c['epoch']}  dice={best:.4f}  no_imp={no_imp} -> ep {start}")
else:
    with open(log_path,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_dice","val_iou"])
    print(f"Fresh training -- {sum(p.numel() for p in mm_model.parameters()):,} params")

for epoch in range(start, EPOCHS+1):
    t0 = time.time()
    mm_model.train(); tl=td=ti=0.0; skipped=0

    for bi, (imgs, masks, embs) in enumerate(tr_dl):
        imgs  = imgs.to(device)
        masks = masks.to(device)
        embs  = embs.to(device)
        opt.zero_grad()

        # No AMP autocast — run in float32 on MPS
        lg   = mm_model(imgs, embs)
        loss = combo(lg, masks)

        if not torch.isfinite(loss):
            skipped += 1; opt.zero_grad(); continue

        loss.backward()
        nn.utils.clip_grad_norm_(mm_model.parameters(), MAX_NORM)
        opt.step()

        tl += loss.item()
        td += dice_sc(lg.detach(), masks)
        ti += iou_sc(lg.detach(), masks)

        if (bi+1) % LOG_EVERY == 0:
            print(f"  Ep{epoch:02d} [{bi+1:4d}/{len(tr_dl)}]  "
                  f"loss={tl/(bi+1-skipped):.4f}  {time.time()-t0:.0f}s", flush=True)

    good = len(tr_dl) - skipped
    if good == 0: print(f"Ep {epoch}: ALL batches NaN — stopping"); break
    tl/=good; td/=good; ti/=good

    mm_model.eval(); vl=vd=vi=0.0
    with torch.no_grad():
        for imgs, masks, embs in va_dl:
            imgs=imgs.to(device); masks=masks.to(device); embs=embs.to(device)
            lg = mm_model(imgs, embs)
            if torch.isfinite(combo(lg, masks)):
                vl += combo(lg, masks).item()
                vd += dice_sc(lg, masks)
                vi += iou_sc(lg, masks)
    n=len(va_dl); vl/=n; vd/=n; vi/=n
    sch.step()

    elapsed = time.time()-t0
    with open(log_path,"a",newline="") as f:
        csv.writer(f).writerow([epoch,f"{tl:.6f}",f"{vl:.6f}",f"{vd:.6f}",f"{vi:.6f}"])
    print(f"Ep {epoch:03d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}  "
          f"dice={vd:.4f}  iou={vi:.4f}  ({elapsed:.0f}s)")

    _save_ckpt(ckpt_latest, epoch, vd)
    if vd > best:
        best=vd; no_imp=0
        _save_ckpt(ckpt_best, epoch, vd)
        print(f"  ✓ best saved (dice={vd:.4f})")
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"Early stop ep {epoch}"); break

    empty_cache(device)

print(f"\nMultimodal done. Best dice={best:.4f}")
del opt, sch, mm_model; gc.collect(); empty_cache(device)

