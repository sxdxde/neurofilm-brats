"""Train image-only baseline (zero text embedding)

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
# IMAGE-ONLY BASELINE  (M1 Mac / MPS edition — same arch, text_emb = zeros)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os, gc, csv, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

PROJECT_DIR = Path(".").resolve()
OUT_DIR  = Path(os.environ.get("BRATS_OUT_DIR",  str(PROJECT_DIR/"outputs")))
CKPT_DIR = Path(os.environ.get("BRATS_CKPT_DIR", str(PROJECT_DIR/"checkpoints")))

SEED=42; torch.manual_seed(SEED)
device = get_device()
print(f"Device : {device}  |  {mem_info(device)}")

BATCH=8; EPOCHS=50; LR=1e-4; PATIENCE=10; MAX_NORM=1.0; LOG_EVERY=50

if "tr_dl" not in dir():
    tr_ds = TensorDataset(torch.from_numpy(X_tr).unsqueeze(1),
                          torch.from_numpy(Y_tr).unsqueeze(1),
                          torch.from_numpy(E_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va).unsqueeze(1),
                          torch.from_numpy(Y_va).unsqueeze(1),
                          torch.from_numpy(E_va))
    tr_dl = DataLoader(tr_ds,batch_size=BATCH,shuffle=True, num_workers=0,pin_memory=False)
    va_dl = DataLoader(va_ds,batch_size=BATCH,shuffle=False,num_workers=0,pin_memory=False)

def dice_loss(lg,t,s=1e-6):
    p=torch.sigmoid(lg);i=(p*t).sum((1,2,3));u=p.sum((1,2,3))+t.sum((1,2,3))
    return 1-((2*i+s)/(u+s)).mean()
def focal(lg,t,a=0.25,g=2.0):
    bce=nn.functional.binary_cross_entropy_with_logits(lg,t,reduction="none")
    return (a*(1-torch.exp(-bce))**g*bce).mean()
def combo(lg,t):
    return 0.5*dice_loss(lg,t)+0.3*nn.functional.binary_cross_entropy_with_logits(lg,t)+0.2*focal(lg,t)
def dice_sc(lg,t,s=1e-6):
    p=(torch.sigmoid(lg)>0.5).float();i=(p*t).sum((1,2,3));u=p.sum((1,2,3))+t.sum((1,2,3))
    return ((2*i+s)/(u+s)).mean().item()
def iou_sc(lg,t,s=1e-6):
    p=(torch.sigmoid(lg)>0.5).float();i=(p*t).sum((1,2,3));u=p.sum((1,2,3))+t.sum((1,2,3))-i
    return ((i+s)/(u+s)).mean().item()

ZERO_EMB = torch.zeros(BATCH, 512, device=device)

io_model = FiLMUNet(text_dim=512).to(device)
opt = optim.AdamW(io_model.parameters(), lr=LR, weight_decay=1e-4)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

ckpt_io_best   = CKPT_DIR/"best_model_image_only.pth"
ckpt_io_latest = CKPT_DIR/"latest_model_image_only.pth"
log_io         = OUT_DIR/"training_log_image_only.csv"
best=0.0; no_imp=0; start=1

def _save_ckpt_io(path, epoch, vd):
    torch.save({"epoch":epoch,"model_state":io_model.state_dict(),
                "optimizer_state":opt.state_dict(),"scheduler_state":sch.state_dict(),
                "val_dice":vd,"no_imp":no_imp}, path)

resume_path = ckpt_io_latest if ckpt_io_latest.exists() else (ckpt_io_best if ckpt_io_best.exists() else None)
if resume_path:
    c=torch.load(resume_path,map_location=device,weights_only=False)
    io_model.load_state_dict(c["model_state"]); opt.load_state_dict(c["optimizer_state"])
    sch.load_state_dict(c["scheduler_state"]); best=c["val_dice"]; no_imp=c.get("no_imp",0)
    start=c["epoch"]+1; print(f"Resumed ep {c['epoch']}  dice={best:.4f} -> ep {start}")
else:
    with open(log_io,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_dice","val_iou"])
    print("Fresh training (image-only baseline)")

for epoch in range(start, EPOCHS+1):
    t0=time.time()
    io_model.train(); tl=td=ti=0.0; skipped=0

    for bi,(imgs,masks,_) in enumerate(tr_dl):
        imgs=imgs.to(device); masks=masks.to(device)
        ze=ZERO_EMB[:imgs.size(0)]
        opt.zero_grad()
        lg=io_model(imgs,ze); loss=combo(lg,masks)
        if not torch.isfinite(loss): skipped+=1; opt.zero_grad(); continue
        loss.backward()
        nn.utils.clip_grad_norm_(io_model.parameters(),MAX_NORM)
        opt.step()
        tl+=loss.item(); td+=dice_sc(lg.detach(),masks); ti+=iou_sc(lg.detach(),masks)
        if (bi+1)%LOG_EVERY==0:
            print(f"  IO{epoch:02d} [{bi+1:4d}/{len(tr_dl)}]  loss={tl/(bi+1-skipped):.4f}  {time.time()-t0:.0f}s",flush=True)

    good=len(tr_dl)-skipped
    if good==0: print(f"Ep {epoch}: ALL NaN -- stopping"); break
    tl/=good; td/=good; ti/=good

    io_model.eval(); vl=vd=vi=0.0
    with torch.no_grad():
        for imgs,masks,_ in va_dl:
            imgs=imgs.to(device); masks=masks.to(device)
            ze=ZERO_EMB[:imgs.size(0)]; lg=io_model(imgs,ze)
            if torch.isfinite(combo(lg,masks)):
                vl+=combo(lg,masks).item(); vd+=dice_sc(lg,masks); vi+=iou_sc(lg,masks)
    n=len(va_dl); vl/=n; vd/=n; vi/=n
    sch.step()
    elapsed=time.time()-t0
    with open(log_io,"a",newline="") as f:
        csv.writer(f).writerow([epoch,f"{tl:.6f}",f"{vl:.6f}",f"{vd:.6f}",f"{vi:.6f}"])
    print(f"IO {epoch:03d}/{EPOCHS}  train={tl:.4f}  val={vl:.4f}  dice={vd:.4f}  iou={vi:.4f}  ({elapsed:.0f}s)")
    _save_ckpt_io(ckpt_io_latest,epoch,vd)
    if vd>best: best=vd;no_imp=0;_save_ckpt_io(ckpt_io_best,epoch,vd);print(f"  ✓ best (dice={vd:.4f})")
    else:
        no_imp+=1
        if no_imp>=PATIENCE: print(f"Early stop ep {epoch}"); break
    empty_cache(device)

print(f"\nImage-only done. Best dice={best:.4f}")
del opt,sch,io_model; gc.collect(); empty_cache(device)

