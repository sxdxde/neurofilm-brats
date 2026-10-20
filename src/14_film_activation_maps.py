"""Explainability: FiLM activation heatmaps

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — 6.3 Attention / Explainability: FiLM Activation Heatmaps

**Deliverable 6.3 — Attention and Explainability**

Visualises **where the model focuses** and **how text guidance changes that focus** for 5 tumour-containing test slices.

| Column | What it shows |
|---|---|
| **Input MRI** | Normalised FLAIR slice (greyscale) |
| **Ground Truth** | Green tumour boundary contour |
| **MM Activation** | Mean `e4` encoder activation map with text conditioning (hot colourmap overlay) |
| **IO Activation** | Same map with zero text embedding — image-only mode |
| **Text Contribution** | `|pred_multimodal − pred_imageonly|` — pixels where text changed the output |

The last column is most diagnostic: bright pixels over the tumour confirm text guidance is directing attention to the correct region. If text is not helping, this map would show random noise rather than tumour-localised differences.

**Fully self-contained** — re-defines `FiLMUNet` inline and loads from saved checkpoints. Run this cell alone without retraining.
"""

# ── 6.3 Attention / Explainability — FiLM Activation Heatmaps ───────────────
# Self-contained: re-defines model, loads checkpoints, loads test data if needed.
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
# %matplotlib inline  # notebook magic

ON_KAGGLE   = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(".").resolve()
CKPT_DIR    = Path("/kaggle/working/checkpoints") if ON_KAGGLE else PROJECT_DIR/"checkpoints"
OUT_DIR     = Path("/kaggle/working/outputs")     if ON_KAGGLE else PROJECT_DIR/"outputs"
FIG_DIR     = OUT_DIR/"figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = Path(os.environ.get("BRATS_DATASET_DIR", str(PROJECT_DIR/"dataset")))
device = torch.device("mps" if torch.backends.mps.is_available()
         else "cuda" if torch.cuda.is_available() else "cpu")

# ── Load test data if not already in memory ───────────────────────────────────
if "Y_te" not in dir():
    print("X_te/Y_te/E_te not in memory — loading test split from disk...")
    import gc
    from tqdm.notebook import tqdm
    for _p in [str(PROJECT_DIR),
               "/kaggle/input/datasets/sudarshannn2005/brats-multimodal-code"]:
        if _p not in sys.path: sys.path.insert(0, _p)
    os.environ.setdefault("BRATS_DATASET_DIR", str(DATA_DIR))
    from data_loader_utils import build_patient_split_samples, BraTSMultimodalDataset

    EMB_PATH = OUT_DIR / "clip_text_embeddings.pt"
    assert EMB_PATH.exists(), f"Missing {EMB_PATH} — run the CLIP embeddings cell first"
    emb_dict = torch.load(EMB_PATH, map_location="cpu", weights_only=False)

    _, _, test_s = build_patient_split_samples()
    ds = BraTSMultimodalDataset(test_s, include_blank=True)
    n  = len(ds)
    X_te = np.empty((n,128,128), dtype=np.float32)
    Y_te = np.empty((n,128,128), dtype=np.float32)
    E_te = np.empty((n,512),     dtype=np.float32)
    for i, (img, mask, _, meta) in enumerate(tqdm(ds, desc="test", leave=False)):
        X_te[i] = img[0].numpy()
        Y_te[i] = mask[0].numpy()
        pid = int(meta["patient_id"].split("_")[-1])
        E_te[i] = emb_dict.get(pid, torch.zeros(512)).numpy()
    del ds; gc.collect()
    print(f"Loaded test split: {n} slices")

# ── Model definition ──────────────────────────────────────────────────────────
class _FiLM(nn.Module):
    def __init__(self, td, ch):
        super().__init__()
        self.fc = nn.Linear(td, 2*ch)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x, e):
        g, b = self.fc(e).chunk(2, 1)
        return (1+g.view(-1,x.shape[1],1,1))*x + b.view(-1,x.shape[1],1,1)

class _CBR(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class _FiLMUNet(nn.Module):
    def __init__(self, td=512):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1=_CBR(1,32); self.e2=_CBR(32,64)
        self.e3=_CBR(64,128); self.e4=_CBR(128,256); self.bn=_CBR(256,512)
        self.film_bn=_FiLM(td,512); self.film_d4=_FiLM(td,256)
        self.film_d3=_FiLM(td,128); self.film_d2=_FiLM(td,64); self.film_d1=_FiLM(td,32)
        self.u4=nn.ConvTranspose2d(512,256,2,stride=2); self.d4=_CBR(512,256)
        self.u3=nn.ConvTranspose2d(256,128,2,stride=2); self.d3=_CBR(256,128)
        self.u2=nn.ConvTranspose2d(128,64,2,stride=2);  self.d2=_CBR(128,64)
        self.u1=nn.ConvTranspose2d(64,32,2,stride=2);   self.d1=_CBR(64,32)
        self.head = nn.Conv2d(32,1,1)
    def forward(self, x, emb=None):
        s1=self.e1(x); s2=self.e2(self.pool(s1))
        s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3)); b=self.bn(self.pool(s4))
        if emb is not None: b=self.film_bn(b,emb)
        d4=self.d4(torch.cat([self.u4(b),s4],1))
        if emb is not None: d4=self.film_d4(d4,emb)
        d3=self.d3(torch.cat([self.u3(d4),s3],1))
        if emb is not None: d3=self.film_d3(d3,emb)
        d2=self.d2(torch.cat([self.u2(d3),s2],1))
        if emb is not None: d2=self.film_d2(d2,emb)
        d1=self.d1(torch.cat([self.u1(d2),s1],1))
        if emb is not None: d1=self.film_d1(d1,emb)
        return self.head(d1)

# ── Load checkpoints ──────────────────────────────────────────────────────────
mm_vis = _FiLMUNet().to(device); io_vis = _FiLMUNet().to(device)
for m, p, lbl in [(mm_vis, CKPT_DIR/"best_model.pth", "Multimodal"),
                  (io_vis, CKPT_DIR/"best_model_image_only.pth", "Image-Only")]:
    if p.exists():
        c = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(c["model_state"]); m.eval()
        print(f"Loaded {lbl} ep{c['epoch']} dice={c['val_dice']:.4f}")
    else:
        print(f"WARNING: {p} missing")

# ── Forward hooks on e4 ───────────────────────────────────────────────────────
_acts = {}
h_mm = mm_vis.e4.register_forward_hook(lambda mo,i,o: _acts.update({"mm": o.detach()}))
h_io = io_vis.e4.register_forward_hook(lambda mo,i,o: _acts.update({"io": o.detach()}))

# ── Pick 5 test slices ────────────────────────────────────────────────────────
has_tumor = np.where(Y_te.sum(axis=(1,2)) > 0)[0]
if len(has_tumor) == 0: has_tumor = np.arange(len(Y_te))
idxs = np.random.RandomState(7).choice(has_tumor, min(5, len(has_tumor)), replace=False)

fig, axes = plt.subplots(len(idxs), 5, figsize=(20, 4*len(idxs)))
if len(idxs) == 1: axes = axes[np.newaxis]
for col, t in enumerate(["Input MRI","Ground Truth","MM Activation","IO Activation","Text Contribution"]):
    axes[0,col].set_title(t, fontsize=10, fontweight="bold")

with torch.no_grad():
    for row, idx in enumerate(idxs):
        img = torch.from_numpy(X_te[idx]).unsqueeze(0).unsqueeze(0).to(device)
        emb = torch.from_numpy(E_te[idx]).unsqueeze(0).to(device)
        pred_mm = torch.sigmoid(mm_vis(img, emb))
        pred_io = torch.sigmoid(io_vis(img, torch.zeros_like(emb)))
        H, W = img.shape[2], img.shape[3]
        act_mm = F.interpolate(_acts["mm"].mean(1,keepdim=True),(H,W),mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        act_io = F.interpolate(_acts["io"].mean(1,keepdim=True),(H,W),mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        diff   = (pred_mm - pred_io).abs()[0,0].cpu().numpy()
        sl = X_te[idx]; disp = (sl-sl.min())/(sl.max()-sl.min()+1e-8)
        axes[row,0].imshow(disp, cmap="gray"); axes[row,0].axis("off")
        axes[row,1].imshow(disp, cmap="gray"); axes[row,1].contour(Y_te[idx],colors="lime",linewidths=1); axes[row,1].axis("off")
        for c2, act in enumerate([act_mm, act_io, diff]):
            axes[row,c2+2].imshow(disp, cmap="gray", alpha=0.6)
            axes[row,c2+2].imshow(act, cmap="hot", alpha=0.5); axes[row,c2+2].axis("off")
        axes[row,0].set_ylabel(f"Slice {idx}", fontsize=8)

h_mm.remove(); h_io.remove()
plt.suptitle("6.3 FiLM Encoder Activations and Text Contribution\nCol3=Multimodal  Col4=Image-Only  Col5=|MM-IO|", fontsize=10)
plt.tight_layout()
out = FIG_DIR/"fig_attention_maps.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved -> {out}")
del mm_vis, io_vis

