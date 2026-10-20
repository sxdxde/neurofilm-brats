"""Grad-CAM++ on final decoder block (d1)

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — 6.3b Grad-CAM Explainability
**Deliverable 6.3b — Grad-CAM Saliency Maps**

Implements Grad-CAM targeting `e4.b[-3]` (second Conv2d of encoder-4 block) for 5 tumor-containing test slices. Compares multimodal vs image-only attention patterns. Saves `outputs/figures/fig_gradcam.png`.
"""

# ── 6.3b — Grad-CAM Explainability (v3 – decoder-layer target) ──────────────
# Fix: target d1 (last decoder block, 128×128) instead of e4 (encoder, 16×16).
# Coarse encoder features produce blurry blobs after 8× upsample; decoder
# features have direct spatial correspondence to the output segmentation mask.
# Score = sum of positive logits → gradient = 1 at predicted-tumor pixels only.
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(".").resolve()
OUT_DIR  = PROJECT_DIR / "outputs"
CKPT_DIR = PROJECT_DIR / "checkpoints"
FIG_DIR  = OUT_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_VAL = PROJECT_DIR / "dataset" / "FLAIR_BRATS2020_split" / "val"

DEVICE = (torch.device("mps")  if torch.backends.mps.is_available()  else
          torch.device("cuda") if torch.cuda.is_available()           else
          torch.device("cpu"))

# ── Inline model (inplace=False so retain_grad works) ────────────────────────
class FiLM(nn.Module):
    def __init__(self, text_dim, channels):
        super().__init__()
        self.fc = nn.Linear(text_dim, 2 * channels)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x, emb):
        gb = self.fc(emb); g, b = gb.chunk(2, dim=1)
        return (1 + g.view(-1, x.shape[1], 1, 1)) * x + b.view(-1, x.shape[1], 1, 1)

class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=False),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=False))
    def forward(self, x): return self.b(x)

class FiLMUNet(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBnRelu(1,   32);  self.e2 = ConvBnRelu(32,  64)
        self.e3 = ConvBnRelu(64, 128);  self.e4 = ConvBnRelu(128, 256)
        self.bn = ConvBnRelu(256, 512)
        self.film_bn = FiLM(text_dim, 512); self.film_d4 = FiLM(text_dim, 256)
        self.film_d3 = FiLM(text_dim, 128); self.film_d2 = FiLM(text_dim, 64)
        self.film_d1 = FiLM(text_dim, 32)
        self.u4 = nn.ConvTranspose2d(512,256,2,stride=2); self.d4 = ConvBnRelu(512,256)
        self.u3 = nn.ConvTranspose2d(256,128,2,stride=2); self.d3 = ConvBnRelu(256,128)
        self.u2 = nn.ConvTranspose2d(128, 64,2,stride=2); self.d2 = ConvBnRelu(128, 64)
        self.u1 = nn.ConvTranspose2d( 64, 32,2,stride=2); self.d1 = ConvBnRelu( 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, x, emb=None):
        s1=self.e1(x); s2=self.e2(self.pool(s1))
        s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3))
        b=self.bn(self.pool(s4))
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

def load_model(ckpt_path, device):
    m = FiLMUNet(512).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval(); return m

mm_model = load_model(CKPT_DIR / "best_model.pth",           DEVICE)
io_model = load_model(CKPT_DIR / "best_model_image_only.pth", DEVICE)
emb_dict = torch.load(OUT_DIR / "clip_text_embeddings.pt", map_location="cpu", weights_only=False)
print(f"Models loaded on {DEVICE}.")

# ── Select 5 tumor-containing test slices (prefer high-Dice rows) ─────────────
df = pd.read_csv(OUT_DIR / "results_table.csv")
df["_within"] = df.groupby("patient_id").cumcount()
tumor_df = (df[df["has_tumor"] == 1]
            .sort_values("dice_mm", ascending=False)
            .reset_index(drop=True))
rng = np.random.default_rng(42)
n_pick = min(5, len(tumor_df))
# spread picks across the dice range so we see diverse cases
picks = np.linspace(0, len(tumor_df) - 1, n_pick, dtype=int)

slices_data = []
for ci in picks:
    row   = tumor_df.iloc[ci]
    pid   = int(row["patient_id"])
    ws    = int(row["_within"])
    ivol  = np.load(DATA_VAL / "images" / f"image_{pid}.npy").astype(np.float32)
    mvol  = np.load(DATA_VAL / "masks"  / f"mask_{pid}.npy")
    img_s = ivol[:, :, ws]
    mu, sg = img_s.mean(), img_s.std()
    img_s = (img_s - mu) / (sg + 1e-6)
    msk_s = (mvol[:, :, ws, 1:].sum(-1) > 0).astype(np.float32)
    emb   = emb_dict.get(pid, torch.zeros(512)).numpy()
    slices_data.append((img_s, msk_s, emb, pid, ws))
print(f"Loaded {len(slices_data)} slices.")

# ── Grad-CAM targeting d1 (final decoder block, 128×128) ─────────────────────
# Why d1 and not e4:
#  • e4 is 16×16 — 8× upsample smears the map into a diffuse blob.
#  • d1 is 128×128 — same spatial resolution as the output mask.
#    Skip connections bring back fine spatial detail from e1, so the
#    features here directly encode *where* the model will predict tumor.
# Score = sum of positive logits (= predicted-tumor probability, no sigmoid
#   saturation).  Gradients are 1.0 at every predicted-tumor pixel and 0
#   elsewhere, so the Grad-CAM weights highlight exactly those regions.

def compute_gradcam(model, img_np, emb_np, device, use_emb=True):
    img_t = torch.tensor(img_np[None, None], dtype=torch.float32, device=device)
    emb_t = (torch.tensor(emb_np[None], dtype=torch.float32, device=device)
             if use_emb else torch.zeros(1, 512, device=device))
    model.zero_grad()

    with torch.enable_grad():
        # ── Full encoder + bottleneck ─────────────────────────────────────────
        s1 = model.e1(img_t)
        s2 = model.e2(model.pool(s1))
        s3 = model.e3(model.pool(s2))
        s4 = model.e4(model.pool(s3))
        b  = model.bn(model.pool(s4))
        if use_emb: b = model.film_bn(b, emb_t)

        # ── Decoder d4 → d3 → d2 (full) ──────────────────────────────────────
        d4 = model.d4(torch.cat([model.u4(b), s4], 1))
        if use_emb: d4 = model.film_d4(d4, emb_t)
        d3 = model.d3(torch.cat([model.u3(d4), s3], 1))
        if use_emb: d3 = model.film_d3(d3, emb_t)
        d2 = model.d2(torch.cat([model.u2(d3), s2], 1))
        if use_emb: d2 = model.film_d2(d2, emb_t)

        # ── Split d1 at b[3] — capture feature map before final BN+ReLU ──────
        d1_in = torch.cat([model.u1(d2), s1], 1)   # [1, 64, 128, 128]
        feat  = model.d1.b[:4](d1_in)               # [1, 32, 128, 128]
        feat.retain_grad()
        d1_out = model.d1.b[4:](feat)               # BN + ReLU
        if use_emb: d1_out = model.film_d1(d1_out, emb_t)
        out = model.head(d1_out)                     # [1, 1, 128, 128]

        # Score: sum of positive logits (predicted-tumor pixels)
        score = torch.relu(out).sum()
        if score.item() < 1e-8:   # model predicts nothing — use full logit sum
            score = out.sum()
        score.backward()

    grad = feat.grad          # [1, 32, 128, 128]
    fmap = feat.detach()

    if grad is None:
        print("  WARNING: grad is None — Grad-CAM failed")
        return np.zeros((128, 128), dtype=np.float32)

    # ── Grad-CAM formula ─────────────────────────────────────────────────────
    # Use Grad-CAM++ weights: mean over (h,w) of positive-gradient contribution
    weights = torch.relu(grad).mean(dim=(2, 3), keepdim=True)   # [1,32,1,1]
    cam = torch.relu((weights * fmap).sum(dim=1)).squeeze().cpu().numpy()  # [128,128]

    if cam.ndim < 2 or min(cam.shape) < 2:
        return np.zeros((128, 128), dtype=np.float32)

    # ── Normalize + mild Gaussian smooth to remove speckling ─────────────────
    from scipy.ndimage import gaussian_filter
    cam = gaussian_filter(cam.astype(np.float32), sigma=2)
    lo, hi = cam.min(), cam.max()
    return (cam - lo) / (hi - lo + 1e-8)

# ── Figure: 5 rows × 4 cols ──────────────────────────────────────────────────
# Cols: FLAIR+GT | MM CAM | IO CAM | CAM difference (MM − IO)
try:
    from skimage import measure as sk_measure
    _has_skimage = True
except ImportError:
    _has_skimage = False

n   = len(slices_data)
fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
if n == 1: axes = axes[None, :]
mm_ovs, io_ovs = [], []

for ri, (img_s, msk_s, emb, pid, s_idx) in enumerate(slices_data):
    cam_mm = compute_gradcam(mm_model, img_s, emb,           DEVICE, use_emb=True)
    cam_io = compute_gradcam(io_model, img_s, np.zeros(512), DEVICE, use_emb=False)

    gt = msk_s > 0.5
    if gt.sum() > 0:
        mm_ovs.append(float((cam_mm > 0.5)[gt].mean()))
        io_ovs.append(float((cam_io > 0.5)[gt].mean()))

    img_dn  = (img_s - img_s.min()) / (img_s.max() - img_s.min() + 1e-8)
    img_rgb = np.stack([img_dn] * 3, axis=-1)

    def draw_contour(ax, mask):
        if _has_skimage and mask.sum() > 0:
            for ct in sk_measure.find_contours(mask, 0.5):
                ax.plot(ct[:, 1], ct[:, 0], "lime", lw=1.5)

    # Col 0: FLAIR + GT contour
    axes[ri, 0].imshow(img_dn, cmap="gray"); draw_contour(axes[ri, 0], msk_s)
    axes[ri, 0].axis("off"); axes[ri, 0].set_title(f"p{pid} s{s_idx}\nFLAIR + GT")

    # Col 1: MM Grad-CAM
    blend_mm = np.clip(0.45 * img_rgb + 0.55 * cm.hot(cam_mm)[..., :3], 0, 1)
    axes[ri, 1].imshow(blend_mm); draw_contour(axes[ri, 1], msk_s)
    axes[ri, 1].axis("off"); axes[ri, 1].set_title("MM Grad-CAM (d1)")

    # Col 2: IO Grad-CAM
    blend_io = np.clip(0.45 * img_rgb + 0.55 * cm.hot(cam_io)[..., :3], 0, 1)
    axes[ri, 2].imshow(blend_io); draw_contour(axes[ri, 2], msk_s)
    axes[ri, 2].axis("off"); axes[ri, 2].set_title("IO Grad-CAM (d1)")

    # Col 3: signed difference MM − IO (blue=IO-dominates, red=MM-dominates)
    diff = cam_mm - cam_io   # in [-1, 1]
    diff_norm = (diff + 1) / 2   # map to [0,1]
    axes[ri, 3].imshow(diff_norm, cmap="RdBu_r", vmin=0, vmax=1)
    draw_contour(axes[ri, 3], msk_s)
    axes[ri, 3].axis("off"); axes[ri, 3].set_title("MM − IO\n(red=MM-higher)")

plt.suptitle("6.3b — Grad-CAM Explainability (Final Decoder Layer d1, 128×128)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

mm_ov = float(np.mean(mm_ovs)) if mm_ovs else 0.0
io_ov = float(np.mean(io_ovs)) if io_ovs else 0.0
print(f"MM Grad-CAM mask overlap (>0.5): {mm_ov:.4f}")
print(f"IO Grad-CAM mask overlap (>0.5): {io_ov:.4f}")
del mm_model, io_model; gc.collect()

