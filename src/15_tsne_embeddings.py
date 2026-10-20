"""t-SNE of CLIP image/text embeddings

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — 6.5 Embedding Analysis: t-SNE of CLIP Image and Text Embeddings

**Deliverable 6.5 — Embedding Analysis**

Demonstrates cross-modal alignment by visualising CLIP image and text embeddings in a shared 2-D latent space.

**What is t-SNE?**
t-SNE (t-Distributed Stochastic Neighbour Embedding) maps 512-D CLIP vectors to 2-D while preserving local structure. Points nearby in 512-D space stay nearby in the plot.

**Plot legend:**
- **Blue circles** — CLIP visual encoder output for 80 test FLAIR slices
- **Orange triangles** — CLIP text encoder output for the paired radiology reports
- **Grey lines** — connect each slice to its patient's text embedding

**How to read alignment:**
Short grey lines with interleaved clusters → image and text embeddings are close in CLIP space → the model can meaningfully fuse them.
Long lines with separated clusters → the modalities are weakly aligned, limiting how much the text can guide segmentation.

**Fully self-contained** — loads `outputs/clip_text_embeddings.pt`. Run this cell alone.
"""

# ── 6.5 t-SNE of CLIP Image + Text Embeddings ───────────────────────────────
# Self-contained. Requires: outputs/clip_text_embeddings.pt, X_te/Y_te/E_te in memory.
import os, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor
# %matplotlib inline  # notebook magic

ON_KAGGLE   = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(".").resolve()
OUT_DIR     = Path("/kaggle/working/outputs") if ON_KAGGLE else PROJECT_DIR/"outputs"
FIG_DIR     = OUT_DIR/"figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH    = OUT_DIR/"clip_text_embeddings.pt"

assert EMB_PATH.exists(), f"Run Cell 10 first to generate {EMB_PATH}"
emb_dict = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
pids     = list(emb_dict.keys())
txt_mat  = torch.stack([emb_dict[p] for p in pids])
txt_mat  = txt_mat / txt_mat.norm(dim=-1, keepdim=True)
print(f"Loaded text embeddings: {len(pids)} patients")

print("Loading CLIP visual encoder (CPU)...")
clip_mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc     = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Ready.")

N    = 80
ht   = np.where(Y_te.sum(axis=(1,2)) > 50)[0]
sidx = np.random.RandomState(42).choice(ht, min(N, len(ht)), replace=False)

uniq, umap = {}, []
for si in sidx:
    key = tuple(E_te[si][:8].tolist())
    uniq[key] = torch.from_numpy(E_te[si]); umap.append(key)
print(f"Sampled {len(sidx)} slices from {len(uniq)} patients")

img_embs = []
with torch.no_grad():
    for s in range(0, len(sidx), 16):
        batch = sidx[s:s+16]
        imgs  = []
        for si in batch:
            sl = X_te[si]; u8 = ((sl-sl.min())/(sl.max()-sl.min()+1e-8)*255).astype("uint8")
            imgs.append(np.stack([u8]*3,-1))
        inp = proc(images=imgs, return_tensors="pt", padding=True)
        out = clip_mdl.get_image_features(**inp)
        if hasattr(out, "pooler_output"): out = out.pooler_output
        out = out / out.norm(dim=-1, keepdim=True)
        img_embs.append(out.cpu().numpy())
        if (s//16) % 3 == 0: print(f"  {min(s+16,len(sidx))}/{len(sidx)} image embeddings")

img_np = np.concatenate(img_embs, 0)
txt_np = np.stack([uniq[k].numpy() for k in umap])
txt_np = txt_np / (np.linalg.norm(txt_np,1,keepdims=True)+1e-8)

print("Running t-SNE (30-60s)...")
coords = TSNE(n_components=2, perplexity=min(30,len(sidx)//3),
              random_state=42, init="pca", learning_rate="auto").fit_transform(
              np.concatenate([img_np, txt_np],0))
im2d = coords[:len(sidx)]; tx2d = coords[len(sidx):]
print("Done.")

fig, ax = plt.subplots(figsize=(10,8))
for i in range(len(sidx)):
    ax.plot([im2d[i,0],tx2d[i,0]],[im2d[i,1],tx2d[i,1]],color="grey",alpha=0.2,lw=0.6,zorder=1)
ax.scatter(im2d[:,0],im2d[:,1],c="steelblue",s=40,alpha=0.85,zorder=3,label=f"FLAIR image (n={len(sidx)})")
ax.scatter(tx2d[:,0],tx2d[:,1],c="darkorange",s=40,alpha=0.85,marker="^",zorder=3,
           label=f"Text report (n={len(uniq)} patients)")
ax.set_title("6.5 t-SNE: CLIP Image vs Text Embeddings\nBlue=FLAIR  Orange=Reports  Grey=same-patient pair",fontsize=12)
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.legend(fontsize=10); ax.grid(alpha=0.25)
plt.tight_layout()
out = FIG_DIR/"fig_tsne_embeddings.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved -> {out}")
del clip_mdl

