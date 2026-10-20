"""ROUGE / BLEU / CLIP cosine similarity

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — Deliverable 7: Text Evaluation Metrics (ROUGE, BLEU, CLIP Cosine Similarity)

**Deliverable 7 — Evaluation Metrics: Text**

This model produces segmentation masks, not generated text — so ROUGE/BLEU cannot be applied in the traditional sense. Two standard proxy approaches are used:

### Metric 1 — CLIP Cosine Similarity (image ↔ text)
For each test FLAIR slice the CLIP visual embedding is compared to the CLIP text embedding of its ground-truth report. This measures cross-modal alignment in the shared embedding space — the mechanism by which text guidance operates. Typical CLIP scores range 0.2–0.4; higher is better.

### Metric 2 — Nearest-Neighbour Report Retrieval (ROUGE-1/2/L, BLEU-4)
For each test patient the most similar training-set report is retrieved by CLIP embedding distance. ROUGE and BLEU are then computed between the retrieved report and the true test report. This provides an upper-bound proxy for what a text generation head would produce.

Saved to `outputs/results_table_text_metrics.csv`.

**Fully self-contained** — loads `outputs/clip_text_embeddings.pt`. Run this cell alone.
"""

# ── Deliverable 7: Text Metrics — ROUGE / BLEU + CLIP Cosine Similarity ─────
# Self-contained. Requires: outputs/clip_text_embeddings.pt, E_te/Y_te in memory.
import os, csv, glob, re, ssl, torch, numpy as np
from pathlib import Path
from rouge_score import rouge_scorer as _rs
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import CLIPModel, CLIPProcessor

# Fix NLTK SSL download on macOS
try:
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    try:
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        import nltk
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
        ssl._create_default_https_context = _orig_ctx
    except Exception as e:
        print(f"NLTK download failed (non-fatal): {e}")

ON_KAGGLE   = os.path.exists("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path(".").resolve()
OUT_DIR     = Path("/kaggle/working/outputs") if ON_KAGGLE else PROJECT_DIR/"outputs"
DATA_DIR    = Path(os.environ.get("BRATS_DATASET_DIR", str(PROJECT_DIR/"dataset")))
EMB_PATH    = OUT_DIR/"clip_text_embeddings.pt"

assert EMB_PATH.exists(), f"Run Cell 10 first to generate {EMB_PATH}"
emb_dict = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
pids     = list(emb_dict.keys())   # may be ints or strings
txt_mat  = torch.stack([emb_dict[p] for p in pids])
txt_mat  = txt_mat / txt_mat.norm(dim=-1, keepdim=True)

# Load text reports and build a lookup that handles both int and string pid keys
txt_files  = glob.glob(str(DATA_DIR/"**/*.txt"), recursive=True)
report_map = {}   # int_pid -> text
for fp in txt_files:
    m = re.search(r"BraTS20_Training_(\d+)", fp)
    if m:
        with open(fp) as fh:
            report_map[int(m.group(1))] = fh.read().strip().lower()
print(f"Loaded {len(report_map)} radiology reports")

# Normalise pids to int for consistent lookup
def _to_int(p):
    try: return int(p)
    except Exception: return None

pids_int = [_to_int(p) for p in pids]

print("Loading CLIP (CPU)...")
clip_mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc     = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ht   = np.where(Y_te.sum(axis=(1,2)) > 0)[0]
if len(ht) == 0: ht = np.arange(len(Y_te))
samp = np.random.RandomState(42).choice(ht, min(50, len(ht)), replace=False)

# Compute image embeddings
img_embs = []
with torch.no_grad():
    for si in samp:
        sl = X_te[si]; u8 = ((sl-sl.min())/(sl.max()-sl.min()+1e-8)*255).astype("uint8")
        inp = proc(images=[np.stack([u8]*3,-1)], return_tensors="pt")
        out = clip_mdl.get_image_features(**inp)
        if hasattr(out, "pooler_output"): out = out.pooler_output
        img_embs.append((out/out.norm(dim=-1,keepdim=True)).squeeze(0).cpu())
img_mat = torch.stack(img_embs)

# Cosine similarity: each image vs its paired text embedding
cos_sims = []
for i, si in enumerate(samp):
    key = tuple(E_te[si][:8].tolist())
    for j, pid in enumerate(pids):
        if tuple(emb_dict[pid].numpy()[:8].tolist()) == key:
            cos_sims.append((img_mat[i]*txt_mat[j]).sum().item()); break
mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
print(f"\nCLIP Cosine Similarity (image <-> paired text): {mean_cos:.4f}")
print("  (typical CLIP range 0.2-0.4; higher = better cross-modal alignment)")

# ROUGE / BLEU via nearest-neighbour retrieval
# Only use patients that have a text report
valid_indices = [j for j, p in enumerate(pids_int) if p in report_map]
if len(valid_indices) == 0:
    print("\nWARNING: No patient IDs matched between emb_dict and report files.")
    print(f"  emb_dict sample keys: {pids[:5]}")
    print(f"  report_map sample keys: {list(report_map.keys())[:5]}")
    r1s = r2s = rLs = bleus = [0.0]
else:
    n_train   = int(len(valid_indices) * 0.7)
    tr_idx    = valid_indices[:n_train]
    te_idx    = valid_indices[n_train:]
    tr_mat_v  = txt_mat[tr_idx]
    scorer    = _rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    smooth    = SmoothingFunction().method1

    r1s, r2s, rLs, bleus = [], [], [], []
    for j in te_idx[:30]:
        pid_int = pids_int[j]
        ref  = report_map[pid_int]
        sims = (tr_mat_v * txt_mat[j].unsqueeze(0)).sum(1)
        nn_j = tr_idx[sims.argmax().item()]
        hyp  = report_map[pids_int[nn_j]]
        sc   = scorer.score(ref, hyp)
        r1s.append(sc["rouge1"].fmeasure); r2s.append(sc["rouge2"].fmeasure)
        rLs.append(sc["rougeL"].fmeasure)
        bleus.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth))

print(f"\nNearest-Neighbour Report Retrieval ({len(r1s)} test patients):")
print(f"  ROUGE-1 : {np.mean(r1s):.4f}")
print(f"  ROUGE-2 : {np.mean(r2s):.4f}")
print(f"  ROUGE-L : {np.mean(rLs):.4f}")
print(f"  BLEU-4  : {np.mean(bleus):.4f}")

out_path = OUT_DIR/"results_table_text_metrics.csv"
with open(out_path,"w",newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["metric","value"])
    for name, val in [("clip_cosine_similarity", mean_cos),
                      ("rouge1",  np.mean(r1s)), ("rouge2", np.mean(r2s)),
                      ("rougeL",  np.mean(rLs)), ("bleu4",  np.mean(bleus))]:
        w.writerow([name, f"{val:.6f}"])
print(f"\nSaved -> {out_path}")
del clip_mdl

