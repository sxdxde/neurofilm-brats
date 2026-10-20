"""Precision-recall trade-off analysis

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — Precision–Recall Trade-off Analysis
**Improvement 4 — Precision/Recall Trade-off Analysis**

Loads `results_table.csv` and plots per-slice precision vs recall scatter (MM and IO) coloured by Dice, plus a histogram of per-slice precision differences. Saves `outputs/figures/fig_precision_recall_tradeoff.png`.
"""

# ── Precision–Recall Trade-off Analysis ──────────────────────────────────────
# Self-contained: loads from outputs/results_table.csv
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs"); FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DIR / "results_table.csv")
prec_mm = df["precision_mm"].values; prec_io = df["precision_io"].values
rec_mm  = df["recall_mm"].values;    rec_io  = df["recall_io"].values
dice_mm = df["dice_mm"].values

# ── Statistics ────────────────────────────────────────────────────────────────
mm_wins_prec = (prec_mm > prec_io).mean() * 100
io_wins_rec  = (rec_io  > rec_mm ).mean() * 100
delta_prec   = (prec_mm - prec_io).mean()
delta_rec    = (rec_io  - rec_mm ).mean()

print(f"Slices where MM precision > IO precision: {mm_wins_prec:.1f}%")
print(f"Slices where IO recall > MM recall: {io_wins_rec:.1f}%")
print(f"Mean precision advantage (MM - IO): {delta_prec:+.4f}")
print(f"Mean recall advantage (IO - MM): {delta_rec:+.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter (recall vs precision), coloured by Dice
sc = ax1.scatter(rec_mm, prec_mm, c=dice_mm, cmap="viridis", alpha=0.3, s=8, label="MM")
ax1.scatter(rec_io, prec_io, c="orange", alpha=0.3, s=8, marker="^", label="IO")
fig.colorbar(sc, ax=ax1, label="MM Dice")
ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
ax1.set_title("Precision vs Recall (per slice)")
ax1.legend(markerscale=3); ax1.grid(True, alpha=0.3)

# Right: histogram of (precision_MM - precision_IO)
delta = prec_mm - prec_io
ax2.hist(delta, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
ax2.axvline(0, color="red", linewidth=1.5, linestyle="--")
ax2.axvline(delta.mean(), color="navy", linewidth=1.5, linestyle="-", label=f"mean={delta.mean():+.4f}")
ax2.set_xlabel("\u0394 Precision (MM \u2212 IO)"); ax2.set_ylabel("Count")
ax2.set_title("MM precision advantage per slice")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Precision\u2013Recall Trade-off Analysis: Multimodal vs Image-Only", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR/"fig_precision_recall_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/figures/fig_precision_recall_tradeoff.png")

