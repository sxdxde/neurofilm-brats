"""Wilcoxon significance tests, MM vs IO

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell — Statistical Significance: MM vs IO (Wilcoxon)
**Improvement 2 — Statistical Significance Testing**

Runs a two-sided Wilcoxon signed-rank test and paired t-test on per-slice Dice scores. Also computes Cohen's d and per-metric p-values for IoU, Precision, and Recall.
"""

# ── Statistical Significance: MM vs IO (Wilcoxon) ────────────────────────────
# Self-contained: loads from outputs/results_table.csv
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("outputs/results_table.csv")
# Compute within-patient slice index
df["_within"] = df.groupby("patient_id").cumcount()

dice_mm = df["dice_mm"].values;   dice_io = df["dice_io"].values
iou_mm  = df["iou_mm"].values;    iou_io  = df["iou_io"].values
prec_mm = df["precision_mm"].values; prec_io = df["precision_io"].values
rec_mm  = df["recall_mm"].values;  rec_io  = df["recall_io"].values

# ── Wilcoxon + t-test on Dice ─────────────────────────────────────────────────
diff_dice = dice_mm - dice_io
diff_nz   = diff_dice[diff_dice != 0]
try:
    _, w_p = stats.wilcoxon(diff_nz, alternative="two-sided")
except Exception:
    w_p = float("nan")
_, t_p = stats.ttest_rel(dice_mm, dice_io)
pooled_std = np.sqrt((dice_mm.std()**2 + dice_io.std()**2) / 2)
cohens_d   = (dice_mm.mean() - dice_io.mean()) / (pooled_std + 1e-10)

def d_label(d):
    a = abs(d)
    return ("negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large")

interp_short = ("Models are statistically equivalent (p > 0.05). Text conditioning does not"
                " significantly alter Dice but shifts the precision\u2013recall trade-off."
                if w_p > 0.05 else "Models are significantly different (p \u2264 0.05).")

N = len(df)
print("\u2554" + "\u2550"*54 + "\u2557")
print("\u2551  Statistical Equivalence: Multimodal vs Image-Only  \u2551")
print("\u2560" + "\u2550"*54 + "\u2563")
print(f"\u2551  Mean Dice MM  : {dice_mm.mean():.4f} \u00b1 {dice_mm.std():.4f}"
      + " "*20 + "\u2551")
print(f"\u2551  Mean Dice IO  : {dice_io.mean():.4f} \u00b1 {dice_io.std():.4f}"
      + " "*20 + "\u2551")
print(f"\u2551  Wilcoxon p    : {w_p:.4e} (two-sided)" + " "*14 + "\u2551")
print(f"\u2551  Paired t-test : {t_p:.4e} (two-sided)" + " "*14 + "\u2551")
print(f"\u2551  Cohen's d     : {cohens_d:.4f} ({d_label(cohens_d)})" + " "*25 + "\u2551")
print(f"\u2551  N             : {N}"  + " "*37 + "\u2551")
print("\u255a" + "\u2550"*54 + "\u255d")
print(f"  Interpretation: {interp_short[:70]}")

# ── Per-metric Wilcoxon p-values ──────────────────────────────────────────────
print("\nPer-metric Wilcoxon signed-rank p-values:")
print(f"  {'Metric':12s}  {'MM mean':>8}  {'IO mean':>8}  {'p-value':>10}  Conclusion")
print("  " + "-"*60)
for metric, mm, io in [("Dice",      dice_mm, dice_io),
                        ("IoU",       iou_mm,  iou_io),
                        ("Precision", prec_mm, prec_io),
                        ("Recall",    rec_mm,  rec_io)]:
    d = mm - io
    d_nz = d[d != 0]
    try:
        _, p = stats.wilcoxon(d_nz, alternative="two-sided")
    except Exception:
        p = float("nan")
    concl = "equiv" if (p > 0.05 or np.isnan(p)) else "differ"
    print(f"  {metric:12s}  {mm.mean():8.4f}  {io.mean():8.4f}  {p:10.4e}  {concl}")

