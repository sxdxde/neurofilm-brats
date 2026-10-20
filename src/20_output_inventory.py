"""Output inventory check

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 21 — Output Inventory Check

Scans `outputs/` and `checkpoints/` and prints a checklist of every expected deliverable file with its size on disk.

A `[OK]` means the file exists and was generated successfully.
A `[MISS]` means the corresponding cell has not been run yet or saved to a different path.

Use this cell before submission to confirm all required files are present:
- `best_model.pth` and `best_model_image_only.pth` (trained checkpoints)
- `training_log.csv` and `training_log_image_only.csv` (epoch logs)
- `results_table.csv` (per-slice evaluation metrics)
- All figure PNGs in `outputs/figures/`
"""

import os; from pathlib import Path
ON_KAGGLE = os.path.exists("/kaggle/input")
OUT_DIR   = Path("/kaggle/working/outputs")     if ON_KAGGLE else Path(".").resolve()/"outputs"
CKPT_DIR  = Path("/kaggle/working/checkpoints") if ON_KAGGLE else Path(".").resolve()/"checkpoints"
FIG_DIR   = OUT_DIR/"figures"
print("="*50, "\nOUTPUT INVENTORY\n"+"="*50)
for sec, files in [
    ("Checkpoints",[CKPT_DIR/"best_model.pth", CKPT_DIR/"best_model_image_only.pth"]),
    ("Logs",       [OUT_DIR/"clip_text_embeddings.pt", OUT_DIR/"training_log.csv",
                    OUT_DIR/"training_log_image_only.csv", OUT_DIR/"results_table.csv"]),
    ("Figures",    list(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else [])]:
    print(f"\n{sec}:")
    for p in files:
        ok=Path(p).exists()
        print(f"  [{'OK' if ok else 'MISS'}] {Path(p).name}{(' ('+str(round(Path(p).stat().st_size/1e6,1))+' MB)') if ok else ''}")

