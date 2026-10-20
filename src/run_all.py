"""NeuroFiLM: run all stages in order inside ONE shared namespace (mirrors the notebook).

Usage (from the project root, on an Apple Silicon Mac):
    python src/run_all.py            # everything
    python src/run_all.py 04 07      # only stages whose filename starts with 04 or 07
Stages 01-06 define state (device, data arrays, model class) that later stages need,
so when selecting stages include their prerequisites.
"""
import sys, runpy
from pathlib import Path
from IPython.display import display

HERE = Path(__file__).parent
stages = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
if len(sys.argv) > 1:
    stages = [p for p in stages if p.name[:2] in sys.argv[1:]]

ns = {"display": display, "__name__": "__main__"}
for p in stages:
    print(f"\n########## {p.name} ##########")
    exec(compile(p.read_text(), str(p), "exec"), ns)
