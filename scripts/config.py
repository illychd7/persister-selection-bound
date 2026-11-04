"""Folder layout used by every script.

Defaults are relative to the repository root; override any of them with environment variables:

    PSB_DATA     raw and preprocessed data      (default: <repo>/data)
    PSB_RESULTS  JSON/CSV results               (default: <repo>/results)
    PSB_FIGS     figure output                  (default: <repo>/figures)

Run scripts from anywhere as `python scripts/<name>.py`.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = os.environ.get("PSB_DATA", str(ROOT / "data"))
RESULTS = os.environ.get("PSB_RESULTS", str(ROOT / "results"))
FIGS = os.environ.get("PSB_FIGS", str(ROOT / "figures"))

PC9 = DATA + "/pc9"          # GSE150949: PC9 time course and pooled Watermelon experiment
GEO = DATA + "/geo"          # GSE291678 / GSE291679 supplementary files (ClonMapper, MDA-MB-231)
TNBC = DATA + "/tnbc"        # cell metadata exported from the authors' h5ad
MERLIN = DATA + "/merlin"    # GSE299711 MeRLin in vitro (exported from Seurat RDS)
NEWDATA = DATA + "/newdata"  # preprocessed TNBC and MeRLin matrices

for _d in (DATA, RESULTS, FIGS):
    os.makedirs(_d, exist_ok=True)
