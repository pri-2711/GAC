"""Run Phase 5 (PCA) end to end: results/pca/ tables, model artifacts, figures and validation summary.

Usage:  python scripts/run_phase5.py [--no-figures]
Needs the outputs of Phases 1-4 (results/mfcc/, results/feature_analysis/). Earlier phases are only read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import pca_analysis as pa  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    d = pa.run_pipeline(make_figures=not args.no_figures)
    print("\n".join(d["checks"].format_lines()))
    print(f"{sum(ok for _, ok, _ in d['checks'])}/{len(d['checks'])} checks passed")
    sys.exit(0 if d["checks"].passed else 1)
