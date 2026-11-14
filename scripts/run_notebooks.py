"""Execute the phase notebooks from top to bottom, headlessly.

Usage:
    python scripts/run_notebooks.py                 # run notebooks 01-04 in order, saving outputs in place
    python scripts/run_notebooks.py --only 3 4      # run only phases 3 and 4
    python scripts/run_notebooks.py --no-save       # run, but do not overwrite the .ipynb files

The notebooks write their results to results/ (and dataset/ for Phase 1), so each phase needs the
previous ones to have been run once. Re-running reproduces the same numbers.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
PHASES = {1: "01_audio_standardization.ipynb", 2: "02_audio_analysis.ipynb",
          3: "03_mfcc_extraction.ipynb", 4: "04_feature_analysis.ipynb"}


def run(phase: int, save: bool, timeout: int) -> None:
    path = NOTEBOOK_DIR / PHASES[phase]
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    # The kernel starts in notebooks/, which is where the notebooks expect to run from.
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
    if save:
        nbformat.write(nb, path)
    print(f"Phase {phase}: {path.name} executed in {time.time() - start:.0f}s" + ("" if save else " (not saved)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="+", choices=sorted(PHASES), help="phase numbers to run (default: all)")
    parser.add_argument("--no-save", action="store_true", help="do not write executed outputs back to the notebooks")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for phase in args.only or sorted(PHASES):
        run(phase, save=not args.no_save, timeout=args.timeout)
