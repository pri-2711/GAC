"""Command-line entry point for Phase 1 (same logic as the notebook).

Usage:  python scripts/run_phase1.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audio_utils  # noqa: E402

if __name__ == "__main__":
    *_, report = audio_utils.run_pipeline(force="--force" in sys.argv)
    print(report)
