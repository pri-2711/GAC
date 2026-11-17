"""Predict the region and accent of a NEW recording with the final Phase 6 model.

Usage:  python scripts/predict_accent.py path/to/recording.m4a   [--bundle results/model_development/final_model/bundle]

The recording may be M4A/OGG/WAV (non-WAV files are standardised with the same FFmpeg step as Phase 1). The pipeline
extracts exactly the training features (78 Phase 3 features, and a Log-Mel spectrogram when the model has a CNN branch),
applies the saved normalisation, runs the network, then decides hierarchically: region first, then the accent among the
accents belonging to that region (the region -> accent mapping is read from the model bundle).

IMPORTANT: the model was trained on 35 recordings from 5 speakers (one speaker per accent). Its output reflects similarity
to those five recordings' voices and recording setups; it is NOT evidence of speaker-independent accent recognition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import model_dev as mdl  # noqa: E402
import config as cfg  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--bundle", type=Path, default=cfg.RESULTS_DIR / "model_development" / "final_model" / "bundle")
    a = ap.parse_args()
    res = mdl.AccentPredictor(a.bundle).predict_file(a.audio)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nNote: trained on one speaker per accent; this is not a speaker-independent accent classifier.", file=sys.stderr)
