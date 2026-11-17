"""Support code for the Streamlit frontend (app.py): upload validation and the call into the real model.

No feature extraction or normalisation is implemented here. Uploaded audio is standardised and classified by
model_dev.AccentPredictor (the same code path as scripts/predict_accent.py and the Phase 6 evaluation):
  decode + standardise (FFmpeg, 16 kHz mono PCM-16, as Phase 1) -> 78 Phase 3 features -> saved training-set
  z-score -> final MLP -> hierarchical region/accent decision.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import audio_utils as au
import config as cfg

BUNDLE_DIR = cfg.RESULTS_DIR / "model_development" / "final_model" / "bundle"
FINAL_INFO = cfg.RESULTS_DIR / "model_development" / "final_model" / "final_model_info.json"
COMPARISON_CSV = cfg.RESULTS_DIR / "model_development" / "evaluation" / "experiment_comparison.csv"

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
MAX_UPLOAD_MB = 50
MIN_SECONDS = 1.0        # below this the recording-level mean/std features are meaningless
WARN_SHORT_SECONDS = 5.0  # training recordings are 8.3 - 18.3 s
WARN_LONG_SECONDS = 30.0
MAX_SECONDS = 180.0
SILENT_PEAK = 1e-4       # essentially digital silence (peak amplitude, full scale = 1.0)
QUIET_RMS_DBFS = -55.0


class AudioProblem(Exception):
    """An input problem that should be shown to the user as a clear message (not a crash)."""


@dataclass
class PreparedAudio:
    name: str
    digest: str
    signal: np.ndarray
    sample_rate: int
    duration_s: float
    original_format: str
    original_sample_rate: int | None
    original_channels: int | str | None
    rms_dbfs: float
    peak: float
    warnings: list[str] = field(default_factory=list)


def load_predictor():
    """Load the final model bundle (the real Phase 6 MLP). Imported lazily because TensorFlow is slow to import."""
    import model_dev as mdl
    return mdl.AccentPredictor(BUNDLE_DIR)


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def prepare_upload(name: str, data: bytes, predictor) -> PreparedAudio:
    """Validate an uploaded file and standardise it with the project's pipeline. Raises AudioProblem with a user-facing message."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioProblem(f"“{ext or 'no extension'}” is not a supported file type. Please upload one of: "
                           + ", ".join(e.upper().lstrip('.') for e in SUPPORTED_EXTENSIONS) + ".")
    if not data:
        raise AudioProblem("The uploaded file is empty (0 bytes). Please choose a valid recording.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioProblem(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a shorter recording.")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"upload{ext}"
        src.write_bytes(data)
        try:
            props = au.inspect_original(src)
        except Exception:                                        # inspection is informational only
            props = {"sample_rate": None, "channels": None}
        try:
            y, sr = predictor.load_audio(src)
        except Exception as exc:
            raise AudioProblem("The file could not be decoded as audio. It may be corrupted, empty, or not really a "
                               f"{ext.lstrip('.').upper()} file. (Decoder message: {str(exc)[:160]})") from exc

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or len(y) == 0:
        raise AudioProblem("No audio samples were found in this file.")
    if not np.isfinite(y).all():
        raise AudioProblem("The decoded audio contains invalid (NaN/infinite) samples.")
    duration = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-12))

    if duration < MIN_SECONDS:
        raise AudioProblem(f"The recording is too short ({duration:.2f} s). Please upload at least {MIN_SECONDS:.0f} second of speech "
                           f"(the model was trained on recordings of 8–18 s).")
    if duration > MAX_SECONDS:
        raise AudioProblem(f"The recording is too long ({duration:.0f} s). Please upload at most {MAX_SECONDS:.0f} seconds.")
    if peak < SILENT_PEAK:
        raise AudioProblem("The recording is silent (no audible signal). Please upload a recording that contains speech.")

    warnings = []
    if duration < WARN_SHORT_SECONDS:
        warnings.append(f"Short recording ({duration:.1f} s): the model was trained on recordings of 8–18 s, so the result is less meaningful.")
    if duration > WARN_LONG_SECONDS:
        warnings.append(f"Long recording ({duration:.0f} s): the model was trained on 8–18 s recordings; the whole file is analysed.")
    if rms_db < QUIET_RMS_DBFS:
        warnings.append(f"The recording is very quiet (RMS {rms_db:.0f} dBFS) and may be mostly silence or noise.")
    return PreparedAudio(name=name, digest=digest_of(data), signal=y, sample_rate=int(sr), duration_s=float(duration),
                         original_format=ext.lstrip(".").upper(), original_sample_rate=props.get("sample_rate"),
                         original_channels=props.get("channels"), rms_dbfs=float(rms_db), peak=peak, warnings=warnings)


def classify(prepared: PreparedAudio, predictor) -> dict:
    """Real model inference on the standardised signal (no hard-coded or sample output)."""
    return predictor.predict_audio(prepared.signal, prepared.sample_rate)


# ----------------------------------------------------------------------------- facts shown in the app (read from files)
def model_facts(predictor) -> dict:
    labels = predictor.labels
    info = json.loads(FINAL_INFO.read_text(encoding="utf-8")) if FINAL_INFO.is_file() else {}
    maps = labels["maps"]
    comp = pd.read_csv(COMPARISON_CSV).set_index("run") if COMPARISON_CSV.is_file() else None
    return {"kind": labels["kind"], "n_features": len(labels["feature_columns"]), "accents": maps["accents"], "regions": maps["regions"],
            "accent_to_region": maps["accent_to_region"], "region_to_accents": maps["region_to_accents"],
            "n_params": int(predictor.model.count_params()),
            "n_trainable": int(sum(int(np.prod(w.shape)) for w in predictor.model.trainable_weights)),
            "n_fit": info.get("n_fit_recordings"), "n_validation": len(info.get("validation_recordings", [])), "comparison": comp}


def dataset_facts() -> dict:
    meta = pd.read_csv(cfg.METADATA_CSV)
    meta = meta[meta.conversion_status.astype(str).str.startswith("success")]
    per = meta.groupby(["region", "accent"], sort=False).agg(recordings=("recording_id", "count"), speakers=("speaker_id", "nunique"),
                                                            format=("original_format", lambda s: "/".join(sorted(set(s))))).reset_index()
    return {"n_recordings": len(meta), "n_accents": meta.accent.nunique(), "n_speakers": meta.speaker_id.nunique(),
            "per_accent": per, "per_speaker": int(meta.groupby("speaker_id").size().min())}
