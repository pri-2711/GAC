"""Phase 2 – descriptive audio quality analysis (read-only; never modifies audio).

All thresholds are named constants below so they are easy to review and change.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

import config as cfg

RESULTS_DIR = cfg.PROJECT_ROOT / "results" / "audio_analysis"

# Framing for energy analysis (25 ms window, 10 ms hop at 16 kHz)
FRAME_LENGTH = 400
HOP_LENGTH = 160
# Energy classes are defined relative to each file's own loud-speech level
# (99th-percentile frame level), so they are robust to different recording gains.
SPEECH_DB_BELOW_REF = 30      # frame >= ref-30 dB  -> speech-level energy
SILENCE_DB_BELOW_REF = 45     # frame <  ref-45 dB  -> silence; in between -> quiet/low-energy
MIN_PAUSE_S = 0.20            # a gap of non-speech frames this long counts as an internal pause
MIN_SPEECH_S = 0.05           # speech bursts shorter than this are ignored (clicks)
# Quality thresholds
CLIP_LEVEL = 0.999
LOW_PEAK_DBFS = -20.0
LOW_SPEECH_RMS_DBFS = -35.0
NEAR_SILENT_SPEECH_RATIO = 0.10
DURATION_RATIO_LIMITS = (0.5, 2.0)   # vs. median duration of all recordings
LEVEL_DEVIATION_DB = 6.0             # speech RMS vs. median across all recordings


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load without resampling or normalising (sr=None keeps the native 16 kHz)."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def _dbfs(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def _runs(mask: np.ndarray):
    """Yield (start, end_exclusive) for consecutive True runs."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def frame_energy(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-10))


def speech_activity(y: np.ndarray, sr: int) -> dict:
    """Energy-based activity analysis. Nothing is removed from the signal."""
    db = frame_energy(y)
    ref = np.percentile(db, 99)
    speech = db >= ref - SPEECH_DB_BELOW_REF
    silence = db < ref - SILENCE_DB_BELOW_REF
    quiet = ~speech & ~silence
    frame_s = HOP_LENGTH / sr

    # Drop very short bursts so isolated clicks are not treated as speech
    for s, e in _runs(speech):
        if (e - s) * frame_s < MIN_SPEECH_S:
            speech[s:e] = False
    idx = np.flatnonzero(speech)
    if idx.size == 0:
        return dict(leading_silence_s=np.nan, trailing_silence_s=np.nan, n_internal_pauses=0,
                    internal_pause_total_s=0.0, longest_pause_s=0.0, speech_ratio=0.0,
                    low_energy_frame_pct=100.0, quiet_frame_pct=0.0, silence_frame_pct=100.0,
                    noise_floor_dbfs=float(np.percentile(db, 10)), speech_level_dbfs=float(ref),
                    approx_snr_db=np.nan, frame_db=db, speech_mask=speech, ref_db=ref, frame_s=frame_s)
    first, last = idx[0], idx[-1]
    pauses = [(s, e) for s, e in _runs(~speech[first:last + 1]) if (e - s) * frame_s >= MIN_PAUSE_S]
    pause_lens = [(e - s) * frame_s for s, e in pauses]
    noise_floor = float(np.percentile(db, 10))
    return dict(
        leading_silence_s=first * frame_s,
        trailing_silence_s=(len(db) - 1 - last) * frame_s,
        n_internal_pauses=len(pauses),
        internal_pause_total_s=float(sum(pause_lens)),
        longest_pause_s=float(max(pause_lens, default=0.0)),
        speech_ratio=float(speech.mean()),
        low_energy_frame_pct=float((~speech).mean() * 100),
        quiet_frame_pct=float(quiet.mean() * 100),
        silence_frame_pct=float(silence.mean() * 100),
        noise_floor_dbfs=noise_floor,
        speech_level_dbfs=float(ref),
        approx_snr_db=float(np.median(db[speech]) - noise_floor),
        frame_db=db, speech_mask=speech, ref_db=ref, frame_s=frame_s)


def quality_metrics(y: np.ndarray, sr: int, activity: dict) -> dict:
    peak = float(np.max(np.abs(y)))
    clipped = np.abs(y) >= CLIP_LEVEL
    clip_runs = [r for r in _runs(clipped) if r[1] - r[0] >= 3]
    speech_mask = activity["speech_mask"]
    # RMS over speech-active frames only (frame-level energies converted back to power)
    if speech_mask.any():
        speech_rms = float(np.sqrt(np.mean(10 ** (activity["frame_db"][speech_mask] / 10))))
    else:
        speech_rms = 1e-10
    return dict(
        duration_s=len(y) / sr,
        peak_amplitude=peak, peak_dbfs=_dbfs(peak),
        overall_rms_dbfs=_dbfs(float(np.sqrt(np.mean(y ** 2)))),
        speech_rms_dbfs=_dbfs(speech_rms),
        dc_offset=float(np.mean(y)),
        clipped_sample_pct=float(clipped.mean() * 100),
        clipping_events=len(clip_runs))


def analyse_recording(path: Path) -> tuple[dict, dict]:
    y, sr = load_audio(path)
    act = speech_activity(y, sr)
    q = quality_metrics(y, sr, act)
    scalar = {k: v for k, v in act.items() if k not in ("frame_db", "speech_mask", "ref_db", "frame_s")}
    return {**q, **scalar, "sample_rate": sr}, act


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag columns and a readable 'quality_flags' string."""
    df = df.copy()
    med_dur = df["duration_s"].median()
    med_level = df["speech_rms_dbfs"].median()
    df["flag_clipping"] = df["clipped_sample_pct"] > 0  # any sample at full scale (>= CLIP_LEVEL)
    df["flag_low_amplitude"] = (df["peak_dbfs"] < LOW_PEAK_DBFS) | (df["speech_rms_dbfs"] < LOW_SPEECH_RMS_DBFS)
    df["flag_near_silent"] = (df["speech_ratio"] < NEAR_SILENT_SPEECH_RATIO) | (df["overall_rms_dbfs"] < -60)
    lo, hi = DURATION_RATIO_LIMITS
    df["flag_unusual_duration"] = (df["duration_s"] < lo * med_dur) | (df["duration_s"] > hi * med_dur)
    df["level_dev_db"] = df["speech_rms_dbfs"] - med_level
    df["flag_level_inconsistent"] = df["level_dev_db"].abs() > LEVEL_DEVIATION_DB
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["quality_flags"] = df[flag_cols].apply(
        lambda r: ", ".join(c[5:] for c in flag_cols if r[c]), axis=1)
    return df


def representative_recordings(meta: pd.DataFrame) -> pd.DataFrame:
    """Per accent: the recording whose duration is closest to that accent's median
    (deterministic, avoids cherry-picking)."""
    picks = []
    for accent in [a.accent for a in cfg.ACCENTS]:
        sub = meta[meta["accent"] == accent]
        if sub.empty:
            continue
        picks.append(sub.loc[(sub["duration_seconds"] - sub["duration_seconds"].median()).abs().idxmin()])
    return pd.DataFrame(picks)
