"""Phase 3 – MFCC feature extraction (read-only on audio).

Reuses config.py (paths, accents), audio_analysis.load_audio (native-rate loading)
and the Phase 1 metadata / Phase 2 quality flags. Audio files are never modified.

Naming note: librosa returns 13 rows (indices 0..12). We label them MFCC 1..13 in
files and figures, i.e. MFCC 1 = librosa index 0 (the "c0" coefficient, which mostly
reflects overall spectral energy / loudness).
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

import audio_analysis as aa
import common
import config as cfg

MFCC_DIR = cfg.MFCC_DIR
FRAME_DIR = MFCC_DIR / "frame_features"
FIG_DIR = MFCC_DIR / "figures"
PHASE2_METRICS_CSV = cfg.PHASE2_METRICS_CSV

# ---- Extraction parameters (single source of truth; saved inside every NPZ) ----
PARAMS = dict(
    sample_rate=16_000, n_mfcc=13, n_fft=512, hop_length=160, win_length=400,
    window="hann", dct_type=2, norm="ortho",
    n_mels=128,          # librosa default, written down explicitly for reproducibility
    center=True,         # librosa default: frames are centred on their time stamp
    delta_width=9,       # librosa default: regression over 9 frames (~90 ms)
)
N_COEF = PARAMS["n_mfcc"]
FEATURE_GROUPS = {"mfcc": "MFCC", "delta": "Delta MFCC", "delta2": "Delta-Delta MFCC"}
META_COLUMNS = ["recording_id", "speaker_id", "accent", "region", "original_format"]
FLAG_COLUMNS = ["quality_flags", "flag_clipping", "flag_low_amplitude", "flag_near_silent",
                "flag_unusual_duration", "flag_level_inconsistent"]


def coef_label(i: int) -> str:
    """0-based array row -> 'MFCC 1'..'MFCC 13' (1-based label)."""
    return f"MFCC {i + 1}"


def feature_column_names() -> list[str]:
    cols = []
    for group in FEATURE_GROUPS:                 # mfcc, delta, delta2
        for i in range(N_COEF):
            for stat in ("mean", "std"):
                cols.append(f"{group}_{i + 1:02d}_{stat}")
    return cols


def extract_frame_features(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Frame-level MFCC, delta and delta-delta matrices, each shaped (13, n_frames)."""
    if sr != PARAMS["sample_rate"]:
        raise ValueError(f"expected {PARAMS['sample_rate']} Hz audio, got {sr} Hz")
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=PARAMS["n_mfcc"], n_fft=PARAMS["n_fft"],
        hop_length=PARAMS["hop_length"], win_length=PARAMS["win_length"],
        window=PARAMS["window"], dct_type=PARAMS["dct_type"], norm=PARAMS["norm"],
        n_mels=PARAMS["n_mels"], center=PARAMS["center"])
    delta = librosa.feature.delta(mfcc, width=PARAMS["delta_width"], order=1)
    delta2 = librosa.feature.delta(mfcc, width=PARAMS["delta_width"], order=2)
    return {"mfcc": mfcc, "delta": delta, "delta2": delta2}


def recording_level_stats(features: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean and std (over frames) of every coefficient of every feature group."""
    row = {}
    for group, mat in features.items():
        means, stds = mat.mean(axis=1), mat.std(axis=1, ddof=0)
        for i in range(N_COEF):
            row[f"{group}_{i + 1:02d}_mean"] = float(means[i])
            row[f"{group}_{i + 1:02d}_std"] = float(stds[i])
    return row


def save_frame_npz(rec_id: str, meta_row: pd.Series, features: dict, n_samples: int) -> Path:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAME_DIR / f"{rec_id}.npz"
    frame_times = librosa.frames_to_time(np.arange(features["mfcc"].shape[1]),
                                         sr=PARAMS["sample_rate"], hop_length=PARAMS["hop_length"])
    np.savez_compressed(
        path, mfcc=features["mfcc"], delta=features["delta"], delta2=features["delta2"],
        frame_times=frame_times, n_samples=np.int64(n_samples),
        recording_id=np.str_(rec_id), speaker_id=np.str_(meta_row["speaker_id"]),
        accent=np.str_(meta_row["accent"]), region=np.str_(meta_row["region"]),
        **{f"param_{k}": np.array(v) for k, v in PARAMS.items()})
    return path


def load_frame_npz(rec_id: str) -> dict[str, np.ndarray]:
    with np.load(FRAME_DIR / f"{rec_id}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_metadata_with_flags() -> tuple[pd.DataFrame, list[str]]:
    """Phase 1 metadata joined with Phase 2 quality flags (by recording_id).

    Returns (metadata, ids_without_phase2_flags). A recording added after Phase 2 was
    run (e.g. future Surti files) is kept, with empty flags, and reported.
    """
    meta = common.load_metadata()
    p2 = common.load_quality_metrics(keep_empty_strings=True)
    keep = ["recording_id"] + FLAG_COLUMNS + ["speech_rms_dbfs", "noise_floor_dbfs", "peak_dbfs"]
    merged = meta.merge(p2[keep], on="recording_id", how="left")
    missing = merged.loc[merged["quality_flags"].isna(), "recording_id"].tolist()
    return merged, missing


def extract_all(meta: pd.DataFrame):
    """Extract features for every recording. Nothing is skipped silently: any
    failure is recorded and re-raised after the loop so it cannot go unnoticed.

    Returns (feature_df, frame_store, errors); frame_store maps recording_id -> arrays.
    """
    rows, store, errors = [], {}, []
    for _, r in meta.iterrows():
        rec_id = r["recording_id"]
        try:
            y, sr = aa.load_audio(cfg.PROJECT_ROOT / r["processed_file_path"])
            feats = extract_frame_features(y, sr)
            save_frame_npz(rec_id, r, feats, len(y))
            store[rec_id] = {**feats, "n_samples": len(y)}
            rows.append({**{c: r[c] for c in META_COLUMNS},
                         **{c: r[c] for c in FLAG_COLUMNS},
                         **recording_level_stats(feats)})
        except Exception as exc:
            errors.append((rec_id, f"{type(exc).__name__}: {exc}"))
    df = pd.DataFrame(rows)
    ordered = META_COLUMNS + FLAG_COLUMNS + feature_column_names()
    return df[ordered], store, errors


def save_feature_table(df: pd.DataFrame) -> tuple[Path, Path]:
    MFCC_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MFCC_DIR / "mfcc_recording_features.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    cols_path = MFCC_DIR / "mfcc_feature_columns.json"
    cols_path.write_text(json.dumps({
        "metadata_columns_NOT_for_model_input": META_COLUMNS + FLAG_COLUMNS,
        "acoustic_feature_columns": feature_column_names(),
        "extraction_parameters": PARAMS,
        "note": "MFCC 1 = librosa coefficient index 0 (c0). Never use metadata columns as inputs."},
        indent=2), encoding="utf-8")
    return csv_path, cols_path


def accent_summary(df: pd.DataFrame, store: dict) -> pd.DataFrame:
    """Long-format summary per accent x feature group x coefficient.

    mean/std/min/max across recordings use the recording-level means (n recordings).
    pooled_frame_* use every frame of the accent (longer recordings count more).
    """
    rows = []
    for accent in cfg.ordered_accents(df["accent"]):
        sub = df[df["accent"] == accent]
        if sub.empty:
            continue
        for group in FEATURE_GROUPS:
            pooled = np.concatenate([store[i][group] for i in sub["recording_id"]], axis=1)
            for k in range(N_COEF):
                m = sub[f"{group}_{k + 1:02d}_mean"]
                rows.append(dict(
                    accent=accent, feature_group=group, coefficient=k + 1,
                    n_recordings=len(sub), n_frames=pooled.shape[1],
                    mean_of_recording_means=m.mean(),
                    std_across_recordings=m.std(ddof=1) if len(sub) > 1 else np.nan,
                    min_recording_mean=m.min(), max_recording_mean=m.max(),
                    mean_within_recording_std=sub[f"{group}_{k + 1:02d}_std"].mean(),
                    pooled_frame_mean=pooled[k].mean(), pooled_frame_std=pooled[k].std()))
    return pd.DataFrame(rows)


def validate(meta: pd.DataFrame, df: pd.DataFrame, store: dict, errors: list,
             missing_flags: list) -> tuple[list[tuple[str, bool, str]], bool]:
    """Run all Phase 3 checks. Returns (list of (check, passed, detail), all_passed)."""
    checks = common.CheckList()
    add = checks.add

    add("No extraction errors", not errors, f"{len(errors)} error(s): {errors}" if errors else "")
    add("All metadata recordings processed", set(meta["recording_id"]) == set(store),
        f"metadata={len(meta)}, processed={len(store)}")
    add("Feature table has one row per recording, IDs match metadata",
        len(df) == len(meta) and set(df["recording_id"]) == set(meta["recording_id"]) and df["recording_id"].is_unique)
    ok_shapes, ok_finite, bad = True, True, []
    for rid, s in store.items():
        n_expected = 1 + s["n_samples"] // PARAMS["hop_length"]
        for g in FEATURE_GROUPS:
            m = s[g]
            if m.shape != (N_COEF, n_expected):
                ok_shapes = False; bad.append(f"{rid}:{g}:{m.shape}!=({N_COEF},{n_expected})")
            if not np.isfinite(m).all():
                ok_finite = False; bad.append(f"{rid}:{g}:non-finite")
        if not (s["mfcc"].shape == s["delta"].shape == s["delta2"].shape):
            ok_shapes = False; bad.append(f"{rid}: delta shape mismatch")
    add("Every recording has 13 coefficients; MFCC/delta/delta-delta shapes agree with frame count", ok_shapes, "; ".join(bad))
    add("No NaN or infinite values in any frame-level matrix", ok_finite)
    add("No NaN or infinite values in recording-level features", np.isfinite(df[feature_column_names()].to_numpy()).all())
    add("Feature table has 78 acoustic columns (13 x 2 stats x 3 groups)", len(feature_column_names()) == 78 and
        all(c in df.columns for c in feature_column_names()))
    # NPZ files exist, trace to recording IDs, and carry identical parameters
    npz_ok, param_ok, detail = True, True, []
    for rid in meta["recording_id"]:
        p = FRAME_DIR / f"{rid}.npz"
        if not p.is_file():
            npz_ok = False; detail.append(f"missing {p.name}"); continue
        z = load_frame_npz(rid)
        if str(z["recording_id"]) != rid:
            npz_ok = False; detail.append(f"{rid}: id mismatch")
        for k, v in PARAMS.items():
            if z[f"param_{k}"].item() != v:
                param_ok = False; detail.append(f"{rid}: param {k}")
    add("A frame-level NPZ exists for each recording and stores its recording ID", npz_ok, "; ".join(detail))
    add("Extraction parameters identical across all recordings (stored in each NPZ)", param_ok)
    add("Phase 2 quality flags available for every recording", not missing_flags,
        f"missing flags for: {missing_flags}" if missing_flags else "")
    add("Flagged recordings were kept (none removed)", len(df) == len(meta),
        f"{int((df['quality_flags'].astype(str) != '').sum())} flagged recordings retained")
    return checks, checks.passed


def format_validation_summary(meta, df, store, errors, checks, passed) -> str:
    per_accent = {a: int((df["accent"] == a).sum()) for a in cfg.ordered_accents(cfg.configured_accents() + list(df["accent"]))}
    frames = {rid: s["mfcc"].shape[1] for rid, s in store.items()}
    L = ["Phase 3 – MFCC extraction and validation summary", "=" * 50, "",
         "Extraction parameters:"]
    L += [f"  {k:<12}: {v}" for k, v in PARAMS.items()]
    L += ["", f"Recordings in metadata : {len(meta)}", f"Recordings processed   : {len(store)}",
          f"Extraction errors      : {len(errors)}", "", "Recordings per accent:"]
    for accent, n in per_accent.items():
        L.append(f"  {accent:<12}: {n}" + ("  (pending future collection)" if n == 0 else ""))
    L += ["", f"Frames per recording   : min {min(frames.values())}, max {max(frames.values())}, "
              f"total {sum(frames.values())}",
          f"Features per recording : 13 coefficients x (MFCC, delta, delta-delta) x (mean, std) = 78",
          f"Recordings with Phase 2 quality flags kept: {int((df['quality_flags'].astype(str) != '').sum())}", "",
          "Validation checks:"]
    L += common.CheckList(checks).format_lines(arrow="  -> ")
    L += ["", f"OVERALL: {'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}"]
    return "\n".join(L) + "\n"
