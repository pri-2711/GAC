"""Phase 6 data layer: Log-Mel spectrograms, additional acoustic features, cross-validation folds,
training-fold-only normalisation and augmentation.

Everything is derived from the standardised WAV files and the Phase 3 feature table. Nothing from
Phases 1-5 is modified. Classes (accents, regions) are discovered from the data, never hard-coded.

Evaluation protocol (see the report): stratified K-fold over RECORDINGS where K = the number of
recordings of the smallest accent (7 now), so every recording is tested exactly once. Because there
is one speaker per accent, this is a WITHIN-SPEAKER protocol and cannot measure generalisation to
unseen speakers.
"""
from __future__ import annotations

import json

import librosa
import numpy as np
import pandas as pd

import audio_analysis as aa
import common
import config as cfg
import feature_analysis as fa
import mfcc_features as mf

OUT = cfg.RESULTS_DIR / "model_development"
SPEC_DIR = OUT / "spectrograms"
EXTRA_DIR = OUT / "additional_features"

# ---- Log-Mel configuration (single source of truth, saved next to the spectrograms) ----
SPEC = dict(sample_rate=16_000, n_mels=64, n_fft=512, win_length=400, hop_length=160, window="hann",
            power=2.0, fmin=0.0, fmax=8000.0, center=True, to_db="librosa.power_to_db(ref=1.0, top_db=80.0)",
            ref=1.0, top_db=80.0)
T_FRAMES = 800          # model input length: 8.0 s at a 10 ms hop (the shortest recording is 8.29 s)

# ---- Augmentation (training recordings only; see make_augmenter) ----
AUGMENT = dict(gain_db=3.0, noise_std=0.05, time_mask_max=40, time_mask_p=0.5,
               freq_mask_max=6, freq_mask_p=0.5, feature_jitter_std=0.10)


# --------------------------------------------------------------------------- #
# Recordings and labels
# --------------------------------------------------------------------------- #
def load_recordings() -> pd.DataFrame:
    """One row per successfully converted recording (Phase 1), ordered by configured accent then ID."""
    meta = common.load_metadata()
    order = {a: i for i, a in enumerate(cfg.ordered_accents(meta.accent))}
    meta = meta.assign(_o=meta.accent.map(order)).sort_values(["_o", "recording_id"]).drop(columns="_o")
    return meta.reset_index(drop=True)


def class_maps(meta: pd.DataFrame) -> dict:
    """Accent and region class lists (accents in config order; regions by first appearance) and the
    region -> accents mapping used by hierarchical inference. Nothing assumes one accent per region."""
    accents = cfg.ordered_accents(meta.accent)
    acc_region = meta.drop_duplicates("accent").set_index("accent")["region"].to_dict()
    regions = list(dict.fromkeys(acc_region[a] for a in accents))
    return {"accents": accents, "regions": regions, "accent_to_region": {a: acc_region[a] for a in accents},
            "region_to_accents": {r: [a for a in accents if acc_region[a] == r] for r in regions}}


def encode_labels(meta: pd.DataFrame, maps: dict) -> tuple[np.ndarray, np.ndarray]:
    ai = {a: i for i, a in enumerate(maps["accents"])}
    ri = {r: i for i, r in enumerate(maps["regions"])}
    return meta.region.map(ri).to_numpy(), meta.accent.map(ai).to_numpy()


# --------------------------------------------------------------------------- #
# Log-Mel spectrograms
# --------------------------------------------------------------------------- #
def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """(n_mels, n_frames) log-power Mel spectrogram in dB. Absolute level is kept (ref = 1.0)."""
    if sr != SPEC["sample_rate"]:
        raise ValueError(f"expected {SPEC['sample_rate']} Hz audio, got {sr} Hz")
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=SPEC["n_fft"], hop_length=SPEC["hop_length"], win_length=SPEC["win_length"],
        window=SPEC["window"], center=SPEC["center"], power=SPEC["power"], n_mels=SPEC["n_mels"],
        fmin=SPEC["fmin"], fmax=SPEC["fmax"])
    return librosa.power_to_db(S, ref=SPEC["ref"], top_db=SPEC["top_db"]).astype(np.float32)


def build_spectrograms(meta: pd.DataFrame, save: bool = True) -> dict[str, np.ndarray]:
    specs = {}
    for _, r in meta.iterrows():
        y, sr = aa.load_audio(cfg.PROJECT_ROOT / r["processed_file_path"])
        specs[r["recording_id"]] = compute_logmel(y, sr)
    if save:
        SPEC_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SPEC_DIR / "log_mel_spectrograms.npz", **specs)
        (SPEC_DIR / "spectrogram_config.json").write_text(json.dumps(
            {**SPEC, "model_input_frames_T": T_FRAMES, "model_input_shape": [SPEC["n_mels"], T_FRAMES, 1],
             "window_note": "25 ms Hann window (400 samples) zero-padded to an FFT of 512, hop 160 samples (10 ms)",
             "padding_cropping": "recordings longer than T are cropped (centre crop for evaluation, random crop for "
                                 "training); shorter ones are padded at the end with the training mean (0 after normalisation)",
             "normalisation": "per-Mel-band mean/std fitted on the TRAINING fold only"}, indent=2), encoding="utf-8")
        pd.DataFrame([{"recording_id": k, "n_mels": v.shape[0], "n_frames": v.shape[1], "duration_s": round(v.shape[1] * 0.01, 2),
                       "min_db": float(v.min()), "max_db": float(v.max()), "mean_db": float(v.mean())}
                      for k, v in specs.items()]).to_csv(SPEC_DIR / "spectrogram_manifest.csv", index=False)
    return specs


def load_spectrograms() -> dict[str, np.ndarray]:
    z = np.load(SPEC_DIR / "log_mel_spectrograms.npz")
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# Numerical features: Phase 3 schema (78, unchanged) + optional validated additions
# --------------------------------------------------------------------------- #
def load_feature_table() -> tuple[pd.DataFrame, list[str]]:
    """Phase 3 recording-level table and the acoustic columns listed in its schema."""
    schema = fa.load_schema()
    feats = pd.read_csv(fa.PHASE3_CSV)
    return feats, schema["features"]


EXTRA_FEATURES = [f"{base}_{s}" for base in ("spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "zero_crossing_rate")
                  for s in ("mean", "std")] + ["f0_mean_hz", "f0_std_hz", "f0_median_hz", "f0_p95_minus_p5_hz", "voiced_fraction"]


def extract_extra_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Spectral shape (centroid, bandwidth, roll-off), zero-crossing rate and F0 (pYIN) statistics.

    Same framing as Phase 3 (512 FFT, 400 window, 160 hop). F0 statistics use voiced frames only
    (pYIN, 65-400 Hz, which covers adult speech); if too few frames are voiced the F0 values are NaN and the
    validation reports it. RMS energy is deliberately NOT added: it is a recording-level quantity
    (MFCC 1 already carries it) that Phases 2-4 showed is confounded with the recording setup.
    Formants are not extracted: no reliable formant tracker is available without an external tool.
    """
    kw = dict(n_fft=512, hop_length=160, win_length=400, window="hann")
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, **kw)[0]
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, **kw)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=400, hop_length=160)[0]
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    v = f0[voiced & np.isfinite(f0)]
    row = {"spectral_centroid_hz_mean": cen.mean(), "spectral_centroid_hz_std": cen.std(),
           "spectral_bandwidth_hz_mean": bw.mean(), "spectral_bandwidth_hz_std": bw.std(),
           "spectral_rolloff_hz_mean": ro.mean(), "spectral_rolloff_hz_std": ro.std(),
           "zero_crossing_rate_mean": zcr.mean(), "zero_crossing_rate_std": zcr.std(),
           "voiced_fraction": float(np.mean(voiced))}
    if len(v) >= 20:
        row.update(f0_mean_hz=v.mean(), f0_std_hz=v.std(), f0_median_hz=float(np.median(v)),
                   f0_p95_minus_p5_hz=float(np.percentile(v, 95) - np.percentile(v, 5)))
    else:
        row.update(f0_mean_hz=np.nan, f0_std_hz=np.nan, f0_median_hz=np.nan, f0_p95_minus_p5_hz=np.nan)
    return {k: float(row[k]) for k in EXTRA_FEATURES}


def build_extra_features(meta: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    rows = []
    for _, r in meta.iterrows():
        y, sr = aa.load_audio(cfg.PROJECT_ROOT / r["processed_file_path"])
        rows.append({"recording_id": r["recording_id"], **extract_extra_features(y, sr)})
    df = pd.DataFrame(rows)
    if save:
        EXTRA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(EXTRA_DIR / "additional_acoustic_features.csv", index=False)
        (EXTRA_DIR / "additional_features_schema.json").write_text(json.dumps({
            "columns": EXTRA_FEATURES, "separate_from_phase3_schema": True,
            "not_extracted": {"rms_energy": "recording-level, confounded with the recording setup (Phases 2-4); MFCC 1 already carries level",
                              "formants": "no reliable tracker available without external software"},
            "f0": "pYIN, 65-400 Hz, statistics over voiced frames"}, indent=2), encoding="utf-8")
    return df


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def make_folds(meta: pd.DataFrame, seed: int = 42, n_folds: int | None = None) -> list[dict]:
    """Stratified (by accent) folds over recordings.

    n_folds defaults to the size of the smallest accent, so with 7 recordings per accent each fold
    tests one recording of every accent. Within the remaining training recordings, one recording per
    accent (when an accent has >= 3 of them) is held back as the VALIDATION set used for early
    stopping / checkpoint selection; the rest is the training set. Everything is by RECORDING:
    augmented copies are generated later, inside the training set only.
    """
    rng = np.random.default_rng(seed)
    per = {a: list(g.recording_id) for a, g in meta.groupby("accent", sort=False)}
    k = n_folds or min(len(v) for v in per.values())
    test = [[] for _ in range(k)]
    for a, ids in per.items():
        ids = list(rng.permutation(ids))
        for j, rid in enumerate(ids):
            test[j % k].append(rid)
    folds = []
    for f in range(k):
        rest = [r for r in meta.recording_id if r not in set(test[f])]
        val = []
        for a, ids in per.items():
            cand = [r for r in ids if r in rest]
            if len(cand) >= 3:
                val.append(cand[int(rng.integers(len(cand)))])
        train = [r for r in rest if r not in set(val)]
        folds.append({"fold": f + 1, "train": train, "val": val, "test": sorted(test[f])})
    return folds


def final_split(meta: pd.DataFrame, seed: int = 42) -> dict:
    """Split for the deployable final model: ALL recordings are used for fitting except one validation
    recording per accent (checkpoint selection). There is no test set, so no accuracy is claimed for it."""
    rng = np.random.default_rng(seed + 999)
    val = [str(rng.choice(list(g.recording_id))) for _, g in meta.groupby("accent", sort=False)]
    return {"fold": 0, "train": [r for r in meta.recording_id if r not in set(val)], "val": val, "test": []}


def folds_table(folds: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"fold": f["fold"], "recording_id": r, "role": role}
                         for f in folds for role in ("train", "val", "test") for r in f[role]])


# --------------------------------------------------------------------------- #
# Normalisation (fitted on the training recordings of ONE fold only)
# --------------------------------------------------------------------------- #
def fit_feature_scaler(X_train: np.ndarray) -> dict:
    sd = X_train.std(axis=0)
    return {"mean": X_train.mean(axis=0), "std": np.where(sd > 1e-8, sd, 1.0)}


def apply_feature_scaler(X: np.ndarray, sc: dict) -> np.ndarray:
    return ((X - sc["mean"]) / sc["std"]).astype(np.float32)


def fit_spec_scaler(specs: list[np.ndarray]) -> dict:
    """Per-Mel-band mean and std over all frames of the training recordings."""
    cat = np.concatenate(specs, axis=1)
    sd = cat.std(axis=1)
    return {"mean": cat.mean(axis=1).astype(np.float32), "std": np.where(sd > 1e-6, sd, 1.0).astype(np.float32)}


def normalise_spec(spec: np.ndarray, sc: dict) -> np.ndarray:
    return ((spec - sc["mean"][:, None]) / sc["std"][:, None]).astype(np.float32)


def fit_crop(spec: np.ndarray, T: int, mode: str = "center", rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop (centre or random) or end-pad with 0 (= the training mean after normalisation) to exactly T frames."""
    n = spec.shape[1]
    if n >= T:
        start = (n - T) // 2 if mode == "center" or rng is None else int(rng.integers(0, n - T + 1))
        return spec[:, start:start + T]
    out = np.zeros((spec.shape[0], T), dtype=spec.dtype)
    out[:, :n] = spec
    return out


# --------------------------------------------------------------------------- #
# Augmentation (training recordings only; applied on the fly, in the training fold)
# --------------------------------------------------------------------------- #
def augment_spec(raw_db: np.ndarray, sc: dict, T: int, rng: np.random.Generator, cfg_aug: dict = AUGMENT) -> np.ndarray:
    """One random mild variant of a training recording's spectrogram, normalised and cropped to T.

    - random crop (a small time shift when the recording is longer than T),
    - gain change of up to +-gain_db (a constant shift in dB),
    - small Gaussian jitter (std in normalised units),
    - at most one mild time mask and one mild frequency mask.
    No pitch or speed change (those would distort accent-related information).
    """
    s = raw_db + rng.uniform(-cfg_aug["gain_db"], cfg_aug["gain_db"])
    s = fit_crop(normalise_spec(s, sc), T, "random", rng)
    s = s + rng.normal(0, cfg_aug["noise_std"], s.shape).astype(np.float32)
    if rng.random() < cfg_aug["time_mask_p"]:
        w = int(rng.integers(1, cfg_aug["time_mask_max"] + 1))
        a = int(rng.integers(0, max(1, T - w)))
        s[:, a:a + w] = 0.0
    if rng.random() < cfg_aug["freq_mask_p"]:
        w = int(rng.integers(1, cfg_aug["freq_mask_max"] + 1))
        a = int(rng.integers(0, max(1, s.shape[0] - w)))
        s[a:a + w, :] = 0.0
    return s


def augment_features(x: np.ndarray, rng: np.random.Generator, cfg_aug: dict = AUGMENT) -> np.ndarray:
    """Small Gaussian jitter on (already normalised) numerical features. This is feature-space jitter, not
    features re-extracted from augmented audio."""
    return (x + rng.normal(0, cfg_aug["feature_jitter_std"], x.shape)).astype(np.float32)
