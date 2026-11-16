"""Phase 5 - Principal Component Analysis (PCA) of the Phase 3 recording-level MFCC features.

PCA here is an EXPLORATORY dimensionality-reduction tool, not an accent classifier. With one
speaker per accent, accent, speaker, device and wording are inseparable, so nothing computed in
this module can show that accents are separable in a speaker-independent way.

Design choices
* Reads the Phase 1-4 outputs only; nothing from earlier phases is modified.
* Accents, formats and feature columns are discovered from the data (schema JSON), never hard-coded.
* Only the acoustic columns listed in mfcc_feature_columns.json enter PCA; metadata is kept aside.
* numpy only (SVD of the standardised matrix) - no extra dependency. StandardScaler semantics
  (population std, ddof=0) and PCA semantics (variance with n-1) are reproduced explicitly.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

import audio_utils as au
import common
import config as cfg
import feature_analysis as fa

PCA_DIR = cfg.RESULTS_DIR / "pca"          # Phase 5
FIG_DIR = PCA_DIR / "figures"
SPEECH_30 = fa.FA_DIR / "speech_only_recording_features.csv"        # Phase 4, standard 30 dB rule
SPEECH_20 = fa.FA_DIR / "speech_only_recording_features_ref20.csv"  # Phase 4, 20 dB sensitivity rule
EXCLUDED_BASE = ("mfcc_01", "mfcc_02")      # sensitivity: MFCC 1 and 2 (mean and std)
VARIANCE_TARGETS = (0.80, 0.90, 0.95)
N_LEADING = 5                                # components used for distance-style summaries
CONDITION_COLUMNS = {                        # column -> label (all are METADATA, never PCA input)
    "speech_ratio": "Speech-frame share (30 dB rule)",
    "pct_speech_frames_ref20": "Speech-frame share (20 dB rule, %)",
    "noise_floor_dbfs": "Noise floor (dBFS)",
    "approx_snr_db": "Approx. SNR (dB)",
    "speech_level_dbfs": "Speech level (dBFS)",
    "overall_rms_dbfs": "Overall RMS level (dBFS)",
    "duration_s": "Duration (s)",
    "original_sample_rate_hz": "Original sample rate (Hz)",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pca_inputs() -> dict:
    """Phase 3 features + metadata + quality metrics, plus the Phase 4 speech-only tables."""
    d = fa.load_inputs()
    table, features = d["table"].copy(), d["features"]
    checks = d["checks"]

    # Original file properties (sample rate is not stored in the Phase 1 metadata, which describes the WAVs).
    meta = d["meta"].set_index("recording_id")
    rates = {}
    for rid in table.recording_id:
        rates[rid] = au.inspect_original(cfg.PROJECT_ROOT / meta.loc[rid, "original_file_path"])["sample_rate"]
    table["original_sample_rate_hz"] = table.recording_id.map(rates)
    checks.add("Original sample rate found for every recording", table.original_sample_rate_hz.notna().all())

    p2 = d["p2"][["recording_id", "approx_snr_db", "overall_rms_dbfs", "speech_level_dbfs"]]
    table = table.merge(p2, on="recording_id", how="left")

    speech = {}
    for name, path in (("speech_30", SPEECH_30), ("speech_20", SPEECH_20)):
        s = pd.read_csv(path)
        same_ids = set(s.recording_id) == set(table.recording_id)
        same_cols = all(c in s.columns for c in features)
        checks.add(f"Phase 4 {path.name}: same recordings and all {len(features)} feature columns", same_ids and same_cols)
        speech[name] = s.set_index("recording_id").loc[table.recording_id].reset_index()
    table["pct_speech_frames_ref20"] = speech["speech_20"]["pct_speech_frames"].to_numpy()
    d.update(table=table, speech=speech)
    return d


# --------------------------------------------------------------------------- #
# PCA (numpy implementation of StandardScaler + PCA)
# --------------------------------------------------------------------------- #
def prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Feature matrix as float; refuses NaN/inf; drops (and reports) zero-variance columns.

    A column with no variance cannot be standardised (division by zero) and carries no information.
    """
    X = df[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(f"Non-finite feature values at (row, column) = {bad[:5].tolist()} ...")
    keep = X.std(axis=0) > 0
    dropped = [f for f, k in zip(features, keep) if not k]
    return X[:, keep], [f for f, k in zip(features, keep) if k], dropped


def fit_pca(X: np.ndarray, features: list[str]) -> dict:
    """Standardise (mean 0, population std 1) then PCA via SVD. Deterministic sign convention:
    the loading with the largest absolute value in each component is made positive."""
    n, p = X.shape
    mean, scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    Z = (X - mean) / scale
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    k = min(n - 1, p)                                    # centred data has rank <= n-1
    s, vt = s[:k], vt[:k]
    flip = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
    flip[flip == 0] = 1
    vt = vt * flip[:, None]
    eigval = s ** 2 / (n - 1)
    total = (Z ** 2).sum() / (n - 1)
    return {"features": list(features), "mean": mean, "scale": scale, "components": vt,
            "eigenvalues": eigval, "evr": eigval / total, "n_train": n, "scores": Z @ vt.T}


def transform(model: dict, X: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler + PCA to new rows (same feature order as model['features'])."""
    return ((X - model["mean"]) / model["scale"]) @ model["components"].T


def components_for(evr: np.ndarray, targets=VARIANCE_TARGETS) -> dict[float, int | None]:
    cum = np.cumsum(evr)
    return {t: (int(np.searchsorted(cum, t - 1e-12) + 1) if cum[-1] >= t - 1e-12 else None) for t in targets}


def save_model(model: dict, path=None) -> tuple:
    path = path or PCA_DIR / "pca_model.npz"
    np.savez(path, mean=model["mean"], scale=model["scale"], components=model["components"],
             eigenvalues=model["eigenvalues"], explained_variance_ratio=model["evr"])
    meta = path.with_name("pca_model_meta.json")
    meta.write_text(json.dumps({
        "description": "Fitted StandardScaler (population std) + PCA (SVD) on the Phase 3 recording-level features. "
                       "Apply with: ((X - mean) / scale) @ components.T",
        "feature_order": model["features"], "n_training_recordings": model["n_train"],
        "n_components": int(model["components"].shape[0])}, indent=2), encoding="utf-8")
    return path, meta


def load_model(path=None) -> dict:
    path = path or PCA_DIR / "pca_model.npz"
    z = np.load(path)
    meta = json.loads(path.with_name("pca_model_meta.json").read_text(encoding="utf-8"))
    return {"features": meta["feature_order"], "mean": z["mean"], "scale": z["scale"], "components": z["components"],
            "eigenvalues": z["eigenvalues"], "evr": z["explained_variance_ratio"], "n_train": meta["n_training_recordings"]}


# --------------------------------------------------------------------------- #
# Result tables
# --------------------------------------------------------------------------- #
def pc_names(k: int) -> list[str]:
    return [f"PC{i + 1}" for i in range(k)]


def variance_table(model: dict) -> pd.DataFrame:
    k = len(model["evr"])
    return pd.DataFrame({"component": pc_names(k), "eigenvalue": model["eigenvalues"],
                         "explained_variance_ratio": model["evr"],
                         "cumulative_explained_variance": np.cumsum(model["evr"])})


def loadings_table(model: dict) -> pd.DataFrame:
    k = model["components"].shape[0]
    df = pd.DataFrame(model["components"].T, columns=pc_names(k))
    df.insert(0, "feature", model["features"])
    df.insert(1, "label", [fa.feature_label(f) for f in model["features"]])
    df.insert(2, "feature_group", [fa.split_feature_name(f)[0] for f in model["features"]])
    df.insert(3, "statistic", [fa.split_feature_name(f)[2] for f in model["features"]])
    return df


def group_contributions(loadings: pd.DataFrame, n_pc: int = N_LEADING) -> pd.DataFrame:
    """Share of each PC's squared loadings (they sum to 1) carried by each feature family x statistic."""
    rows = []
    for pc in pc_names(n_pc):
        sq = loadings[pc] ** 2
        for (grp, stat), idx in loadings.groupby(["feature_group", "statistic"]).groups.items():
            rows.append({"component": pc, "feature_group": grp, "statistic": stat,
                         "share_of_squared_loadings": float(sq.loc[idx].sum())})
    return pd.DataFrame(rows)


def top_loadings(loadings: pd.DataFrame, n_pc: int = 5, top: int = 10) -> pd.DataFrame:
    rows = []
    for pc in pc_names(n_pc):
        order = loadings[pc].abs().sort_values(ascending=False).index[:top]
        for rank, i in enumerate(order, 1):
            rows.append({"component": pc, "rank": rank, "feature": loadings.feature[i], "label": loadings.label[i],
                         "loading": loadings.loc[i, pc]})
    return pd.DataFrame(rows)


def scores_table(table: pd.DataFrame, model: dict, id_cols: list[str]) -> pd.DataFrame:
    sc = pd.DataFrame(model["scores"], columns=pc_names(model["scores"].shape[1]))
    return pd.concat([table[id_cols].reset_index(drop=True), sc], axis=1)


# --------------------------------------------------------------------------- #
# Descriptive structure metrics (no significance tests; see module docstring)
# --------------------------------------------------------------------------- #
def silhouette(points: np.ndarray, labels) -> float:
    """Mean silhouette (-1..1) with Euclidean distance; labels with a single member contribute 0."""
    labels = np.asarray(labels)
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    out = []
    for i in range(len(points)):
        same = labels == labels[i]
        if same.sum() < 2:
            out.append(0.0)
            continue
        a = d[i, same & (np.arange(len(points)) != i)].mean()
        b = min(d[i, labels == g].mean() for g in set(labels) if g != labels[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


def eta2_by_pc(scores: np.ndarray, groupings: dict[str, np.ndarray], n_pc: int) -> pd.DataFrame:
    rows = []
    for i in range(n_pc):
        row = {"component": f"PC{i + 1}"}
        for name, labels in groupings.items():
            row[f"eta2_{name}"] = fa.eta_squared(scores[:, i], labels)
        rows.append(row)
    return pd.DataFrame(rows)


def one_vs_rest_eta2(scores: np.ndarray, accents_col: np.ndarray, accents: list[str], n_pc: int) -> pd.DataFrame:
    rows = []
    for i in range(n_pc):
        row = {"component": f"PC{i + 1}"}
        for a in accents:
            row[a] = fa.eta_squared(scores[:, i], np.where(accents_col == a, a, "rest"))
        rows.append(row)
    return pd.DataFrame(rows)


def condition_associations(scores: np.ndarray, table: pd.DataFrame, n_pc: int) -> pd.DataFrame:
    rows = []
    acc = table.accent.to_numpy()
    for col, label in CONDITION_COLUMNS.items():
        y = table[col].to_numpy(dtype=float)
        if np.std(y) == 0:
            continue
        for i in range(n_pc):
            x = scores[:, i]
            rows.append({"component": f"PC{i + 1}", "condition": col, "label": label,
                         "pearson_overall": fa.corr_overall(x, y),
                         "pearson_within_accent": fa.corr_within_accent(x, y, acc)})
    return pd.DataFrame(rows)


def accent_pair_separation(scores: np.ndarray, accents_col: np.ndarray, accents: list[str]) -> pd.DataFrame:
    """Per pair of accents: do their PC1 / PC2 ranges overlap, and how far apart are the centroids
    relative to the spread of the two groups (first 2 and first N_LEADING components)?"""
    rows = []
    for a, b in combinations(accents, 2):
        ia, ib = accents_col == a, accents_col == b
        row = {"accent_a": a, "accent_b": b}
        for i in (0, 1):
            ra, rb = scores[ia, i], scores[ib, i]
            row[f"PC{i + 1}_ranges_overlap"] = bool(ra.min() <= rb.max() and rb.min() <= ra.max())
        for k in (2, N_LEADING):
            pa, pb = scores[ia, :k], scores[ib, :k]
            spread = np.sqrt((((pa - pa.mean(0)) ** 2).sum(1).mean() + ((pb - pb.mean(0)) ** 2).sum(1).mean()) / 2)
            row[f"centroid_distance_over_spread_PC1-{k}"] = float(np.linalg.norm(pa.mean(0) - pb.mean(0)) / spread)
        rows.append(row)
    return pd.DataFrame(rows)


def outlier_table(scores: np.ndarray, table: pd.DataFrame, evr_eig: np.ndarray, k: int = N_LEADING) -> pd.DataFrame:
    """Distance to the own-accent centroid and standardised distance to the overall centroid, PC1..PCk."""
    acc = table.accent.to_numpy()
    s = scores[:, :k]
    own = np.zeros(len(s))
    for a in set(acc):
        m = acc == a
        own[m] = np.linalg.norm(s[m] - s[m].mean(0), axis=1)
    global_d = np.sqrt(((s - s.mean(0)) ** 2 / evr_eig[:k]).sum(1))      # Mahalanobis-style in PC1..PCk
    out = pd.DataFrame({"recording_id": table.recording_id, "accent": acc, f"distance_to_own_accent_centroid_PC1-{k}": own,
                        f"standardised_distance_to_overall_centroid_PC1-{k}": global_d})
    c = f"distance_to_own_accent_centroid_PC1-{k}"
    out["rank_within_accent"] = out.groupby("accent")[c].rank(ascending=False).astype(int)
    return out.sort_values(f"standardised_distance_to_overall_centroid_PC1-{k}", ascending=False).reset_index(drop=True)


def structure_metrics(scores: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Silhouette of the accent labelling and, as a reference, of the original-format labelling."""
    rows = []
    for k in (2, N_LEADING):
        rows.append({"components": f"PC1-{k}", "silhouette_accent": silhouette(scores[:, :k], table.accent),
                     "silhouette_original_format": silhouette(scores[:, :k], table.original_format)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analyses (kept separate from the primary PCA)
# --------------------------------------------------------------------------- #
def variant_matrices(d: dict) -> dict[str, dict]:
    """Name -> {df, features, purpose}. Every variant is a different view of the same recordings."""
    table, feats = d["table"], d["features"]
    keep = [f for f in feats if not f.startswith(EXCLUDED_BASE)]
    return {
        "primary": {"df": table, "features": feats, "label": "Primary: all frames, all features",
                    "purpose": "The reference analysis (Phase 3 recording-level features)."},
        "speech_only_30db": {"df": d["speech"]["speech_30"], "features": feats, "label": "Speech-only (30 dB rule)",
                             "purpose": "Does removing non-speech frames change the structure? Same 78-feature schema."},
        "speech_only_20db": {"df": d["speech"]["speech_20"], "features": feats, "label": "Speech-only (20 dB rule)",
                             "purpose": "Same, with the stricter rule that works better for the noisy Surti recordings."},
        "without_mfcc1_mfcc2": {"df": table, "features": keep, "label": "Without MFCC 1 and MFCC 2",
                                "purpose": "MFCC 1 and 2 were linked to speech-frame share in Phase 4; how much do they drive PCA?"},
    }


def permutation_baseline(X: np.ndarray, features: list[str], n_perm: int = 500, seed: int = 0, n_pc: int = 5) -> pd.DataFrame:
    """Explained variance of PC1..PCn when every feature column is shuffled independently.

    Shuffling destroys the correlations between features, so this shows how much variance leading
    components would capture by chance alone for a matrix of this size (n recordings x p features).
    """
    rng = np.random.default_rng(seed)
    runs = np.array([fit_pca(np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])]), features)["evr"][:n_pc]
                     for _ in range(n_perm)])
    return pd.DataFrame({"component": pc_names(n_pc), "chance_mean": runs.mean(0), "chance_95th_percentile": np.percentile(runs, 95, axis=0),
                         "chance_max": runs.max(0), "n_permutations": n_perm, "seed": seed})


def leave_one_accent_out(table: pd.DataFrame, features: list[str], primary: dict, accents: list[str]) -> pd.DataFrame:
    """Fit PCA without one accent, project that accent into the resulting space and compare to the full PCA."""
    X_all = table[features].to_numpy(float)
    acc = table.accent.to_numpy()
    rows = []
    for a in accents:
        rest = acc != a
        Xr, fr, _ = prepare_matrix(table[rest], features)
        m = fit_pca(Xr, fr)
        proj = transform(m, X_all[~rest][:, [features.index(f) for f in fr]])
        rest_scores = m["scores"]
        cen_rest, cen_out = rest_scores[:, :3].mean(0), proj[:, :3].mean(0)
        rest_spread = np.sqrt(((rest_scores[:, :3] - cen_rest) ** 2).sum(1).mean())
        row = {"left_out_accent": a, "n_fit": int(rest.sum()),
               "PC1_variance_fit_without": m["evr"][0], "PC1_variance_primary": primary["evr"][0]}
        for i in range(3):                                 # |cosine| of loading vectors (sign-free)
            common_f = [f for f in fr if f in primary["features"]]
            v1 = m["components"][i][[fr.index(f) for f in common_f]]
            v2 = primary["components"][i][[primary["features"].index(f) for f in common_f]]
            row[f"PC{i + 1}_loading_cosine_with_primary"] = float(abs(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        row["held_out_centroid_distance_over_fit_spread_PC1-3"] = float(np.linalg.norm(cen_out - cen_rest) / rest_spread)
        row["held_out_PC1_range_outside_fit_range"] = bool(proj[:, 0].min() > rest_scores[:, 0].max() or proj[:, 0].max() < rest_scores[:, 0].min())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_primary(primary: dict, variant: dict, table: pd.DataFrame, n: int = 3) -> dict:
    """|r| between primary and variant scores for PC1..n (same index and best-matching index)."""
    row = {}
    for i in range(n):
        r = [abs(np.corrcoef(primary["scores"][:, i], variant["scores"][:, j])[0, 1]) for j in range(min(n + 2, variant["scores"].shape[1]))]
        row[f"abs_r_primary_PC{i + 1}_vs_same_PC"] = r[i]
        row[f"best_matching_PC_for_primary_PC{i + 1}"] = f"PC{int(np.argmax(r)) + 1}"
        row[f"abs_r_primary_PC{i + 1}_vs_best_PC"] = max(r)
    return row


def run_sensitivity(d: dict, primary: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    variants = variant_matrices(d)
    table, accents = d["table"], d["accents"]
    fits, rows, score_rows = {}, [], []
    for name, v in variants.items():
        df = v["df"].set_index("recording_id").loc[table.recording_id].reset_index()
        X, feats, dropped = prepare_matrix(df, v["features"])
        m = fit_pca(X, feats)
        fits[name] = m
        sm = structure_metrics(m["scores"], table)
        k90 = components_for(m["evr"])
        row = {"variant": name, "description": v["label"], "purpose": v["purpose"], "n_features": len(feats),
               "dropped_zero_variance": ";".join(dropped), "PC1_variance": m["evr"][0], "PC2_variance": m["evr"][1],
               "PC3_variance": m["evr"][2], "PC1-2_cumulative": m["evr"][:2].sum(),
               **{f"components_for_{int(t * 100)}pct": k90[t] for t in VARIANCE_TARGETS}}
        for i in range(3):
            row[f"eta2_accent_PC{i + 1}"] = fa.eta_squared(m["scores"][:, i], table.accent)
            row[f"eta2_format_PC{i + 1}"] = fa.eta_squared(m["scores"][:, i], table.original_format)
        row["silhouette_accent_PC1-2"] = sm.silhouette_accent[0]
        row["silhouette_accent_PC1-5"] = sm.silhouette_accent[1]
        row["silhouette_format_PC1-2"] = sm.silhouette_original_format[0]
        row.update(compare_to_primary(primary, m, table))
        rows.append(row)
        for i, rid in enumerate(table.recording_id):
            score_rows.append({"variant": name, "recording_id": rid, "accent": table.accent[i],
                               **{f"PC{j + 1}": m["scores"][i, j] for j in range(3)}})
    return pd.DataFrame(rows), fits, pd.DataFrame(score_rows)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(d: dict, primary: dict, outputs: dict, sens_fits: dict) -> common.CheckList:
    checks = common.CheckList()
    add = checks.add
    table, feats = d["table"], d["features"]
    n, p = len(table), len(primary["features"])
    k = primary["components"].shape[0]
    sc, comp = primary["scores"], primary["components"]
    add("PCA input = exactly the schema's acoustic columns, same order", primary["features"] == feats, f"{p} features")
    meta_cols = set(d["schema"]["metadata"]) | set(CONDITION_COLUMNS) | {"accent", "recording_id", "speaker_id", "original_format"}
    add("No metadata / condition column used as a PCA input", not (set(primary["features"]) & meta_cols))
    add("PCA input matrix has no NaN/inf", np.isfinite(table[feats].to_numpy(float)).all())
    add("Score matrix shape = (recordings, components)", sc.shape == (n, k), f"{sc.shape}")
    add("Scores, loadings, variance contain no NaN/inf", all(np.isfinite(x).all() for x in (sc, comp, primary["evr"], primary["eigenvalues"])))
    add("Explained-variance ratios sum to 1 over all components", abs(primary["evr"].sum() - 1) < 1e-9, f"sum={primary['evr'].sum():.12f}")
    add("Explained variance is non-increasing", bool((np.diff(primary["evr"]) <= 1e-12).all()))
    add("Loading vectors are orthonormal", np.allclose(comp @ comp.T, np.eye(k), atol=1e-8))
    add("Scores are uncorrelated (diagonal covariance)", np.allclose(np.cov(sc.T) - np.diag(primary["eigenvalues"]), 0, atol=1e-8))
    Z = (table[feats].to_numpy(float) - primary["mean"]) / primary["scale"]
    add("Standardised features have mean 0 and std 1", np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-9))
    add("Scores reconstruct the standardised data (full rank)", np.allclose(sc @ comp, Z, atol=1e-7))
    again = fit_pca(table[feats].to_numpy(float), feats)
    add("Refitting gives identical scores (deterministic)", np.allclose(again["scores"], sc, atol=1e-10))
    m = load_model()
    add("Saved scaler+PCA model reproduces the saved scores", np.allclose(transform(m, table[feats].to_numpy(float)), sc, atol=1e-9))
    s_csv = outputs["scores"]
    add("pca_scores.csv: one row per recording, same IDs and order", list(s_csv.recording_id) == list(table.recording_id))
    add("pca_scores.csv matches the in-memory scores", np.allclose(s_csv[pc_names(k)].to_numpy(), sc, atol=1e-9))
    add("pca_loadings.csv rows follow the schema feature order", list(outputs["loadings"].feature) == feats)
    add("Component loadings CSV matches the model", np.allclose(outputs["loadings"][pc_names(k)].to_numpy(), comp.T, atol=1e-9))
    add("Explained variance CSV has one row per component", len(outputs["variance"]) == k)
    add("Sensitivity fits cover all variants without NaN/inf", all(np.isfinite(f["scores"]).all() for f in sens_fits.values()), f"{len(sens_fits)} variants")
    add("Sensitivity 'without MFCC 1/2' drops exactly the MFCC 1 and 2 columns",
        set(feats) - set(sens_fits["without_mfcc1_mfcc2"]["features"]) == {f for f in feats if f.startswith(EXCLUDED_BASE)})
    add("PCA input files untouched (Phase 3 features CSV hash unchanged during the run)",
        common.hash_files([fa.PHASE3_CSV])[fa.PHASE3_CSV.relative_to(cfg.PROJECT_ROOT).as_posix()] == d["hash_before"])
    return checks


def run_pipeline(make_figures: bool = True) -> dict:
    """Run the whole of Phase 5 and write results/pca/. Returns the objects used by the notebook/validation."""
    import pca_plots

    PCA_DIR.mkdir(parents=True, exist_ok=True)
    d = load_pca_inputs()
    d["hash_before"] = common.hash_files([fa.PHASE3_CSV])[fa.PHASE3_CSV.relative_to(cfg.PROJECT_ROOT).as_posix()]
    table, feats, accents = d["table"], d["features"], d["accents"]

    X, used, dropped = prepare_matrix(table, feats)
    primary = fit_pca(X, used)
    k = primary["components"].shape[0]
    save_model(primary)

    id_cols = ["recording_id", "speaker_id", "accent", "region", "original_format", "original_sample_rate_hz"]
    out = {"scores": scores_table(table, primary, id_cols), "variance": variance_table(primary),
           "loadings": loadings_table(primary)}
    out["scores"].to_csv(PCA_DIR / "pca_scores.csv", index=False)
    out["variance"].to_csv(PCA_DIR / "explained_variance.csv", index=False)
    out["loadings"].to_csv(PCA_DIR / "pca_loadings.csv", index=False)
    out["group_contrib"] = group_contributions(out["loadings"])
    out["group_contrib"].to_csv(PCA_DIR / "loading_group_contributions.csv", index=False)
    out["top"] = top_loadings(out["loadings"])
    out["top"].to_csv(PCA_DIR / "top_loadings.csv", index=False)

    n_pc = min(8, k)
    sc, acc = primary["scores"], table.accent.to_numpy()
    sr_labels = table.original_sample_rate_hz.astype(str).to_numpy()
    out["eta2"] = eta2_by_pc(sc, {"accent": acc, "original_format": table.original_format.to_numpy(),
                                  "original_sample_rate": sr_labels}, n_pc)
    out["one_vs_rest"] = one_vs_rest_eta2(sc, acc, accents, n_pc)
    out["conditions"] = condition_associations(sc, table, n_pc)
    out["pairs"] = accent_pair_separation(sc, acc, accents)
    out["outliers"] = outlier_table(sc, table, primary["eigenvalues"])
    out["structure"] = structure_metrics(sc, table)
    for name, key in (("pc_grouping_eta2.csv", "eta2"), ("pc_accent_one_vs_rest_eta2.csv", "one_vs_rest"),
                      ("pc_condition_associations.csv", "conditions"), ("accent_pair_separation.csv", "pairs"),
                      ("outlier_summary.csv", "outliers"), ("separation_metrics.csv", "structure")):
        out[key].to_csv(PCA_DIR / name, index=False)

    out["chance"] = permutation_baseline(X, used)
    out["chance"].to_csv(PCA_DIR / "permutation_baseline.csv", index=False)
    out["loo"] = leave_one_accent_out(table, used, primary, accents)
    out["loo"].to_csv(PCA_DIR / "leave_one_accent_out.csv", index=False)
    out["sensitivity"], sens_fits, sens_scores = run_sensitivity(d, primary)
    out["sensitivity"].to_csv(PCA_DIR / "sensitivity_summary.csv", index=False)
    sens_scores.to_csv(PCA_DIR / "sensitivity_scores.csv", index=False)

    manifest = {"source_features_csv": fa.PHASE3_CSV.relative_to(cfg.PROJECT_ROOT).as_posix(), "source_sha256": d["hash_before"],
                "schema": fa.PHASE3_SCHEMA.relative_to(cfg.PROJECT_ROOT).as_posix(), "n_recordings": len(table),
                "n_features_used": len(used), "dropped_zero_variance_features": dropped,
                "accents": accents, "n_components": k,
                "components_for_variance": {f"{int(t * 100)}%": v for t, v in components_for(primary["evr"]).items()},
                "scaler": "population std (ddof=0), as sklearn StandardScaler",
                "pca": "SVD of the standardised matrix; explained variance uses n-1; sign: largest |loading| positive"}
    (PCA_DIR / "pca_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    d["checks"].extend(validate(d, primary, out, sens_fits))
    text = ["Phase 5 PCA validation", "=" * 22, *d["checks"].format_lines(),
            "", f"{sum(ok for _, ok, _ in d['checks'])}/{len(d['checks'])} checks passed"]
    (PCA_DIR / "validation_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    d.update(primary=primary, out=out, sens_fits=sens_fits, used=used)
    if make_figures:
        pca_plots.make_all(d)
    return d
