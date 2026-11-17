"""Phase 6 validation checks and the hierarchical-inference examples.

phase6_checks(deep=False) is also used by scripts/validate_project.py (read-only, no training).
deep=True additionally reloads every saved checkpoint and re-derives the per-fold normalisation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common
import config as cfg
import feature_analysis as fa
import mfcc_features as mf
import model_data as md
import model_dev as mdl
import model_plots as mp

OUT = md.OUT
NN_DIRS = {"cnn_only": "cnn", "mlp_only": "mlp", "fusion_model": "fusion"}
ROOT = cfg.PROJECT_ROOT


def expected_trainable_params(n_features: int, n_regions: int, n_accents: int) -> int:
    """Independent hand count of the trainable parameters of the fusion network."""
    conv = 3 * 3 * (1 * 32 + 32 * 64 + 64 * 128)
    bn_conv = 2 * (32 + 64 + 128)
    cnn_dense = 128 * 128 + 128
    mlp = n_features * 64 + 64 + 2 * 64 + 64 * 32 + 32
    shared = 160 * 64 + 64
    heads = (64 * n_regions + n_regions) + (64 * n_accents + n_accents)
    return conv + bn_conv + cnn_dense + mlp + shared + heads


def run_dirs() -> list[Path]:
    """Every directory that holds a finished neural-network run."""
    return sorted(p.parent for p in OUT.rglob("run_info.csv"))


class _LoggingDict(dict):
    """dict that records which recordings were read (to prove augmentation only touches training ids)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.touched: set[str] = set()

    def __getitem__(self, key):
        self.touched.add(key)
        return super().__getitem__(key)


def phase6_checks(deep: bool = False) -> common.CheckList:
    c = common.CheckList()
    add = c.add
    meta = md.load_recordings()
    maps = md.class_maps(meta)
    feats, cols = md.load_feature_table()
    specs = md.load_spectrograms()
    folds_tab = pd.read_csv(OUT / "evaluation" / "cross_validation_folds.csv")
    folds = md.make_folds(meta)

    # 1. recordings
    ids = list(meta.recording_id)
    add("All recordings represented: Phase 1 metadata == Phase 3 table == spectrogram store",
        set(ids) == set(feats.recording_id) == set(specs), f"{len(ids)} recordings")
    add("Every recording has a label for both heads", meta.accent.notna().all() and meta.region.notna().all())

    # 2. numerical inputs
    X78 = feats.set_index("recording_id").loc[ids, cols]
    extra = pd.read_csv(md.EXTRA_DIR / "additional_acoustic_features.csv").set_index("recording_id").loc[ids]
    add("Model input columns == Phase 3 schema (78, unchanged, no metadata)", cols == fa.load_schema()["features"] and len(cols) == 78
        and not set(cols) & set(fa.load_schema()["metadata"]), f"{len(cols)} columns")
    add("No NaN/inf in the 78 features", np.isfinite(X78.to_numpy(float)).all())
    add("Additional features stored separately, valid and finite", list(extra.columns) == md.EXTRA_FEATURES and np.isfinite(extra.to_numpy(float)).all()
        and not set(md.EXTRA_FEATURES) & set(cols), f"{len(md.EXTRA_FEATURES)} columns")

    # 3. spectrograms
    shapes = {k: v.shape for k, v in specs.items()}
    add("Every spectrogram has 64 Mel bands and finite values", all(s[0] == 64 for s in shapes.values()) and all(np.isfinite(v).all() for v in specs.values()))
    same_len = all(specs[r].shape[1] == mf.load_frame_npz(r)["mfcc"].shape[1] for r in ids)
    add("Spectrogram frame counts equal the Phase 3 MFCC frame counts (same 25 ms / 10 ms framing)", same_len)
    cfg_json = json.loads((md.SPEC_DIR / "spectrogram_config.json").read_text(encoding="utf-8"))
    add("Spectrogram config on disk equals the code (64 mels, 400 window, 160 hop, Hann, T=800)",
        cfg_json["n_mels"] == 64 and cfg_json["win_length"] == 400 and cfg_json["hop_length"] == 160 and cfg_json["window"] == "hann" and cfg_json["model_input_frames_T"] == md.T_FRAMES)
    sc0 = md.fit_spec_scaler([specs[r] for r in ids[:5]])
    batch = np.stack([md.fit_crop(md.normalise_spec(specs[r], sc0), md.T_FRAMES) for r in ids])
    add(f"Model-ready spectrogram batch has the consistent shape (batch, 64, {md.T_FRAMES})", batch.shape == (len(ids), 64, md.T_FRAMES), str(batch.shape))
    short = [r for r in ids if specs[r].shape[1] < md.T_FRAMES]
    add("Padding path produces exactly T frames for a too-short input", md.fit_crop(specs[ids[0]][:, :100], md.T_FRAMES).shape[1] == md.T_FRAMES, f"{len(short)} recordings need padding")

    # 4. no pretrained components
    banned = re.compile(r"keras\.applications|load_weights|from_pretrained|tensorflow_hub|tfhub|torchaudio|transformers|imagenet|\.h5['\"]|weights\s*=\s*['\"]")
    hits = []
    for f in ("src/model_dev.py", "src/model_data.py", "src/model_plots.py", "scripts/run_phase6.py"):
        for i, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line) and "banned" not in line and "pattern" not in line:
                hits.append(f"{f}:{i}")
    add("No pretrained model, weights, hub or embedding references in the Phase 6 code", not hits, f"hits: {hits}")
    mdl.set_seeds(7)
    m1 = mdl.build_model("fusion", 78, len(maps["regions"]), len(maps["accents"]))
    w1 = [w.numpy() for w in m1.trainable_weights]
    mdl.set_seeds(7)
    m2 = mdl.build_model("fusion", 78, len(maps["regions"]), len(maps["accents"]))
    w2 = [w.numpy() for w in m2.trainable_weights]
    mdl.set_seeds(8)
    m3 = mdl.build_model("fusion", 78, len(maps["regions"]), len(maps["accents"]))
    w3 = [w.numpy() for w in m3.trainable_weights]
    kernels = [w for w in w1 if w.ndim > 1]
    add("Weights are freshly random: same seed -> identical init, different seed -> different init, kernels non-constant",
        all(np.array_equal(a, b) for a, b in zip(w1, w2)) and not all(np.array_equal(a, b) for a, b in zip(w1, w3)) and all(k.std() > 0 for k in kernels))

    # 5. weights updated during training
    infos = {d: pd.read_csv(d / "run_info.csv") for d in run_dirs()}
    add("All trainable weight tensors changed during training (every fold of every neural run)",
        bool(infos) and all(i.all_trainable_tensors_updated.all() for i in infos.values()), f"{len(infos)} runs, {sum(len(i) for i in infos.values())} fold models")
    add("Best checkpoint on disk reproduces the restored best weights (every fold)", bool(infos) and all(i.checkpoint_matches_restored_weights.all() for i in infos.values()))

    # 6. layer shapes and parameter count
    lt = mdl.layer_table(m1).set_index("layer")
    shp = lambda n: lt.loc[n, "output_shape"]  # noqa: E731
    expect = {"spectrogram": "(None, 64, 800, 1)", "conv1_32": "(None, 64, 800, 32)", "conv1_pool": "(None, 32, 400, 32)",
              "conv2_64": "(None, 32, 400, 64)", "conv2_pool": "(None, 16, 200, 64)", "conv3_128": "(None, 16, 200, 128)",
              "conv3_relu": "(None, 16, 200, 128)", "cnn_gap": "(None, 128)", "cnn_dropout": "(None, 128)", "features": "(None, 78)",
              "mlp_dense_64": "(None, 64)", "mlp_dropout_1": "(None, 64)", "mlp_dense_32": "(None, 32)", "mlp_dropout_2": "(None, 32)",
              "fusion_concat": "(None, 160)", "shared_dense_64": "(None, 64)", "shared_dropout": "(None, 64)",
              "region_output": f"(None, {len(maps['regions'])})", "accent_output": f"(None, {len(maps['accents'])})"}
    bad = {k: (shp(k), v) for k, v in expect.items() if shp(k) != v}
    add("Every layer output shape matches the specified architecture", not bad, f"{len(expect)} layers checked; mismatches: {bad}")
    n_tr = int(sum(int(np.prod(w.shape)) for w in m1.trainable_weights))
    add("Trainable parameter count equals the independent hand calculation", n_tr == expected_trainable_params(78, len(maps["regions"]), len(maps["accents"])), f"{n_tr:,}")
    conv_bias = [l for l in m1.layers if type(l).__name__ == "Conv2D" and l.use_bias]
    add("All three Conv2D layers use use_bias=False, padding='same', 3x3 kernels",
        not conv_bias and all(l.padding == "same" and l.kernel_size == (3, 3) for l in m1.layers if type(l).__name__ == "Conv2D"))
    drop = {l.name: l.rate for l in m1.layers if type(l).__name__ == "Dropout"}
    add("Dropout rates are 0.40 / 0.30 / 0.20 / 0.40", drop == {"cnn_dropout": 0.4, "mlp_dropout_1": 0.3, "mlp_dropout_2": 0.2, "shared_dropout": 0.4}, str(drop))

    # 7-8. head outputs, hierarchical decoding
    prob_ok, map_ok, files = True, True, []
    for d in run_dirs():
        p = pd.read_csv(d / "predictions.csv")
        pr = p[[f"p_region::{r}" for r in maps["regions"]]].to_numpy()
        pa = p[[f"p_accent::{a}" for a in maps["accents"]]].to_numpy()
        prob_ok &= bool(np.isfinite(pr).all() and np.isfinite(pa).all() and (pr >= 0).all() and (pa >= 0).all()
                        and np.allclose(pr.sum(1), 1, atol=1e-4) and np.allclose(pa.sum(1), 1, atol=1e-4))
        map_ok &= bool((p.pred_accent_hier.map(maps["accent_to_region"]) == p.pred_region).all())
        files.append(d.name)
    add("Both heads output valid probabilities (finite, in [0,1], rows sum to 1) in every saved prediction", prob_ok, f"{len(files)} runs")
    add("Hierarchical decoding follows the region -> accent mapping in every prediction", map_ok)
    multi = {"accents": ["a1", "a2", "b1"], "regions": ["A", "B"], "accent_to_region": {"a1": "A", "a2": "A", "b1": "B"}}
    dec = mdl.hierarchical_decode(np.array([[0.9, 0.1]]), np.array([[0.2, 0.3, 0.5]]), multi)[0]
    add("Decoder supports several accents in one region (synthetic mapping): picks the best accent inside the predicted region",
        dec["region"] == "A" and dec["accent"] == "a2" and abs(dec["accent_confidence_within_region"] - 0.6) < 1e-9, str({k: dec[k] for k in ("region", "accent")}))

    # 9. augmentation / split leakage
    allids = set(ids)
    tests = [r for f in folds for r in f["test"]]
    add("Folds: every recording is tested exactly once", sorted(tests) == sorted(ids) and len(set(tests)) == len(ids))
    add("Folds: train, validation and test recordings are disjoint in every fold",
        all(not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"]) or set(f["val"]) & set(f["test"])) and set(f["train"]) | set(f["val"]) | set(f["test"]) == allids for f in folds))
    add("Folds on disk equal the folds regenerated from the seed", all(set(folds_tab[(folds_tab.fold == f["fold"]) & (folds_tab.role == r)].recording_id) == set(f[r]) for f in folds for r in ("train", "val", "test")))
    add("Every fold's test set holds one recording per accent (stratified)", all(sorted(meta.set_index("recording_id").loc[f["test"], "accent"]) == sorted(maps["accents"]) for f in folds))
    ctx_specs = _LoggingDict(specs)
    ctx = {"specs": ctx_specs, "y_region": pd.Series(0, index=ids), "y_accent": pd.Series(0, index=ids), "X": pd.DataFrame(0.0, index=ids, columns=cols)}
    f0 = folds[0]
    sc = {"spec": md.fit_spec_scaler([specs[r] for r in f0["train"]]), "feat": md.fit_feature_scaler(X78.loc[f0["train"]].to_numpy(float))}
    seq = mdl.TrainSequence(f0["train"], ctx, sc, "fusion", True, 8, 4, seed=1)
    for i in range(len(seq)):
        seq[i]
    add("Augmentation reads only the TRAINING recordings of its fold (one full epoch inspected)", ctx_specs.touched == set(f0["train"]) and not ctx_specs.touched & (set(f0["val"]) | set(f0["test"])),
        f"{len(ctx_specs.touched)} recordings touched, all in the training set")
    lab_ok = True
    for d in run_dirs():
        for f in folds:
            lj = json.loads((d / f"fold_{f['fold']}" / "labels.json").read_text(encoding="utf-8"))
            lab_ok &= lj["train_ids"] == f["train"] and lj["val_ids"] == f["val"] and lj["test_ids"] == f["test"]
    add("Every trained fold used exactly the planned train/val/test recordings", lab_ok)

    # 10. normalisation fitted on training folds only
    norm_ok, differs = True, True
    for d in [OUT / "fusion_model"]:
        for f in (folds if deep else folds[:1]):
            z = np.load(d / f"fold_{f['fold']}" / "preprocessing.npz")
            s_tr = md.fit_spec_scaler([specs[r] for r in f["train"]])
            x_tr = md.fit_feature_scaler(X78.loc[f["train"]].to_numpy(float))
            norm_ok &= np.allclose(z["spec_mean"], s_tr["mean"]) and np.allclose(z["feat_mean"], x_tr["mean"]) and np.allclose(z["feat_std"], x_tr["std"])
            x_all = md.fit_feature_scaler(X78.to_numpy(float))
            differs &= not np.allclose(z["feat_mean"], x_all["mean"])
    add("Saved normalisation equals one re-fitted on the training recordings only (and differs from an all-data fit)", bool(norm_ok and differs), "deep: every fold" if deep else "fold 1")

    # 11. checkpoints reload and reproduce the stored predictions
    ck_ok, n_ck = True, 0
    for d in run_dirs():
        kind = "fusion" if "fusion" in str(d) else ("cnn" if "cnn" in str(d) else "mlp")
        p = pd.read_csv(d / "predictions.csv")
        for f in (folds if deep else folds[:1]):
            if not deep and d != OUT / "fusion_model":
                continue
            pred = mdl.AccentPredictor(d / f"fold_{f['fold']}")
            lj = pred.labels
            te = p[(p.fold == f["fold"]) & (p.split == "test")]
            cols_run = lj["feature_columns"]
            ctx_i = {"specs": specs, "y_region": pd.Series(0, index=ids), "y_accent": pd.Series(0, index=ids), "X": feats.set_index("recording_id")[cols_run].reindex(ids) if cols_run == cols else None}
            if ctx_i["X"] is None:                                         # run with additional features
                ctx_i["X"] = X78.join(extra)[cols_run]
            sc_i = {"spec": {"mean": pred.sc["spec"]["mean"], "std": pred.sc["spec"]["std"]}, "feat": pred.sc["feat"]}
            x, _ = mdl.batch_from_ids(list(te.recording_id), ctx_i, sc_i, kind, augment=False)
            out = pred.model.predict(x, verbose=0)["accent_output"]
            ck_ok &= bool(np.allclose(out, te[[f"p_accent::{a}" for a in maps["accents"]]].to_numpy(), atol=1e-4))
            n_ck += 1
    add("Saved checkpoints reload (.keras) and reproduce the stored test probabilities", ck_ok, f"{n_ck} fold models reloaded")

    # 12. controls / sanity
    ctrl = pd.read_csv(OUT / "svm_baseline" / "label_shuffle_control.csv")
    add("SVM label-shuffle control stays near chance (pipeline does not leak labels)", ctrl.test_accent_accuracy.mean() < 0.35,
        f"mean test accuracy with shuffled training labels {ctrl.test_accent_accuracy.mean():.3f} (chance 0.200, 50 shuffles)")

    # 13. figures
    figs = [mp.ARCH_PNG] + [OUT / d / "training_curves.png" for d in NN_DIRS] + [OUT / "evaluation" / "figures" / n for n in
            ("experiment_comparison.png", "train_val_test_accuracy.png", "confusion_matrices_accent_A_to_D.png", "confusion_matrices_fusion_region_and_accent.png")] \
           + [md.SPEC_DIR / "example_log_mel_spectrograms.png"]
    add("Architecture diagram, training curves, confusion matrices and comparison plots exist", all(f.is_file() and f.stat().st_size > 5000 for f in figs),
        f"missing: {[f.name for f in figs if not f.is_file()]}")
    ex = OUT / "hierarchical_inference" / "prediction_examples.csv"
    if ex.is_file():
        e = pd.read_csv(ex)
        add("Reusable predictor examples cover every recording and agree with the batch predictions (<= 1e-4)",
            set(ids) <= set(e.recording_id) and json.loads((OUT / "hierarchical_inference" / "inference_consistency.json").read_text())["max_abs_probability_difference"] < 1e-4)
    else:
        add("Reusable predictor examples written", False)
    # 14. final model, status bookkeeping
    fi = OUT / "final_model" / "final_model_info.json"
    add("Final (selected) model trained, documented as having no test set and no accuracy claim", fi.is_file() and json.loads(fi.read_text(encoding="utf-8")).get("no_test_set") is True)
    if (OUT / "final_model" / "bundle" / "model.keras").is_file():
        fp = mdl.AccentPredictor(OUT / "final_model" / "bundle")
        res = fp.predict_file(ROOT / meta.iloc[0]["processed_file_path"])
        pr, pa = np.array(list(res["region_probabilities"].values())), np.array(list(res["accent_probabilities"].values()))
        add("Final model reloads and the inference pipeline returns valid, mapping-consistent probabilities for a new file",
            abs(pr.sum() - 1) < 1e-4 and abs(pa.sum() - 1) < 1e-4 and maps["accent_to_region"][res["accent"]] == res["region"], f"{res['region']} / {res['accent']}")
    else:
        add("Final model bundle exists", False)
    add("scripts/predict_accent.py (command-line inference) exists", (ROOT / "scripts" / "predict_accent.py").is_file())
    inc = []
    for d in run_dirs():
        n_hist = len(list(d.glob("fold_*/history.csv")))
        if (d / "predictions.csv").is_file() and n_hist != len(folds):
            inc.append(d.name)
    add("Every run that has predictions.csv finished all folds (no incomplete run is reported as complete)", not inc, f"{inc}")
    rs = OUT / "evaluation" / "run_status.csv"
    add("Run status table written (complete vs incomplete runs)", rs.is_file())
    return c


def inference_examples(ctx: dict, out_dir: Path, selected_dir: str = "mlp_only") -> pd.DataFrame:
    """Run the reusable hierarchical predictor with each recording's own HELD-OUT fold model of the selected architecture."""
    meta, maps = ctx["meta"], ctx["maps"]
    folds = md.make_folds(meta)
    fold_of = {r: f["fold"] for f in folds for r in f["test"]}
    preds = pd.read_csv(OUT / selected_dir / "predictions.csv")
    preds = preds[preds.split == "test"].set_index("recording_id")
    predictors = {f["fold"]: mdl.AccentPredictor(OUT / selected_dir / f"fold_{f['fold']}") for f in folds}
    rows, diffs = [], []
    originals = {a: meta[meta.accent == a].recording_id.iloc[0] for a in maps["accents"][:3]}   # also test the original M4A/OGG files
    for _, r in meta.iterrows():
        rid, k = r["recording_id"], fold_of[r["recording_id"]]
        for source, path in (("processed_wav", ROOT / r["processed_file_path"]), ("original_file", ROOT / r["original_file_path"])):
            if source == "original_file" and rid not in originals.values():
                continue
            res = predictors[k].predict_file(path)
            rows.append({"recording_id": rid, "source": source, "fold_model": k, "true_region": r["region"], "true_accent": r["accent"],
                         "predicted_region": res["region"], "predicted_accent": res["accent"],
                         "region_confidence": res["region_confidence"], "accent_confidence_within_region": res["accent_confidence_within_region"],
                         "accent_head_probability": res["accent_head_probability"], "joint_confidence": res["joint_confidence"],
                         "region_correct": res["region"] == r["region"], "accent_correct": res["accent"] == r["accent"]})
            if source == "processed_wav":
                diffs.append(float(np.abs(np.array([res["accent_probabilities"][a] for a in maps["accents"]]) -
                                          preds.loc[rid, [f"p_accent::{a}" for a in maps["accents"]]].to_numpy(float)).max()))
    df = pd.DataFrame(rows)
    df.insert(1, "model", f"{selected_dir}: fold model that did NOT see this recording")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_b = OUT / "final_model" / "bundle"
    if final_b.is_dir():                                       # the deployable final model has seen every recording: illustrative only
        fp = mdl.AccentPredictor(final_b)
        demo = []
        for _, r in meta.iterrows():
            res = fp.predict_file(ROOT / r["processed_file_path"])
            demo.append({"recording_id": r["recording_id"], "true_region": r["region"], "true_accent": r["accent"], "predicted_region": res["region"],
                         "predicted_accent": res["accent"], "region_confidence": res["region_confidence"], "joint_confidence": res["joint_confidence"],
                         "note": "final model was fitted on this recording: illustrative only, NOT an evaluation"})
        pd.DataFrame(demo).to_csv(out_dir / "final_model_demo_predictions.csv", index=False)
    df.to_csv(out_dir / "prediction_examples.csv", index=False)
    (out_dir / "inference_consistency.json").write_text(json.dumps({"max_abs_probability_difference": max(diffs),
        "meaning": "reusable predictor (audio file -> features -> saved normalisation -> model) vs the batch evaluation predictions"}, indent=2), encoding="utf-8")
    return df
