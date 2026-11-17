"""Run Phase 6 (hierarchical dual-branch CNN-MLP model + controlled experiments).

Stages (default: all):
  prepare   Log-Mel spectrograms, additional acoustic features, cross-validation folds, architecture tables
  train     experiments A-D (+ supplementary ablations); resumable: finished runs are skipped unless --force
  finalize  fit the selected architecture once on all recordings for the inference pipeline
  evaluate  comparison tables, figures, hierarchical-inference examples, validation, architecture diagram

Examples:
  python scripts/run_phase6.py                       # everything
  python scripts/run_phase6.py --stage train --runs fusion_model --force
Needs Phases 1-4 outputs. All models are trained from scratch; results go to results/model_development/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
import model_data as md  # noqa: E402
import model_dev as mdl  # noqa: E402
import model_plots as mp  # noqa: E402

OUT = md.OUT
# name -> settings. A-D are the four required experiments; "supplementary_*" runs are ablations reported separately.
RUNS = {
    "svm_baseline": dict(exp="A", label="A: SVM (78 MFCC features)", kind="svm", dir="svm_baseline", features="schema"),
    "cnn_only": dict(exp="B", label="B: CNN only (Log-Mel)", kind="cnn", dir="cnn_only", augment=True, features="schema"),
    "mlp_only": dict(exp="C", label="C: MLP only (78 features)", kind="mlp", dir="mlp_only", augment=True, features="schema"),
    "fusion_model": dict(exp="D", label="D: Dual-branch fusion (proposed)", kind="fusion", dir="fusion_model", augment=True, features="schema"),
    "mlp_only_extra": dict(exp="C-supp", label="C (suppl.): MLP, 78 + 13 additional features", kind="mlp", dir="mlp_only/supplementary_additional_features", augment=True, features="schema+extra"),
    "mlp_only_noaug": dict(exp="C-supp", label="C (suppl.): MLP, no augmentation", kind="mlp", dir="mlp_only/supplementary_no_augmentation", augment=False, features="schema"),
    "fusion_noaug": dict(exp="D-supp", label="D (suppl., post-hoc): fusion, patience 25, no augmentation", kind="fusion", dir="fusion_model/supplementary_patience25_no_augmentation",
                         augment=False, features="schema", train=dict(early_stopping_patience=25, max_epochs=150)),
    # Post-hoc variants (added AFTER seeing that patience-10 early stopping stopped several folds before convergence).
    "fusion_patience25": dict(exp="D-supp", label="D (suppl., post-hoc): fusion, patience 25", kind="fusion", dir="fusion_model/supplementary_patience25",
                              augment=True, features="schema", train=dict(early_stopping_patience=25, max_epochs=150)),
    "cnn_only_patience25": dict(exp="B-supp", label="B (suppl., post-hoc): CNN, patience 25", kind="cnn", dir="cnn_only/supplementary_patience25",
                                augment=True, features="schema", train=dict(early_stopping_patience=25, max_epochs=150)),
}
MAIN = ["svm_baseline", "cnn_only", "mlp_only", "fusion_model"]
SELECTED_RUN = "mlp_only"      # model chosen for the final pipeline (see reports/model_development_report.md, Section 27)


def build_ctx() -> dict:
    meta = md.load_recordings()
    maps = md.class_maps(meta)
    feats, cols = md.load_feature_table()
    extra = pd.read_csv(md.EXTRA_DIR / "additional_acoustic_features.csv")
    X_all = feats.set_index("recording_id").loc[meta.recording_id, cols].join(extra.set_index("recording_id")[md.EXTRA_FEATURES])
    yr, ya = md.encode_labels(meta, maps)
    return {"meta": meta, "maps": maps, "specs": md.load_spectrograms(), "X_all": X_all, "feature_columns": cols,
            "y_region": pd.Series(yr, index=meta.recording_id), "y_accent": pd.Series(ya, index=meta.recording_id)}


# --------------------------------------------------------------------------- #
def stage_prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = md.load_recordings()
    t = time.time()
    specs = md.build_spectrograms(meta)
    print(f"Log-Mel spectrograms: {len(specs)} recordings ({time.time() - t:.0f}s)")
    if not (md.EXTRA_DIR / "additional_acoustic_features.csv").is_file():
        md.build_extra_features(meta)
    print("additional acoustic features written")
    folds = md.make_folds(meta)
    (OUT / "evaluation").mkdir(exist_ok=True)
    md.folds_table(folds).to_csv(OUT / "evaluation" / "cross_validation_folds.csv", index=False)
    maps = md.class_maps(meta)
    (OUT / "hierarchical_inference").mkdir(exist_ok=True)
    (OUT / "hierarchical_inference" / "region_accent_mapping.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
    mp.example_spectrograms(specs, meta, md.SPEC_DIR / "example_log_mel_spectrograms.png", maps)
    for kind in ("fusion", "cnn", "mlp"):
        d = OUT / {"fusion": "fusion_model", "cnn": "cnn_only", "mlp": "mlp_only"}[kind]
        d.mkdir(exist_ok=True)
        m = mdl.build_model(kind, 78, len(maps["regions"]), len(maps["accents"]))
        mdl.layer_table(m).to_csv(d / "architecture_layer_table.csv", index=False)
        lines = []
        m.summary(print_fn=lines.append, line_length=120)
        (d / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        mdl.keras.backend.clear_session()
    print("folds:", len(folds), "| classes:", maps["accents"], "->", maps["regions"])


def run_one(name: str, ctx: dict, folds: list[dict], force: bool) -> None:
    s = RUNS[name]
    run_dir = OUT / s["dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "predictions.csv").is_file() and not force:
        print(f"[skip] {name} (already done; use --force)")
        return
    cols = list(ctx["feature_columns"]) + (md.EXTRA_FEATURES if s["features"] == "schema+extra" else [])
    t0 = time.time()
    if s["kind"] == "svm":
        preds = pd.concat([mdl.svm_fold(f, ctx, cols) for f in folds], ignore_index=True)
        preds.to_csv(run_dir / "predictions.csv", index=False)
        ctrl = []                                                 # label-shuffle sanity control (training labels shuffled)
        for seed in range(50):
            p = pd.concat([mdl.svm_fold(f, ctx, cols, shuffle_labels_seed=seed * 100 + f["fold"]) for f in folds])
            t = p[p.split == "test"]
            ctrl.append({"seed": seed, "test_accent_accuracy": float((t.pred_accent_head == t.true_accent).mean())})
        pd.DataFrame(ctrl).to_csv(run_dir / "label_shuffle_control.csv", index=False)
        infos, hist = [], None
        (run_dir / "svm_parameters.json").write_text(json.dumps({**mdl.SVM_PARAMS, "scaler": "StandardScaler fitted on the training recordings of each fold",
                                                                    "n_features": len(cols), "trained_on": "training recordings only (not the validation recordings)"}, indent=2), encoding="utf-8")
    else:
        all_p, all_h, infos = [], [], []
        for f in folds:
            r = mdl.train_fold(s["kind"], f, ctx, run_dir, augment=s["augment"], feature_columns=cols, train_cfg={**mdl.TRAIN, **s.get("train", {})})
            all_p.append(r["predictions"])
            all_h.append(r["history"])
            infos.append(r["info"])
            te = r["predictions"].query("split == 'test'")
            print(f"  [{name}] fold {f['fold']}: {r['info']['epochs_run']} epochs (best {r['info']['best_epoch']}), "
                  f"test accent acc {(te.pred_accent_head == te.true_accent).mean():.2f}, region acc {(te.pred_region == te.true_region).mean():.2f}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
        preds = pd.concat(all_p, ignore_index=True)
        hist = pd.concat(all_h, ignore_index=True)
        preds.to_csv(run_dir / "predictions.csv", index=False)
        hist.to_csv(run_dir / "history_all_folds.csv", index=False)
        pd.DataFrame(infos).to_csv(run_dir / "run_info.csv", index=False)
        mp.training_curves(hist, f"{s['label']}: training and validation curves", run_dir / "training_curves.png")
    res = mdl.evaluate_run(preds, ctx["maps"])
    mdl.save_evaluation(res, run_dir)
    mdl.overfit_table(preds, hist, infos).to_csv(run_dir / "overfit_summary.csv", index=False)
    print(f"[done] {name}: pooled accent acc {res['accent_head']['accuracy']:.3f}, fold-wise {res['fold_accuracy_mean_sd']['accent_accuracy'][0]:.3f} "
          f"+/- {res['fold_accuracy_mean_sd']['accent_accuracy'][1]:.3f} ({time.time() - t0:.0f}s)", flush=True)


def stage_train(names: list[str], force: bool) -> None:
    ctx = build_ctx()
    folds = md.make_folds(ctx["meta"])
    for n in names:
        run_one(n, ctx, folds, force)


# --------------------------------------------------------------------------- #
def stage_finalize(force: bool = False) -> None:
    """Fit the SELECTED architecture once on all recordings (minus one validation recording per accent) for the
    reusable inference pipeline. Same configuration as the cross-validated run; no tuning; no accuracy is claimed."""
    ctx = build_ctx()
    out = OUT / "final_model"
    if (out / "final_model_info.json").is_file() and not force:
        print("[skip] final model already trained")
        return
    s = RUNS[SELECTED_RUN]
    split = md.final_split(ctx["meta"])
    r = mdl.train_fold(s["kind"], split, ctx, out, augment=s["augment"], feature_columns=ctx["feature_columns"],
                       train_cfg={**mdl.TRAIN, **s.get("train", {})}, bundle_name="bundle")
    r["history"].to_csv(out / "history.csv", index=False)
    r["predictions"].to_csv(out / "training_and_validation_predictions.csv", index=False)
    (out / "final_model_info.json").write_text(json.dumps({
        "architecture": s["label"], "kind": s["kind"], "selected_because": "see report Section 27",
        "n_fit_recordings": len(split["train"]), "validation_recordings": split["val"], "no_test_set": True,
        "evaluation_claim": "none: all recordings were seen in fitting or checkpoint selection; the generalisation estimate is the cross-validated result of the same configuration",
        "training": r["info"], "bundle_dir": "final_model/bundle", "feature_columns": len(ctx["feature_columns"]),
        "train_config": {k: v for k, v in mdl.TRAIN.items()}}, indent=2), encoding="utf-8")
    print("final model:", r["info"])


# --------------------------------------------------------------------------- #
def stage_evaluate() -> int:
    import model_checks as mc
    ctx = build_ctx()
    maps, meta = ctx["maps"], ctx["meta"]
    ev = OUT / "evaluation"
    (ev / "figures").mkdir(parents=True, exist_ok=True)
    rows, over, cms = [], [], {}
    for name, s in RUNS.items():
        d = OUT / s["dir"]
        if not (d / "predictions.csv").is_file():
            continue
        res = json.loads((d / "metrics_summary.json").read_text(encoding="utf-8"))
        ov = pd.read_csv(d / "overfit_summary.csv")
        fa_, fs_ = res["fold_accuracy_mean_sd"]["accent_accuracy"]
        rows.append({"run": name, "experiment": s["exp"], "name": s["label"], "n_test_recordings": res["n"],
                     "pooled_accent_accuracy": res["accent_head"]["accuracy"], "macro_precision": res["accent_head"]["macro_precision"],
                     "macro_recall": res["accent_head"]["macro_recall"], "macro_f1": res["accent_head"]["macro_f1"],
                     "region_accuracy": res["region_head"]["accuracy"], "hierarchical_end_to_end_accuracy": res["hierarchical_end_to_end_accuracy"],
                     "fold_mean": fa_, "fold_sd": fs_,
                     "fold_min": float(pd.read_csv(d / "fold_metrics.csv").accent_accuracy.min()), "fold_max": float(pd.read_csv(d / "fold_metrics.csv").accent_accuracy.max()),
                     "heads_agree_fraction": res["heads_agree_fraction"],
                     **mdl.probability_metrics(pd.read_csv(d / "predictions.csv"), maps)})
        over.append({"name": s["label"], **{c: ov[c].mean() for c in ("train_accent_accuracy", "val_accent_accuracy", "test_accent_accuracy")},
                     "mean_epochs_run": float(ov.epochs_run.mean()) if "epochs_run" in ov else np.nan,
                     "mean_best_epoch": float(ov.best_epoch.mean()) if "best_epoch" in ov else np.nan})
        if name in MAIN:
            cms[s["label"]] = pd.read_csv(d / "confusion_matrix_accent.csv", index_col=0)
    comp = pd.DataFrame(rows)
    comp.to_csv(ev / "experiment_comparison.csv", index=False)
    ovt = pd.DataFrame(over)
    ovt.to_csv(ev / "overfitting_summary.csv", index=False)
    main = comp[comp.run.isin(MAIN)]
    mp.comparison_bars(comp, ev / "figures" / "experiment_comparison.png")
    mp.overfit_bars(ovt, ev / "figures" / "train_val_test_accuracy.png")
    mp.confusion_grid(cms, ev / "figures" / "confusion_matrices_accent_A_to_D.png",
                      "Accent confusion matrices (pooled over the 7 test folds, 35 recordings; within-speaker, exploratory)")
    d_dir = OUT / "fusion_model"
    mp.confusion_grid({"D: region confusion (fusion)": pd.read_csv(d_dir / "confusion_matrix_region.csv", index_col=0),
                       "D: accent confusion (fusion)": pd.read_csv(d_dir / "confusion_matrix_accent.csv", index_col=0)},
                      ev / "figures" / "confusion_matrices_fusion_region_and_accent.png",
                      "Fusion model: region and accent heads (identical class structure because regions map one-to-one to accents)")
    # SVM label-shuffle sanity control summary
    ctrl = pd.read_csv(OUT / "svm_baseline" / "label_shuffle_control.csv")
    ctrl.describe().T.to_csv(ev / "svm_label_shuffle_control_summary.csv")

    # run status (complete / incomplete) and fold-wise accuracy matrix across runs
    status, fold_mat = [], {}
    for name, s in RUNS.items():
        d = OUT / s["dir"]
        n_hist = len(list(d.glob("fold_*/history.csv"))) if s["kind"] != "svm" else (7 if (d / "predictions.csv").is_file() else 0)
        n_fold_dirs = len([q for q in d.glob("fold_*") if q.is_dir()])
        complete = (d / "predictions.csv").is_file()
        status.append({"run": name, "experiment": s["exp"], "label": s["label"], "status": "complete" if complete else "INCOMPLETE / not run",
                       "folds_finished": n_hist, "folds_started": n_fold_dirs, "n_folds_planned": len(md.make_folds(meta))})
        if complete:
            fold_mat[s["label"]] = pd.read_csv(d / "fold_metrics.csv").set_index("fold").accent_accuracy
    pd.DataFrame(status).to_csv(ev / "run_status.csv", index=False)
    fm = pd.DataFrame(fold_mat).T
    fm.to_csv(ev / "fold_accuracy_matrix.csv")
    mp.fold_heatmap(fm, ev / "figures" / "fold_accuracy_heatmap.png")
    bn = OUT / "evaluation" / "cnn_batchnorm_statistics_diagnostic.csv"
    if bn.is_file():
        mp.batchnorm_diagnostic(pd.read_csv(bn), pd.concat([pd.read_csv(OUT / RUNS[n]["dir"] / "fold_metrics.csv").assign(run=RUNS[n]["dir"]) for n in ("cnn_only", "cnn_only_patience25", "fusion_model") if (OUT / RUNS[n]["dir"] / "fold_metrics.csv").is_file()]),
                                ev / "figures" / "cnn_batchnorm_diagnostic.png")

    # architecture diagram with the real parameter counts
    m = mdl.build_model("fusion", 78, len(maps["regions"]), len(maps["accents"]))
    n_tr = int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))
    n_nt = int(sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))
    (OUT / "fusion_model" / "parameter_counts.json").write_text(json.dumps({"trainable": n_tr, "non_trainable_batchnorm_statistics": n_nt, "total": n_tr + n_nt}, indent=2), encoding="utf-8")
    for kind, dname in (("cnn", "cnn_only"), ("mlp", "mlp_only")):
        mk = mdl.build_model(kind, 78, len(maps["regions"]), len(maps["accents"]))
        (OUT / dname / "parameter_counts.json").write_text(json.dumps({"trainable": int(sum(int(np.prod(w.shape)) for w in mk.trainable_weights)),
                                                                        "total": int(mk.count_params())}, indent=2), encoding="utf-8")
    mdl.keras.backend.clear_session()
    mp.architecture_diagram(n_trainable=n_tr, n_bn_stats=n_nt)

    mc.inference_examples(ctx, OUT / "hierarchical_inference", RUNS[SELECTED_RUN]["dir"])
    checks = mc.phase6_checks(deep=True)
    text = ["Phase 6 validation", "=" * 18, *checks.format_lines(), "", f"{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed"]
    (ev / "validation_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(checks.format_lines()))
    print(f"{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")
    print(main[["name", "pooled_accent_accuracy", "macro_f1", "fold_mean", "fold_sd"]].round(3).to_string(index=False))
    return 0 if checks.passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prepare", "train", "finalize", "evaluate", "all"], default="all")
    ap.add_argument("--runs", nargs="*", default=None, help=f"subset of {list(RUNS)} (default: all)")
    ap.add_argument("--force", action="store_true", help="retrain even if results exist")
    a = ap.parse_args()
    names = a.runs or list(RUNS)
    rc = 0
    if a.stage in ("prepare", "all"):
        stage_prepare()
    if a.stage in ("train", "all"):
        stage_train(names, a.force)
    if a.stage in ("finalize", "all"):
        stage_finalize(a.force)
    if a.stage in ("evaluate", "all"):
        rc = stage_evaluate()
    sys.exit(rc)
