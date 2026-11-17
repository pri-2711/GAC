"""Phase 6 - dual-branch CNN-MLP feature-fusion network with hierarchical multi-task heads (Keras).

Everything is trained from scratch: no pretrained model, weights, embeddings or encoders are used;
every layer starts from Keras' random initialisers (Glorot-uniform kernels, zero biases).

Model kinds (all share the same trunk and heads so that they are comparable):
  "fusion": Branch A (CNN, 128) + Branch B (MLP, 32) -> concat (160) -> Dense(64)+Dropout -> two heads  [proposed]
  "cnn"   : Branch A only (128) -> Dense(64)+Dropout -> two heads                                        [experiment B]
  "mlp"   : Branch B only (32)  -> Dense(64)+Dropout -> two heads                                        [experiment C]
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from keras import layers  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import audio_analysis as aa  # noqa: E402
import audio_utils as au  # noqa: E402
import config as cfg  # noqa: E402
import mfcc_features as mf  # noqa: E402
import model_data as md  # noqa: E402

TRAIN = dict(optimizer="Adam", learning_rate=1e-3, batch_size=8, max_epochs=100, repeats_per_epoch=4,
             early_stopping_patience=10, reduce_lr_factor=0.5, reduce_lr_patience=4,
             lambda_region=1.0, lambda_accent=1.0, seed=42, dropout_cnn=0.40, dropout_mlp1=0.30,
             dropout_mlp2=0.20, dropout_shared=0.40)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)           # python, numpy and TensorFlow


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def cnn_branch(x):
    """Branch A. Input (64, T, 1) -> 128-d vector."""
    for i, (f, pool) in enumerate(((32, True), (64, True), (128, False)), 1):
        x = layers.Conv2D(f, (3, 3), padding="same", use_bias=False, name=f"conv{i}_{f}")(x)
        x = layers.BatchNormalization(name=f"conv{i}_bn")(x)
        x = layers.ReLU(name=f"conv{i}_relu")(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f"conv{i}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="cnn_gap")(x)
    x = layers.Dense(128, activation="relu", name="cnn_dense_128")(x)
    return layers.Dropout(TRAIN["dropout_cnn"], name="cnn_dropout")(x)


def mlp_branch(x):
    """Branch B. Input (n_features,) -> 32-d vector."""
    x = layers.Dense(64, activation="relu", name="mlp_dense_64")(x)
    x = layers.BatchNormalization(name="mlp_bn")(x)
    x = layers.Dropout(TRAIN["dropout_mlp1"], name="mlp_dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="mlp_dense_32")(x)
    return layers.Dropout(TRAIN["dropout_mlp2"], name="mlp_dropout_2")(x)


def build_model(kind: str, n_features: int, n_regions: int, n_accents: int, T: int = md.T_FRAMES) -> keras.Model:
    if kind not in ("fusion", "cnn", "mlp"):
        raise ValueError(kind)
    inputs, parts = {}, []
    if kind in ("fusion", "cnn"):
        inputs["spectrogram"] = keras.Input((md.SPEC["n_mels"], T, 1), name="spectrogram")
        parts.append(cnn_branch(inputs["spectrogram"]))
    if kind in ("fusion", "mlp"):
        inputs["features"] = keras.Input((n_features,), name="features")
        parts.append(mlp_branch(inputs["features"]))
    x = layers.Concatenate(axis=-1, name="fusion_concat")(parts) if len(parts) == 2 else parts[0]
    x = layers.Dense(64, activation="relu", name="shared_dense_64")(x)
    x = layers.Dropout(TRAIN["dropout_shared"], name="shared_dropout")(x)
    region = layers.Dense(n_regions, activation="softmax", name="region_output")(x)
    accent = layers.Dense(n_accents, activation="softmax", name="accent_output")(x)
    return keras.Model(inputs, {"region_output": region, "accent_output": accent}, name=f"{kind}_hierarchical_net")


def compile_model(model: keras.Model, lr: float = TRAIN["learning_rate"]) -> None:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss={"region_output": "sparse_categorical_crossentropy", "accent_output": "sparse_categorical_crossentropy"},
                  loss_weights={"region_output": TRAIN["lambda_region"], "accent_output": TRAIN["lambda_accent"]},
                  metrics={"region_output": ["accuracy"], "accent_output": ["accuracy"]})


def layer_table(model: keras.Model) -> pd.DataFrame:
    """Layer name, type, output shape and parameter counts (read from the built model)."""
    rows = []
    for l in model.layers:
        shp = l.output.shape if hasattr(l.output, "shape") else l.output[0].shape
        rows.append({"layer": l.name, "type": type(l).__name__, "output_shape": str(tuple(shp)),
                     "trainable_params": int(sum(int(np.prod(w.shape)) for w in l.trainable_weights)),
                     "non_trainable_params": int(sum(int(np.prod(w.shape)) for w in l.non_trainable_weights))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data feeding
# --------------------------------------------------------------------------- #
class TrainSequence(keras.utils.PyDataset):
    """Draws `repeats` random (augmented) views of every TRAINING recording per epoch.

    The views are not additional independent samples: they all derive from the same recordings.
    With augment=False every view is the plain centre crop (so the number of gradient steps per
    epoch is unchanged in the 'no augmentation' ablation).
    """

    def __init__(self, ids, ctx, scalers, kind, augment, batch_size, repeats, seed, **kw):
        super().__init__(**kw)
        self.ids, self.ctx, self.sc, self.kind, self.augment = list(ids), ctx, scalers, kind, augment
        self.bs, self.repeats = batch_size, repeats
        self.rng = np.random.default_rng(seed)
        self.order = self._new_order()

    def _new_order(self):
        return self.rng.permutation(np.repeat(np.arange(len(self.ids)), self.repeats))

    def __len__(self):
        return int(np.ceil(len(self.order) / self.bs))

    def __getitem__(self, i):
        sel = self.order[i * self.bs:(i + 1) * self.bs]
        return batch_from_ids([self.ids[j] for j in sel], self.ctx, self.sc, self.kind, self.augment, self.rng)

    def on_epoch_end(self):
        self.order = self._new_order()


def batch_from_ids(ids, ctx, sc, kind, augment, rng=None):
    x, y = {}, {"region_output": ctx["y_region"][ids].to_numpy(), "accent_output": ctx["y_accent"][ids].to_numpy()}
    if kind in ("fusion", "cnn"):
        if augment:
            sp = [md.augment_spec(ctx["specs"][r], sc["spec"], md.T_FRAMES, rng) for r in ids]
        else:
            sp = [md.fit_crop(md.normalise_spec(ctx["specs"][r], sc["spec"]), md.T_FRAMES, "center") for r in ids]
        x["spectrogram"] = np.stack(sp)[..., None].astype(np.float32)
    if kind in ("fusion", "mlp"):
        f = md.apply_feature_scaler(ctx["X"].loc[ids].to_numpy(float), sc["feat"])
        x["features"] = md.augment_features(f, rng) if augment else f
    return x, y


# --------------------------------------------------------------------------- #
# Hierarchical decoding and inference
# --------------------------------------------------------------------------- #
def hierarchical_decode(p_region: np.ndarray, p_accent: np.ndarray, maps: dict) -> list[dict]:
    """Region first, then the accent *within the predicted region* (probabilities renormalised).

    Works for any region -> accents mapping, including regions that contain several accents.
    """
    out = []
    accents, regions = maps["accents"], maps["regions"]
    for pr, pa in zip(np.atleast_2d(p_region), np.atleast_2d(p_accent)):
        r = int(np.argmax(pr))
        allowed = [i for i, a in enumerate(accents) if maps["accent_to_region"][a] == regions[r]]
        cond = pa[allowed] / max(pa[allowed].sum(), 1e-12)
        a_idx = allowed[int(np.argmax(cond))]
        out.append({"region": regions[r], "accent": accents[a_idx], "region_confidence": float(pr[r]),
                    "accent_confidence_within_region": float(cond.max()),
                    "accent_head_probability": float(pa[a_idx]),
                    "joint_confidence": float(pr[r] * cond.max()),
                    "accent_head_argmax": accents[int(np.argmax(pa))]})
    return out


class AccentPredictor:
    """Reusable inference: audio file -> Log-Mel + 78 schema features -> saved normalisation -> network ->
    hierarchical region/accent decision. Loads a bundle directory written by train_fold()."""

    def __init__(self, bundle_dir: Path):
        self.dir = Path(bundle_dir)
        self.labels = json.loads((self.dir / "labels.json").read_text(encoding="utf-8"))
        z = np.load(self.dir / "preprocessing.npz")
        self.sc = {"spec": {"mean": z["spec_mean"], "std": z["spec_std"]}, "feat": {"mean": z["feat_mean"], "std": z["feat_std"]}}
        self.model = keras.models.load_model(self.dir / "model.keras", compile=False)
        self.kind = self.labels["kind"]

    def load_audio(self, path):
        """Standardise any audio file exactly like Phase 1 and return (signal, 16000).

        Files that are already 16 kHz mono WAV are read as they are; everything else (M4A, OGG, MP3, or a WAV with another
        sample rate / channel count) is converted with the Phase 1 FFmpeg step (16 kHz, mono, PCM-16, no other processing).
        """
        path = Path(path)
        if path.suffix.lower() == ".wav":
            try:
                info = sf.info(str(path))
                if info.samplerate == cfg.TARGET_SAMPLE_RATE and info.channels == cfg.TARGET_CHANNELS:
                    return aa.load_audio(path)
            except Exception:                                    # unreadable header: let FFmpeg decide below
                pass
        tmp = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
        ok, msg = au.convert_to_wav(path, tmp)
        if not ok:
            raise RuntimeError(f"conversion failed: {msg}")
        return aa.load_audio(tmp)

    _audio = load_audio                                          # backwards-compatible name used by predict_file / checks

    def inputs_from_audio(self, y, sr) -> dict:
        x = {}
        if self.kind in ("fusion", "cnn"):
            spec = md.fit_crop(md.normalise_spec(md.compute_logmel(y, sr), self.sc["spec"]), self.labels["T"], "center")
            x["spectrogram"] = spec[None, ..., None].astype(np.float32)
        if self.kind in ("fusion", "mlp"):
            row = mf.recording_level_stats(mf.extract_frame_features(y, sr))
            vec = np.array([[row[c] for c in self.labels["feature_columns"]]], dtype=float)
            if not np.isfinite(vec).all():
                raise ValueError("non-finite acoustic feature(s) extracted")
            x["features"] = md.apply_feature_scaler(vec, self.sc["feat"])
        return x

    def predict_file(self, path) -> dict:
        y, sr = self.load_audio(path)
        return self.predict_audio(y, sr)

    def predict_audio(self, y, sr) -> dict:
        """Run the network on an already standardised signal (16 kHz mono) and decide hierarchically."""
        out = self.model.predict(self.inputs_from_audio(y, sr), verbose=0)
        pr, pa = out["region_output"][0], out["accent_output"][0]
        res = hierarchical_decode(pr, pa, self.labels["maps"])[0]
        res["region_probabilities"] = dict(zip(self.labels["maps"]["regions"], map(float, pr)))
        res["accent_probabilities"] = dict(zip(self.labels["maps"]["accents"], map(float, pa)))
        return res


# --------------------------------------------------------------------------- #
# Training one fold
# --------------------------------------------------------------------------- #
def trainable_snapshot(model) -> list[np.ndarray]:
    return [w.numpy().copy() for w in model.trainable_weights]


def predict_split(model, ids, ctx, sc, kind, maps) -> pd.DataFrame:
    x, _ = batch_from_ids(list(ids), ctx, sc, kind, augment=False)
    out = model.predict(x, verbose=0, batch_size=8)
    pr, pa = out["region_output"], out["accent_output"]
    dec = hierarchical_decode(pr, pa, maps)
    df = pd.DataFrame({"recording_id": list(ids), "true_region": ctx["meta"].set_index("recording_id").loc[list(ids), "region"].to_numpy(),
                       "true_accent": ctx["meta"].set_index("recording_id").loc[list(ids), "accent"].to_numpy(),
                       "pred_region": [d["region"] for d in dec], "pred_accent_hier": [d["accent"] for d in dec],
                       "pred_accent_head": [maps["accents"][i] for i in pa.argmax(1)],
                       "region_confidence": [d["region_confidence"] for d in dec],
                       "accent_confidence_within_region": [d["accent_confidence_within_region"] for d in dec],
                       "accent_head_probability": [d["accent_head_probability"] for d in dec],
                       "joint_confidence": [d["joint_confidence"] for d in dec]})
    for j, r in enumerate(maps["regions"]):
        df[f"p_region::{r}"] = pr[:, j]
    for j, a in enumerate(maps["accents"]):
        df[f"p_accent::{a}"] = pa[:, j]
    return df


def train_fold(kind: str, fold: dict, ctx: dict, run_dir: Path, augment: bool = True, feature_columns=None,
               train_cfg: dict = TRAIN, verbose: int = 0, bundle_name: str | None = None) -> dict:
    """Fit normalisation on the training recordings, train from scratch, reload the best checkpoint, predict."""
    maps = ctx["maps"]
    feature_columns = feature_columns or ctx["feature_columns"]
    ctx = {**ctx, "X": ctx["X_all"][feature_columns]}
    fdir = Path(run_dir) / (bundle_name or f"fold_{fold['fold']}")
    fdir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, test_ids = fold["train"], fold["val"], fold["test"]

    # ---- normalisation fitted on the TRAINING recordings of this fold only ----
    sc = {"spec": md.fit_spec_scaler([ctx["specs"][r] for r in train_ids]),
          "feat": md.fit_feature_scaler(ctx["X"].loc[train_ids].to_numpy(float))}
    np.savez(fdir / "preprocessing.npz", spec_mean=sc["spec"]["mean"], spec_std=sc["spec"]["std"],
             feat_mean=sc["feat"]["mean"], feat_std=sc["feat"]["std"])
    (fdir / "labels.json").write_text(json.dumps({
        "kind": kind, "maps": maps, "feature_columns": list(feature_columns), "T": md.T_FRAMES,
        "spectrogram_config": md.SPEC, "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "augmentation": md.AUGMENT if augment else None}, indent=2), encoding="utf-8")

    set_seeds(train_cfg["seed"] + fold["fold"])
    model = build_model(kind, len(feature_columns), len(maps["regions"]), len(maps["accents"]))
    compile_model(model, train_cfg["learning_rate"])
    before = trainable_snapshot(model)

    seq = TrainSequence(train_ids, ctx, sc, kind, augment, train_cfg["batch_size"], train_cfg["repeats_per_epoch"],
                        seed=train_cfg["seed"] + 100 * fold["fold"])
    val_x, val_y = batch_from_ids(list(val_ids), ctx, sc, kind, augment=False)
    ckpt = fdir / "model.keras"
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg["early_stopping_patience"], restore_best_weights=True),
                 keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=train_cfg["reduce_lr_factor"], patience=train_cfg["reduce_lr_patience"])]
    hist = model.fit(seq, validation_data=(val_x, val_y), epochs=train_cfg["max_epochs"], callbacks=callbacks, verbose=verbose)
    h = pd.DataFrame(hist.history)
    h.insert(0, "epoch", np.arange(1, len(h) + 1))
    h.insert(0, "fold", fold["fold"])
    h.to_csv(fdir / "history.csv", index=False)

    after = trainable_snapshot(model)
    changed = [bool(not np.allclose(a, b)) for a, b in zip(before, after)]
    reloaded = keras.models.load_model(ckpt, compile=False)                  # the saved best checkpoint
    preds = []
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        if not len(ids):                       # e.g. the final all-data model has no held-out test recordings
            continue
        p = predict_split(reloaded, ids, ctx, sc, kind, maps)
        p.insert(0, "split", split)
        p.insert(0, "fold", fold["fold"])
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    # the restored in-memory model and the reloaded checkpoint must agree
    p_mem = model.predict(val_x, verbose=0)["accent_output"]
    p_ckpt = reloaded.predict(val_x, verbose=0)["accent_output"]
    best = int(np.argmin(hist.history["val_loss"]))
    info = {"fold": fold["fold"], "epochs_run": len(h), "best_epoch": best + 1, "n_params_trainable": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
            "n_params_total": int(model.count_params()), "weights_changed_fraction": float(np.mean(changed)),
            "all_trainable_tensors_updated": bool(all(changed)), "checkpoint_matches_restored_weights": bool(np.allclose(p_mem, p_ckpt, atol=1e-5)),
            "final_lr": float(h["learning_rate"].iloc[-1]) if "learning_rate" in h else None}
    keras.backend.clear_session()
    return {"history": h, "predictions": preds, "info": info, "dir": fdir}


# --------------------------------------------------------------------------- #
# SVM baseline (experiment A)
# --------------------------------------------------------------------------- #
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale")


def svm_fold(fold: dict, ctx: dict, feature_columns, shuffle_labels_seed: int | None = None) -> pd.DataFrame:
    maps, X, meta = ctx["maps"], ctx["X_all"][feature_columns], ctx["meta"].set_index("recording_id")
    tr = fold["train"]
    y = meta.loc[tr, "accent"].to_numpy()
    if shuffle_labels_seed is not None:                                   # sanity control: labels shuffled in TRAINING only
        y = np.random.default_rng(shuffle_labels_seed).permutation(y)
    clf = make_pipeline(StandardScaler(), SVC(**SVM_PARAMS, random_state=0)).fit(X.loc[tr].to_numpy(float), y)
    rows = []
    for split, ids in (("train", tr), ("val", fold["val"]), ("test", fold["test"])):
        pred = clf.predict(X.loc[ids].to_numpy(float))
        for rid, p in zip(ids, pred):
            rows.append({"fold": fold["fold"], "split": split, "recording_id": rid, "true_region": meta.loc[rid, "region"],
                         "true_accent": meta.loc[rid, "accent"], "pred_accent_head": p, "pred_accent_hier": p,
                         "pred_region": maps["accent_to_region"][p]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _prf(y_true, y_pred, labels) -> dict:
    p, r, f, _ = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(skm.accuracy_score(y_true, y_pred)), "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f)}


def evaluate_run(preds: pd.DataFrame, maps: dict, split: str = "test") -> dict:
    """Pooled out-of-fold metrics, fold-wise accuracies and confusion matrices for one experiment."""
    d = preds[preds.split == split]
    acc_m, reg_m = _prf(d.true_accent, d.pred_accent_head, maps["accents"]), _prf(d.true_region, d.pred_region, maps["regions"])
    hier = _prf(d.true_accent, d.pred_accent_hier, maps["accents"])
    e2e = float(((d.pred_region == d.true_region) & (d.pred_accent_hier == d.true_accent)).mean())
    per_fold = d.groupby("fold").apply(lambda g: pd.Series({
        "n_test": len(g), "region_accuracy": float((g.pred_region == g.true_region).mean()),
        "accent_accuracy": float((g.pred_accent_head == g.true_accent).mean()),
        "hierarchical_end_to_end_accuracy": float(((g.pred_region == g.true_region) & (g.pred_accent_hier == g.true_accent)).mean())}),
        include_groups=False).reset_index()
    per_class = pd.DataFrame(skm.precision_recall_fscore_support(d.true_accent, d.pred_accent_head, labels=maps["accents"], zero_division=0),
                             index=["precision", "recall", "f1", "support"], columns=maps["accents"]).T.reset_index(names="accent")
    return {"n": len(d), "accent_head": acc_m, "region_head": reg_m, "hierarchical_accent": hier, "hierarchical_end_to_end_accuracy": e2e,
            "fold_accuracy_mean_sd": {c: [float(per_fold[c].mean()), float(per_fold[c].std(ddof=1)) if len(per_fold) > 1 else 0.0]
                                      for c in ("region_accuracy", "accent_accuracy", "hierarchical_end_to_end_accuracy")},
            "per_fold": per_fold, "per_class": per_class,
            "confusion_accent": pd.DataFrame(skm.confusion_matrix(d.true_accent, d.pred_accent_head, labels=maps["accents"]), index=maps["accents"], columns=maps["accents"]),
            "confusion_region": pd.DataFrame(skm.confusion_matrix(d.true_region, d.pred_region, labels=maps["regions"]), index=maps["regions"], columns=maps["regions"]),
            "heads_agree_fraction": float((d.pred_accent_head == d.pred_accent_hier).mean())}


def probability_metrics(preds: pd.DataFrame, maps: dict, split: str = "test") -> dict:
    """Mean test cross-entropy (NLL) of the accent head and mean top-class probability (NaN for the SVM, which has no probabilities)."""
    d = preds[preds.split == split]
    cols = [f"p_accent::{a}" for a in maps["accents"]]
    if not set(cols) <= set(d.columns):
        return {"test_accent_nll": float("nan"), "mean_top_probability": float("nan"), "mean_true_class_probability": float("nan")}
    P = d[cols].to_numpy(float)
    idx = np.array([maps["accents"].index(a) for a in d.true_accent])
    true_p = P[np.arange(len(d)), idx]
    return {"test_accent_nll": float(-np.log(np.clip(true_p, 1e-12, 1)).mean()), "mean_top_probability": float(P.max(1).mean()),
            "mean_true_class_probability": float(true_p.mean())}


def save_evaluation(res: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    res["per_fold"].to_csv(run_dir / "fold_metrics.csv", index=False)
    res["per_class"].to_csv(run_dir / "per_class_metrics.csv", index=False)
    res["confusion_accent"].to_csv(run_dir / "confusion_matrix_accent.csv")
    res["confusion_region"].to_csv(run_dir / "confusion_matrix_region.csv")
    summary = {k: v for k, v in res.items() if k not in ("per_fold", "per_class", "confusion_accent", "confusion_region")}
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def overfit_table(preds: pd.DataFrame, hist: pd.DataFrame | None, infos: list[dict]) -> pd.DataFrame:
    """Per fold: accent accuracy on the (clean) training, validation and test recordings + epochs used."""
    rows = []
    for f, g in preds.groupby("fold"):
        row = {"fold": f}
        for s in ("train", "val", "test"):
            gs = g[g.split == s]
            row[f"{s}_accent_accuracy"] = float((gs.pred_accent_head == gs.true_accent).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if infos:
        inf = pd.DataFrame(infos)[["fold", "epochs_run", "best_epoch"]]
        out = out.merge(inf, on="fold")
    return out
