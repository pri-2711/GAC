"""Phase 6 figures: architecture diagram, training curves, confusion matrices, comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as cfg
import model_data as md
import plot_utils as pu

REPORT_FIG_DIR = cfg.PROJECT_REPORTS_DIR / "figures"
ARCH_PNG = REPORT_FIG_DIR / "phase6_hierarchical_model_architecture.png"


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #
def architecture_diagram(path: Path = ARCH_PNG, T: int = md.T_FRAMES, n_features: int = 78, n_trainable: int | None = None,
                         n_bn_stats: int | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(27, 11.5))
    ax.set_xlim(0, 176)
    ax.set_ylim(0, 90)
    ax.axis("off")
    C = {"io": "#E8EEF7", "pre": "#FFF2CC", "conv": "#CFE2F3", "mlp": "#D9EAD3", "fuse": "#F4CCCC", "head": "#EAD1F5", "out": "#FCE5CD"}

    def box(x, y, w, h, text, color, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def arrow(p, q, label=None, fs=9, dy=1.0, dx=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333333", shrinkA=0, shrinkB=0))
        if label:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=fs, color="#7A1F1F", style="italic")

    ymid, ycnn, ymlp, H = 42, 63, 21, 15
    ax.text(88, 87.6, "Custom Dual-Branch CNN–MLP Feature-Fusion Network with Hierarchical Multi-Task Classification",
            ha="center", fontsize=17, fontweight="bold")
    ax.text(88, 84.6, "Trained from scratch: random initialisation, no pretrained weights, embeddings or encoders  ·  tensor shapes exclude the batch dimension",
            ha="center", fontsize=11, color="#444444")

    box(1, ymid - 7, 12, 14, "Audio input\n\nstandardised WAV\n16 kHz · mono\n(Phase 1)", C["io"], 9.5)
    arrow((13.6, ymid + 3), (17.4, ycnn + 4))
    arrow((13.6, ymid - 3), (17.4, ymlp + H - 4))

    # Branch A
    ax.text(18, ycnn + H + 3.2, "BRANCH A: 2-D CNN on the Log-Mel spectrogram", fontsize=12.5, fontweight="bold", color="#1F4E79")
    box(18, ycnn, 13, H, f"Log-Mel spectrogram\n64 Mel bands · 25 ms Hann\nwindow (400) · 10 ms hop (160)\nn_fft 512 · power → dB\ncrop / pad to T = {T}", C["pre"], 7.6)
    box(34, ycnn, 12, H, f"Input tensor\n\n(64, {T}, 1)\n\nnormalised per Mel band\n(training fold only)", C["io"], 8.6)
    box(49, ycnn, 15, H, f"Conv block 1\nConv2D 32 · 3×3 · same\nuse_bias = False\nBatchNorm → ReLU\nMaxPool 2×2\n→ (32, {T // 2}, 32)", C["conv"], 8.2)
    box(67, ycnn, 15, H, f"Conv block 2\nConv2D 64 · 3×3 · same\nuse_bias = False\nBatchNorm → ReLU\nMaxPool 2×2\n→ (16, {T // 4}, 64)", C["conv"], 8.2)
    box(85, ycnn, 15, H, f"Conv block 3\nConv2D 128 · 3×3 · same\nuse_bias = False\nBatchNorm → ReLU\n(no pooling)\n→ (16, {T // 4}, 128)", C["conv"], 8.2)
    box(103, ycnn, 14, H, "GlobalAvgPool2D\n→ (128)\nDense 128 · ReLU\nDropout 0.40\n\nCNN output: 128-d", C["conv"], 8.4)
    for a, b in ((31.4, 33.6), (46.4, 48.6), (64.4, 66.6), (82.4, 84.6), (100.4, 102.6)):
        arrow((a, ycnn + H / 2), (b, ycnn + H / 2))

    # Branch B
    ax.text(18, ymlp + H + 3.2, "BRANCH B: MLP on the numerical acoustic features", fontsize=12.5, fontweight="bold", color="#38761D")
    box(18, ymlp, 13, H, f"Phase 3 features\n{n_features} per recording\nMFCC, Δ, ΔΔ\n(mean and std of 13\ncoefficients each)", C["pre"], 8.3)
    box(34, ymlp, 12, H, f"Input vector\n\n({n_features})\n\nz-score using the\ntraining fold only", C["io"], 8.6)
    box(49, ymlp, 15, H, "Dense 64 · ReLU\nBatchNorm\nDropout 0.30\n\n→ (64)", C["mlp"], 8.6)
    box(67, ymlp, 15, H, "Dense 32 · ReLU\nDropout 0.20\n\n→ (32)", C["mlp"], 8.6)
    box(103, ymlp, 14, H, "MLP output\n\n32-d\nrepresentation", C["mlp"], 9)
    for a, b in ((31.4, 33.6), (46.4, 48.6), (64.4, 66.6)):
        arrow((a, ymlp + H / 2), (b, ymlp + H / 2))
    arrow((82.4, ymlp + H / 2), (102.6, ymlp + H / 2))

    # fusion
    box(121, ymid - 8, 12, 16, "Concatenate\n(axis = −1)\n\n128 + 32\n= 160-d\nfused vector", C["fuse"], 9)
    arrow((117.4, ycnn + 3), (120.6, ymid + 3), "128", dx=-1.5)
    arrow((117.4, ymlp + H - 3), (120.6, ymid - 3), "32", dx=-3.2, dy=-4.5)
    box(121, 8, 12, 17.5, "Shared dense\n\nDense 64 · ReLU\nDropout 0.40\n\n→ (64)", C["fuse"], 9)
    arrow((127, ymid - 8.4), (127, 26.1), "160", dx=2.5, dy=-1)

    # heads
    box(137, 33, 12, 12, "Region head\nDense(5)\nsoftmax\n\nname:\nregion_output", C["head"], 8.6)
    box(137, 8, 12, 12, "Accent head\nDense(5)\nsoftmax\n\nname:\naccent_output", C["head"], 8.6)
    arrow((133.4, 14), (136.6, 14), "64", dy=0.5)
    arrow((133.4, 24), (136.6, 37), "64", dx=-1.2)
    ax.text(143, 46.8, "p(region): 5 probabilities", ha="center", fontsize=8.6, color="#7A1F1F", style="italic")
    ax.text(143, 21.2, "p(accent): 5 probabilities", ha="center", fontsize=8.6, color="#7A1F1F", style="italic")

    # hierarchical inference
    box(155, 10, 19, 37, "Hierarchical inference\n(reusable predictor)\n\n1. region =\nargmax p(region)\n\n2. allowed accents =\nthose mapped to that\nregion\n(region → accent map)\n\n3. accent = argmax of\np(accent) over the\nallowed accents,\nrenormalised\n\n→ final region,\naccent and\nconfidence values", C["out"], 8.6)
    arrow((149.4, 39), (154.6, 39))
    arrow((149.4, 14), (154.6, 18))

    params = f"Trainable parameters: {n_trainable:,}" + (f" (plus {n_bn_stats:,} non-trainable BatchNorm statistics)" if n_bn_stats else "") if n_trainable else ""
    ax.text(2, 1.0, "Joint training: total loss = λ_region · CE(region) + λ_accent · CE(accent) with λ = 1.0 / 1.0 (sparse categorical cross-entropy); Adam, lr 1e-3.   "
            f"{params}.\nWith one speaker per accent the 5 regions map one-to-one onto the 5 accents, so the two heads are currently equivalent five-class problems.",
            fontsize=9.6, va="bottom", color="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Training curves and evaluation figures
# --------------------------------------------------------------------------- #
def training_curves(history: pd.DataFrame, title: str, path: Path) -> Path:
    pu.apply_style()
    panels = [("loss", "Total loss (weighted sum)"), ("region_output_loss", "Region loss"), ("accent_output_loss", "Accent loss"),
              ("region_output_accuracy", "Region accuracy"), ("accent_output_accuracy", "Accent accuracy"), ("learning_rate", "Learning rate")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        for fold, g in history.groupby("fold"):
            if col == "learning_rate":
                ax.plot(g.epoch, g[col], color="#4C78A8", alpha=0.4, lw=1.2)
                continue
            ax.plot(g.epoch, g[col], color="#4C78A8", alpha=0.22, lw=1)
            ax.plot(g.epoch, g["val_" + col], color="#D55E00", alpha=0.22, lw=1)
        if col != "learning_rate":
            for pref, c, name in (("", "#4C78A8", "training"), ("val_", "#D55E00", "validation")):
                m = history.groupby("epoch")[pref + col].mean()
                n = history.groupby("epoch")[pref + col].count()
                m = m[n >= max(2, 0.5 * history.fold.nunique())]
                ax.plot(m.index, m.values, color=c, lw=2.4, label=f"{name} (mean of folds)")
            ax.legend(fontsize=8)
        else:
            ax.set_yscale("log")
        ax.set(title=lab, xlabel="Epoch")
        if "accuracy" in col:
            ax.set_ylim(0, 1.03)
    fig.suptitle(f"{title}\nthin lines: individual folds; thick lines: mean over folds (epochs where at least half of the folds are still training)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_grid(cms: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    pu.apply_style()
    n = len(cms)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.2 * rows), squeeze=False)
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=max(1, cm.to_numpy().max()))
        ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm.index)), cm.index, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.iat[i, j]
                ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.to_numpy().max() / 2 else "black", fontsize=11)
        ax.set(title=name, xlabel="Predicted", ylabel="True")
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def comparison_bars(table: pd.DataFrame, path: Path) -> Path:
    pu.apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(table))
    ax.bar(x, table.pooled_accent_accuracy, color="#4C78A8", width=0.55, label="pooled accuracy over all test recordings")
    ax.errorbar(x, table.fold_mean, yerr=table.fold_sd, fmt="o", color="#D55E00", capsize=5, label="fold-wise mean ± SD")
    for i, v in enumerate(table.pooled_accent_accuracy):
        ax.text(i + 0.33, v, f"{v:.2f}", ha="left", va="center", fontsize=9)
    ax.axhline(1 / 5, color="grey", ls=":", label="chance (5 classes)")
    ax.set_xticks(x, table.name, rotation=25, ha="right", fontsize=9)
    ax.set(ylim=(0, 1.12), ylabel="Accent accuracy (within-speaker, exploratory)", title="Experiments A–D and supplementary runs (identical folds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def overfit_bars(table: pd.DataFrame, path: Path) -> Path:
    pu.apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(table))
    w = 0.27
    for k, (c, lab, col) in enumerate((("train_accent_accuracy", "training recordings (clean, no augmentation)", "#4C78A8"),
                                       ("val_accent_accuracy", "validation", "#E69F00"), ("test_accent_accuracy", "test (held-out recordings)", "#D55E00"))):
        ax.bar(x + (k - 1) * w, table[c], w, label=lab, color=col)
    ax.set_xticks(x, table.name, rotation=25, ha="right", fontsize=9)
    ax.set(ylim=(0, 1.1), ylabel="Mean accent accuracy over folds", title="Training vs validation vs test accuracy")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def example_spectrograms(specs: dict, meta: pd.DataFrame, path: Path, maps: dict) -> Path:
    pu.apply_style()
    accents = maps["accents"]
    fig, axes = plt.subplots(1, len(accents), figsize=(4.2 * len(accents), 3.9), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], accents):
        rid = meta[meta.accent == a].recording_id.iloc[0]
        s = specs[rid]
        im = ax.imshow(s[:, :md.T_FRAMES], origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=max(5, float(s.max())))
        ax.set(title=f"{a} ({rid})", xlabel="Frame (10 ms)")
        ax.grid(False)
    axes[0, 0].set_ylabel("Mel band (0-63)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="dB", shrink=0.9)
    fig.suptitle(f"Log-Mel inputs: first {md.T_FRAMES} frames ({md.T_FRAMES / 100:.0f} s) of one recording per accent (before normalisation)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fold_heatmap(fm: pd.DataFrame, path: Path) -> Path:
    """Accent accuracy per fold (columns) for every completed run (rows)."""
    pu.apply_style()
    fig, ax = plt.subplots(figsize=(1.0 * fm.shape[1] + 7, 0.55 * len(fm) + 2))
    im = ax.imshow(fm.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(fm.shape[1]), [f"fold {c}" for c in fm.columns])
    ax.set_yticks(range(len(fm)), fm.index, fontsize=8)
    for i in range(fm.shape[0]):
        for j in range(fm.shape[1]):
            ax.text(j, i, f"{fm.iat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.grid(False)
    ax.set_title("Accent accuracy of each held-out fold (5 recordings per fold, one per accent)")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def batchnorm_diagnostic(bn: pd.DataFrame, fold_metrics: pd.DataFrame, path: Path) -> Path:
    """Stored BatchNorm running variance / true activation variance (last conv layer) against the epoch of the saved checkpoint."""
    pu.apply_style()
    d = bn[bn.bn_layer == "conv3_bn"].merge(fold_metrics[["run", "fold", "accent_accuracy"]], on=["run", "fold"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (run, mk) in (("cnn_only", "o"), ("cnn_only/supplementary_patience25", "s"), ("fusion_model", "^")):
        g = d[d.run == run]
        sc = ax.scatter(g.best_epoch, g.median_var_ratio_running_over_true, c=g.accent_accuracy, cmap="RdYlGn", vmin=0, vmax=1,
                        marker=mk, s=90, edgecolor="black", label=run)
    ax.axhline(1, color="grey", ls=":")
    ax.set(xscale="log", yscale="log", xlabel="Epoch of the saved (best-validation) checkpoint", ylabel="Stored BatchNorm variance / true variance (conv block 3)",
           title="Early checkpoints carry unconverged BatchNorm statistics (ratio ≫ 1)")
    fig.colorbar(sc, ax=ax, label="test accent accuracy of that fold")
    ax.legend(fontsize=8, title="run (marker)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
