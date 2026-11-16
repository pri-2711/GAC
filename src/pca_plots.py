"""Figures for Phase 5 (PCA). Every figure is drawn from the objects returned by pca_analysis.run_pipeline()."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
import pca_analysis as pa
import plot_utils as pu

FIG_DIR = pa.FIG_DIR
DPI_NOTE = "Exploratory only: one speaker per accent."


def short_id(rid: str) -> str:
    """'AHM_S01_R03' -> 'AHM-R03' (speaker part dropped while there is one speaker per accent)."""
    parts = rid.split("_")
    return f"{parts[0]}-{parts[-1]}"


def _scatter(ax, scores, table, i, j, colors, markers, evr, label=True, highlight=None):
    for a in colors:
        for fmt in markers:
            m = ((table.accent == a) & (table.original_format == fmt)).to_numpy()
            if not m.any():
                continue
            faded = highlight is not None and a != highlight
            ax.scatter(scores[m, i], scores[m, j], s=55, color="#BBBBBB" if faded else colors[a], marker=markers[fmt],
                       edgecolor="white" if not faded else "none", linewidth=0.6, alpha=0.7 if faded else 0.95,
                       zorder=2 if faded else 3)
    if label:
        for r, (x, y) in enumerate(zip(scores[:, i], scores[:, j])):
            if highlight is None or table.accent.iloc[r] == highlight:
                ax.annotate(short_id(table.recording_id.iloc[r]), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC{i + 1} ({evr[i]:.1%} of variance)")
    ax.set_ylabel(f"PC{j + 1} ({evr[j]:.1%} of variance)")


def make_all(d: dict) -> None:
    pu.apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table, accents, formats = d["table"], d["accents"], d["formats"]
    colors, markers = pu.accent_colors(accents), pu.format_markers(formats)
    m, out = d["primary"], d["out"]
    evr, sc, k = m["evr"], m["scores"], len(m["evr"])
    handles = pu.legend_handles(accents, colors, formats, markers)

    # 1. scree plot
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, k + 1)
    ax.bar(x, evr * 100, color="#4C78A8")
    for i in range(min(5, k)):
        ax.text(x[i], evr[i] * 100 + 0.4, f"{evr[i] * 100:.1f}%", ha="center", fontsize=8)
    ax.axhline(100 / len(m["features"]), color="#D55E00", ls="--", lw=1,
               label=f"Average per feature ({100 / len(m['features']):.1f}%)")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", title="Scree plot: variance explained by each component")
    ax.legend()
    pu.save_figure(fig, FIG_DIR / "scree_plot.png", show=False)

    # 2. cumulative variance
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cum = np.cumsum(evr) * 100
    ax.plot(x, cum, marker="o", ms=3, color="#4C78A8")
    for t, n_c in pa.components_for(evr).items():
        ax.axhline(t * 100, color="grey", ls=":", lw=1)
        if n_c:
            ax.plot([n_c], [t * 100], "o", color="#D55E00")
            ax.annotate(f"{n_c} PCs for {t:.0%}", (n_c, t * 100), xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance (%)", ylim=(0, 102),
           title="Cumulative explained variance")
    pu.save_figure(fig, FIG_DIR / "cumulative_variance.png", show=False)

    # 3. pairwise score plots
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if j >= k:
            continue
        fig, ax = plt.subplots(figsize=(8, 6.2))
        _scatter(ax, sc, table, i, j, colors, markers, evr)
        ax.set_title(f"PC{i + 1} vs PC{j + 1}: recordings in the reduced space\n({DPI_NOTE} Not an accent classifier.)")
        ax.legend(handles=handles, loc="best", fontsize=8, ncol=2)
        pu.save_figure(fig, FIG_DIR / f"pc{i + 1}_pc{j + 1}_scatter.png", show=False)

    # 4. loadings: top features for PC1-PC3 and group contribution
    load = out["loadings"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    group_color = {"mfcc": "#0072B2", "delta": "#E69F00", "delta2": "#009E73"}
    for ax, pc in zip(axes, ("PC1", "PC2", "PC3")):
        top = load.reindex(load[pc].abs().sort_values(ascending=False).index[:10]).iloc[::-1]
        ax.barh(top.label, top[pc], color=[group_color[g] for g in top.feature_group])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{pc}: 10 largest loadings ({evr[int(pc[2:]) - 1]:.1%} of variance)")
        ax.set_xlabel("Loading (weight on the standardised feature)")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in group_color.values()],
                   labels=[{"mfcc": "MFCC", "delta": "Delta", "delta2": "Delta-delta"}[g] for g in group_color], loc="lower right", fontsize=8)
    fig.tight_layout()
    pu.save_figure(fig, FIG_DIR / "loadings_top_features.png", show=False)

    gc = out["group_contrib"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    pcs = sorted(gc.component.unique(), key=lambda s: int(s[2:]))
    bottom = np.zeros(len(pcs))
    palette = ["#0072B2", "#56B4E9", "#E69F00", "#F0C06A", "#009E73", "#7FD1B3"]
    for color, ((grp, stat), g) in zip(palette, gc.groupby(["feature_group", "statistic"], sort=False)):
        vals = g.set_index("component").loc[pcs, "share_of_squared_loadings"].to_numpy() * 100
        ax.bar(pcs, vals, bottom=bottom, color=color, label=f"{ {'mfcc': 'MFCC', 'delta': 'Delta', 'delta2': 'Delta-delta'}[grp]} {stat}")
        bottom += vals
    ax.set(ylabel="Share of the component's squared loadings (%)", title="Which feature families make up each leading component")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    pu.save_figure(fig, FIG_DIR / "loadings_group_contribution.png", show=False)

    # 5. recording-condition overlays on PC1-PC2
    cols = [("speech_ratio", "Speech-frame share (30 dB rule)", "viridis"),
            ("noise_floor_dbfs", "Noise floor (dBFS)", "magma"),
            ("original_sample_rate_hz", "Original sample rate (Hz)", "cividis")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, (col, label, cmap) in zip(axes, cols):
        for fmt in formats:
            mk = (table.original_format == fmt).to_numpy()
            s = ax.scatter(sc[mk, 0], sc[mk, 1], c=table[col][mk], cmap=cmap, vmin=table[col].min(), vmax=table[col].max(),
                           marker=markers[fmt], s=60, edgecolor="black", linewidth=0.4)
        fig.colorbar(s, ax=ax, label=label)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
        ax.set_title(f"PC1-PC2 coloured by {label.lower()}", fontsize=9)
    axes[0].legend(handles=[plt.Line2D([], [], marker=markers[f], ls="", color="grey", mfc="none", label=f"{f} file") for f in formats], fontsize=8)
    fig.tight_layout()
    pu.save_figure(fig, FIG_DIR / "condition_overlays_pc1_pc2.png", show=False)

    # 6. association of leading components with accent / format / sample rate (eta^2) and with quality metrics (r)
    et = out["eta2"].head(6)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    w = 0.27
    xs = np.arange(len(et))
    for off, (col, lab, c) in enumerate((("eta2_accent", "Accent", "#4C78A8"), ("eta2_original_format", "Original format", "#E69F00"),
                                         ("eta2_original_sample_rate", "Original sample rate", "#009E73"))):
        axes[0].bar(xs + (off - 1) * w, et[col], w, label=lab, color=c)
    axes[0].set_xticks(xs, et.component)
    axes[0].set(ylabel="eta-squared of PC scores", ylim=(0, 1.05), title="How much of each PC is explained by each grouping")
    axes[0].legend(fontsize=8)
    cond = out["conditions"]
    piv = cond.pivot(index="label", columns="component", values="pearson_overall")[["PC1", "PC2", "PC3", "PC4", "PC5"]]
    piv = piv.loc[[v for v in pa.CONDITION_COLUMNS.values() if v in piv.index]]
    im = axes[1].imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(piv.shape[1]), piv.columns)
    axes[1].set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            axes[1].text(c, r, f"{piv.iat[r, c]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], label="Pearson r (all recordings)")
    axes[1].set_title("PC scores vs recording-condition metrics")
    fig.tight_layout()
    pu.save_figure(fig, FIG_DIR / "pc_condition_associations.png", show=False)

    # 7. leave-one-accent-out view for Surti (if present): full PCA vs PCA fitted without it
    focus = "Surti" if "Surti" in accents else None
    if focus:
        rest = table.accent != focus
        Xr, fr, _ = pa.prepare_matrix(table[rest], m["features"])
        mr = pa.fit_pca(Xr, fr)
        proj = pa.transform(mr, table.loc[~rest, fr].to_numpy(float))
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        _scatter(axes[0], sc, table, 0, 1, colors, markers, evr, label=True, highlight=focus)
        axes[0].set_title(f"A. Primary PCA (all accents); {focus} highlighted")
        for a in accents:
            if a == focus:
                continue
            for fmt in formats:
                mk = ((table[rest].accent == a) & (table[rest].original_format == fmt)).to_numpy()
                if mk.any():
                    axes[1].scatter(mr["scores"][mk, 0], mr["scores"][mk, 1], s=55, color=colors[a], marker=markers[fmt],
                                    edgecolor="white", linewidth=0.6)
        axes[1].scatter(proj[:, 0], proj[:, 1], s=70, color=colors[focus], marker="*", edgecolor="black", linewidth=0.5,
                        label=f"{focus} (projected, not used to fit)", zorder=4)
        for r, (xx, yy) in enumerate(proj[:, :2]):
            axes[1].annotate(short_id(table.loc[~rest, "recording_id"].iloc[r]), (xx, yy), fontsize=6, xytext=(3, 3), textcoords="offset points")
        axes[1].axhline(0, color="grey", lw=0.5)
        axes[1].axvline(0, color="grey", lw=0.5)
        axes[1].set_xlabel(f"PC1 ({mr['evr'][0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({mr['evr'][1]:.1%})")
        axes[1].set_title(f"B. PCA fitted WITHOUT {focus}; {focus} projected into it")
        axes[1].legend(handles=handles + [plt.Line2D([], [], marker="*", ls="", color=colors[focus], mec="black", label=f"{focus} (projected)")],
                       fontsize=8, ncol=2, loc="best")
        fig.tight_layout()
        pu.save_figure(fig, FIG_DIR / "surti_leave_out_pc1_pc2.png", show=False)

    # 8. sensitivity: PC1-PC2 of each variant, and cumulative-variance curves
    fits, summary = d["sens_fits"], out["sensitivity"].set_index("variant")
    names = list(fits)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, name in zip(axes.ravel(), names):
        f = fits[name]
        _scatter(ax, f["scores"], table, 0, 1, colors, markers, f["evr"], label=False)
        ax.set_title(f"{summary.loc[name, 'description']} ({summary.loc[name, 'n_features']} features)", fontsize=10)
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2)
    fig.tight_layout()
    pu.save_figure(fig, FIG_DIR / "sensitivity_pc1_pc2.png", show=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, c in zip(names, ("#000000", "#0072B2", "#D55E00", "#009E73")):
        ev = fits[name]["evr"]
        ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev) * 100, label=summary.loc[name, "description"], color=c, lw=1.8)
    ax.axhline(90, color="grey", ls=":", lw=1)
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance (%)", title="Cumulative variance: primary vs sensitivity analyses")
    ax.legend(fontsize=8)
    pu.save_figure(fig, FIG_DIR / "sensitivity_cumulative_variance.png", show=False)
