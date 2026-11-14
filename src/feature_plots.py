"""Plotting helpers for Phase 4 (all figures are drawn from real dataset outputs).

Every function takes the accent list / colour / marker maps, so a newly added accent
(e.g. Surti) is plotted automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import feature_analysis as fa
import mfcc_features as mf

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.grid": True, "grid.alpha": 0.2})


def save(fig, name: str, show: bool = True):
    fa.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = fa.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print("saved:", path.relative_to(fa.cfg.PROJECT_ROOT).as_posix())
    if show:
        plt.show()
    plt.close(fig)


def _legend_handles(accents, colors, formats, markers):
    h = [plt.Line2D([], [], marker="o", ls="", color=colors[a], label=a) for a in accents]
    h += [plt.Line2D([], [], marker=markers[f], ls="", color="grey", mfc="none", label=f"{f} file") for f in formats]
    return h


def _strip(ax, df, col, accents, colors, markers, box=True):
    """Boxplot (if >=3 points) with every recording drawn on top; marker = original format."""
    for i, a in enumerate(accents):
        sub = df[df.accent == a]
        v = sub[col].dropna().to_numpy(float)
        if not len(v):
            continue
        if box and len(v) >= 3:
            bp = ax.boxplot([v], positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                            medianprops={"color": "black"})
            bp["boxes"][0].set_facecolor(colors[a]); bp["boxes"][0].set_alpha(0.35)
        jitter = np.linspace(-0.17, 0.17, len(v))
        for x, (_, r) in zip(jitter, sub[sub[col].notna()].iterrows()):
            ax.scatter(i + x, r[col], color=colors[a], marker=markers[r["original_format"]], s=22,
                       edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(accents))); ax.set_xticklabels([a[:4] for a in accents], fontsize=8)


def feature_grid(table, group, stat, accents, colors, markers, title, name, ylabel_note=""):
    """13 panels (one per coefficient): recording-level values per accent."""
    fig, axes = plt.subplots(4, 4, figsize=(15, 12.5))
    axes = axes.ravel()
    formats = sorted(table.original_format.unique())
    for k in range(mf.N_COEF):
        col = f"{group}_{k + 1:02d}_{stat}"
        _strip(axes[k], table, col, accents, colors, markers)
        axes[k].set_title(f"{mf.FEATURE_GROUPS[group]} {k + 1} ({stat})")
    for ax in axes[mf.N_COEF:]:
        ax.axis("off")
    axes[13].legend(handles=_legend_handles(accents, colors, formats, markers), loc="upper left", frameon=False)
    axes[13].text(0.0, 0.05, "Each dot = one recording." + chr(10) + "Boxes: middle half of recordings." + chr(10) + ylabel_note,
                  transform=axes[13].transAxes, fontsize=9, va="bottom")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    save(fig, name)


def pooled_frames(ids_by_accent: dict, coef: int, masks: dict | None = None, group="mfcc"):
    """Pool frame values of one coefficient per accent (all frames, or speech-only if masks given)."""
    out = {}
    for a, ids in ids_by_accent.items():
        chunks = []
        for rid in ids:
            z = mf.load_frame_npz(rid)
            v = z[group][coef - 1]
            chunks.append(v[masks[rid]] if masks is not None else v)
        out[a] = np.concatenate(chunks) if chunks else np.array([])
    return out


def violin_frames(ids_by_accent, speech_masks, coefs, accents, colors, name, title, compare=True):
    """Frame-level violins: light = all frames, dark = speech-only frames."""
    fig, axes = plt.subplots(1, len(coefs), figsize=(4.2 * len(coefs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, coefs):
        allf = pooled_frames(ids_by_accent, c)
        spf = pooled_frames(ids_by_accent, c, speech_masks)
        for i, a in enumerate(accents):
            sets = [(allf[a], -0.2 if compare else 0, 0.3), (spf[a], 0.2, 0.85)] if compare else [(allf[a], 0, 0.45)]
            for data, off, alpha in sets:
                if len(data) < 2:
                    continue
                vp = ax.violinplot([data], positions=[i + off], widths=0.36 if compare else 0.7,
                                   showextrema=False, showmedians=True)
                for b in vp["bodies"]:
                    b.set_facecolor(colors[a]); b.set_alpha(alpha); b.set_edgecolor("black"); b.set_linewidth(0.5)
                vp["cmedians"].set_color("black")
        ax.set_xticks(range(len(accents))); ax.set_xticklabels([a[:4] for a in accents], fontsize=8)
        ax.set_title(f"MFCC {c}"); ax.set_ylabel("Frame value")
    if compare:
        axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3, label="All frames (left violin)"),
                                plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.85, label="Speech frames only (right violin)")],
                       frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, y=1.02); fig.tight_layout(); save(fig, name)


def hist_frames(ids_by_accent, coefs, accents, colors, name, title):
    fig, axes = plt.subplots(1, len(coefs), figsize=(4.2 * len(coefs), 3.8))
    for ax, c in zip(np.atleast_1d(axes), coefs):
        data = pooled_frames(ids_by_accent, c)
        lo = min(np.percentile(d, 0.5) for d in data.values() if len(d))
        hi = max(np.percentile(d, 99.5) for d in data.values() if len(d))
        bins = np.linspace(lo, hi, 60)
        for a in accents:
            if len(data[a]):
                ax.hist(data[a], bins=bins, density=True, histtype="step", color=colors[a], lw=1.6, label=a)
        ax.set_title(f"MFCC {c}"); ax.set_xlabel("Frame value"); ax.set_ylabel("Density")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02); fig.tight_layout(); save(fig, name)


def pairwise_means(table, cols, accents, colors, markers, name, title):
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3.1 * n, 3.0 * n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                _strip(ax, table, ci, accents, colors, markers, box=False)
            else:
                for a in accents:
                    for fmt in markers:
                        s = table[(table.accent == a) & (table.original_format == fmt)]
                        ax.scatter(s[cj], s[ci], color=colors[a], marker=markers[fmt], s=26, edgecolor="black", linewidth=0.4)
            if i == n - 1: ax.set_xlabel(fa.feature_label(cj), fontsize=8)
            if j == 0: ax.set_ylabel(fa.feature_label(ci), fontsize=8)
    axes[0, n - 1].legend(handles=_legend_handles(accents, colors, sorted(markers), markers), frameon=False, fontsize=7,
                          loc="upper right")
    fig.suptitle(title, y=1.0); fig.tight_layout(); save(fig, name)


def coef_heatmap(ax, summary: pd.DataFrame, column: str, title: str, cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.2f}"):
    """13 coefficients (rows) x 6 feature types (columns) heatmap of one summary column."""
    cols = [(g, s) for g in mf.FEATURE_GROUPS for s in ("mean", "std")]
    mat = np.full((mf.N_COEF, len(cols)), np.nan)
    for j, (g, s) in enumerate(cols):
        sub = summary[(summary.feature_group == g) & (summary.statistic == s)].set_index("coefficient")
        for k in range(1, mf.N_COEF + 1):
            if k in sub.index:
                mat[k - 1, j] = sub.loc[k, column]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    for (r, c), v in np.ndenumerate(mat):
        if np.isfinite(v):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - vmin) / (vmax - vmin + 1e-12) < 0.55 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{mf.FEATURE_GROUPS[g].replace('MFCC', '').strip() or 'MFCC'}" + chr(10) + s for g, s in cols], fontsize=8)
    ax.set_yticks(range(mf.N_COEF)); ax.set_yticklabels([f"MFCC {k}" for k in range(1, mf.N_COEF + 1)], fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(False)
    return im


def correlation_heatmap(corr: pd.DataFrame, name, title):
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    for b in range(0, n, 26):
        ax.axhline(b - 0.5, color="black", lw=1); ax.axvline(b - 0.5, color="black", lw=1)
    ticks = list(range(6, n, 26)) + list(range(19, n, 26))
    ax.set_xticks(range(0, n, 2)); ax.set_xticklabels([corr.columns[i].replace("_", " ") for i in range(0, n, 2)],
                                                     rotation=90, fontsize=5)
    ax.set_yticks(range(0, n, 2)); ax.set_yticklabels([corr.index[i].replace("_", " ") for i in range(0, n, 2)], fontsize=5)
    fig.colorbar(im, ax=ax, label="Pearson correlation across recordings", fraction=0.04)
    ax.set_title(title); ax.grid(False)
    fig.tight_layout(); save(fig, name)


def quality_corr_heatmap(qcorr, features, feature_set, name, title):
    """features (rows) x quality metrics (columns); overall and within-accent side by side."""
    q = qcorr[qcorr.feature_set == feature_set]
    metrics = fa.QUALITY_COLUMNS
    fig, axes = plt.subplots(1, 2, figsize=(11, 15), sharey=True)
    for ax, col, sub in zip(axes, ["pearson_overall", "pearson_within_accent"],
                            ["All recordings pooled (speaker + format + noise differences included)",
                             "Within accent (each accent's own mean removed)"]):
        mat = np.array([[q[(q.feature == f) & (q.quality_metric == m)][col].iloc[0] for m in metrics] for f in features])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([fa.QUALITY_LABELS[m] for m in metrics], rotation=60, ha="right", fontsize=8)
        ax.set_title(sub, fontsize=9); ax.grid(False)
        for b in range(26, len(features), 26):
            ax.axhline(b - 0.5, color="black", lw=1)
    axes[0].set_yticks(range(len(features))); axes[0].set_yticklabels([fa.feature_label(f) for f in features], fontsize=5.5)
    fig.colorbar(im, ax=axes, label="Pearson correlation", fraction=0.025, pad=0.02)
    fig.suptitle(title, y=0.995); save(fig, name)
