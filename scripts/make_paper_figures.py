"""Draw the explanatory diagrams used in reports/research_paper.md (saved to reports/figures/paper_fig_*.png).

Only schematic diagrams are produced here; every data figure in the paper is an existing project result.
Numbers printed inside the diagrams (e.g. 78 features, 64.4% of variance) come from the project's saved results and are
checked by scripts/validate_project.py through the paper's own statements.

Usage:  python scripts/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures"
C = {"data": "#E8EEF7", "proc": "#FFF2CC", "analysis": "#D9EAD3", "model": "#CFE2F3", "out": "#FCE5CD", "head": "#EAD1F5", "warn": "#F4CCCC"}
INK = "#1F2937"


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9", fc=color, ec="#374151", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3, color=INK, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=lw, color="#374151", shrinkA=0, shrinkB=0))


def note(ax, x, y, text, fs=8.3):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs, color="#4B5563", style="italic", linespacing=1.25)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig_workflow():
    fig, ax = canvas(15, 6.2, (0, 100), (0, 46))
    steps = [("Dataset\nCollection", "35 recordings\n5 accents · 5 speakers", "data"), ("Audio\nPreprocessing", "FFmpeg → 16 kHz\nmono · PCM-16 WAV", "proc"),
             ("Audio Quality\nAnalysis", "levels · clipping\nnoise floor · speech activity", "analysis"), ("Feature\nExtraction", "13 MFCC + Δ + ΔΔ\n→ 78 features", "proc"),
             ("Feature\nAnalysis", "η² · overlap · correlation\nformat and noise effects", "analysis"), ("PCA", "PC1–PC4 hold\n64.4% of the variance", "analysis"),
             ("Model\nDevelopment", "SVM · MLP · CNN\nCNN–MLP fusion", "model"), ("Model\nEvaluation", "7-fold cross-validation\n(recording-level)", "model"),
             ("Final Model\nSelection", "MLP with region and\naccent heads", "out"), ("Prediction\nApplication", "Streamlit app:\nupload → region + accent", "out")]
    w, h, gap = 15.2, 9.5, 4.6
    pos = []
    for i, (t, n, c) in enumerate(steps):
        row, col = divmod(i, 5)
        x = 2 + col * (w + gap)
        y = 32 if row == 0 else 8
        box(ax, x, y, w, h, f"{i + 1}. {t}", {"data": C["data"], "proc": C["proc"], "analysis": C["analysis"], "model": C["model"], "out": C["out"]}[c], 9.6, True)
        note(ax, x + w / 2, y - 1.0, n)
        pos.append((x, y))
    for i in range(9):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i == 4:                                              # wrap to the second row
            ax.plot([x1 + w / 2, x1 + w / 2], [y1 - 5.2, 24.5], color="#374151", lw=1.5)
            ax.plot([x1 + w / 2, 2 + w / 2], [24.5, 24.5], color="#374151", lw=1.5)
            arrow(ax, (2 + w / 2, 24.5), (2 + w / 2, y2 + h + 0.4))
        else:
            arrow(ax, (x1 + w + 0.3, y1 + h / 2), (x2 - 0.3, y2 + h / 2))
    ax.text(50, 45.2, "Research workflow of the Gujarati Accent Classifier", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    save(fig, "paper_fig_workflow.png")


def fig_preprocessing():
    fig, ax = canvas(15, 3.9, (0, 100), (0, 26))
    steps = [("Original recording\nM4A or OGG\n16, 44.1 or 48 kHz", "data"), ("Decode\n(FFmpeg)", "proc"), ("Resample to\n16,000 Hz", "proc"), ("Convert to\nmono", "proc"),
             ("Write 16-bit\nPCM WAV", "proc"), ("Validate\n(SoundFile: rate,\nchannels, subtype)", "analysis"), ("Metadata table\n(ID, accent, region,\nformat, duration)", "out")]
    w, h, gap = 11.6, 10.5, 3.2
    for i, (t, c) in enumerate(steps):
        x = 1.5 + i * (w + gap)
        box(ax, x, 9, w, h, t, C[c], 9)
        if i:
            arrow(ax, (x - gap + 0.2, 9 + h / 2), (x - 0.2, 9 + h / 2))
    ax.text(50, 24.6, "Audio preprocessing pipeline (Phase 1)", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(50, 5.3, "The original files are never modified. No filtering, trimming, normalisation or noise reduction is applied: only format conversion.",
            ha="center", fontsize=9.3, color="#4B5563", style="italic")
    save(fig, "paper_fig_preprocessing.png")


def fig_mfcc():
    fig, ax = canvas(15, 6.0, (0, 100), (0, 40))
    row1 = [("Speech\nwaveform\n(16 kHz)", "data"), ("Cut into frames\n25 ms window\n(400 samples), step\n10 ms (160 samples)", "proc"), ("Hann window\n+ FFT (size 512)\n→ power spectrum", "proc"),
            ("Mel filter bank\n(128 bands)\nmimics hearing", "proc"), ("Take the\nlogarithm", "proc")]
    row2 = [("DCT (type 2)\nkeep 13 coefficients\n= MFCC 1–13", "analysis"), ("Delta (Δ)\nchange over time\n(window of 9 frames)", "analysis"), ("Delta-delta (ΔΔ)\nchange of the change", "analysis"),
            ("Mean and standard\ndeviation of each\ncoefficient over the\nwhole recording", "model"), ("13 × 3 × 2\n= 78 numbers\nper recording", "out")]
    w, h, gap = 15.2, 11, 4.6
    for r, (row, y) in enumerate(((row1, 24), (row2, 5))):
        for i, (t, c) in enumerate(row):
            x = 2 + i * (w + gap)
            box(ax, x, y, w, h, t, C[c], 9)
            if i:
                arrow(ax, (x - gap + 0.2, y + h / 2), (x - 0.2, y + h / 2))
    xe = 2 + 4 * (w + gap) + w / 2
    ax.plot([xe, xe], [24 - 0.3, 20.5], color="#374151", lw=1.5)
    ax.plot([xe, 2 + w / 2], [20.5, 20.5], color="#374151", lw=1.5)
    arrow(ax, (2 + w / 2, 20.5), (2 + w / 2, 5 + h + 0.3))
    ax.text(50, 39, "From a recording to 78 MFCC-based features (Phase 3)", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    save(fig, "paper_fig_mfcc_pipeline.png")


def fig_pca():
    fig, ax = canvas(15, 4.9, (0, 100), (0, 33))
    box(ax, 1, 12, 15, 10, "Feature matrix\n35 recordings × 78\nfeatures", C["data"], 9.3)
    box(ax, 21, 12, 15, 10, "Standardise\neach feature\n(mean 0, std 1)", C["proc"], 9.3)
    box(ax, 41, 12, 15, 10, "Singular value\ndecomposition\n(PCA)", C["model"], 9.3)
    for x1, x2 in ((16.3, 20.7), (36.3, 40.7)):
        arrow(ax, (x1, 17), (x2, 17))
    outs = [("Scores\n35 × 34", 25.5), ("Loadings\n78 × 34", 17), ("Explained variance\nper component", 8.5)]
    for t, y in outs:
        box(ax, 62, y - 3.3, 15, 6.6, t, C["analysis"], 9)
        arrow(ax, (56.3, 17), (61.7, y))
    box(ax, 83, 7, 15.5, 20, "Interpretation\nscree and cumulative\nvariance · scatter plots ·\nloading bars · η² with\naccent, format, quality", C["out"], 8.8)
    for _, y in outs:
        arrow(ax, (77.3, y), (82.7, 17 if y == 17 else (21 if y > 17 else 13)))
    ax.text(50, 32.2, "Principal component analysis workflow (Phase 5)", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(50, 3.2, "Sensitivity checks (kept separate): speech-only features (30 dB and 20 dB) · without MFCC 1 and 2 · each accent left out in turn · shuffled-data chance baseline",
            ha="center", fontsize=9, color="#4B5563", style="italic")
    save(fig, "paper_fig_pca_workflow.png")


def fig_mlp():
    fig, ax = canvas(15, 5.6, (0, 100), (0, 36))
    seq = [("Input\n78 features\n(z-scored,\ntraining stats)", "data", "0"), ("Dense 64\nReLU", "model", "5,056"), ("Batch\nNorm", "model", "128"), ("Dropout\n0.30", "warn", "0"),
           ("Dense 32\nReLU", "model", "2,080"), ("Dropout\n0.20", "warn", "0"), ("Shared\nDense 64\nReLU", "model", "2,112"), ("Dropout\n0.40", "warn", "0")]
    w, h, gap = 8.6, 11, 2.6
    for i, (t, c, p) in enumerate(seq):
        x = 1 + i * (w + gap)
        box(ax, x, 14, w, h, t, C[c], 8.4)
        note(ax, x + w / 2, 12.8, f"{p} params" if p != "0" else "", 7.8)
        if i:
            arrow(ax, (x - gap + 0.1, 19.5), (x - 0.1, 19.5))
    x0 = 1 + 7 * (w + gap) + w
    box(ax, 91, 22.5, 8.5, 8, "Region head\nDense 5\nsoftmax", C["head"], 8.4)
    box(ax, 91, 8, 8.5, 8, "Accent head\nDense 5\nsoftmax", C["head"], 8.4)
    arrow(ax, (x0 + 0.2, 21), (90.8, 26.5))
    arrow(ax, (x0 + 0.2, 18), (90.8, 12))
    note(ax, 95.2, 7.3, "325 params", 7.8)
    note(ax, 95.2, 21.8, "325 params", 7.8)
    ax.text(50, 35, "Architecture of the selected final model: MLP with hierarchical region and accent heads", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(50, 4.6, "10,026 trainable parameters (plus 128 BatchNorm statistics). Hierarchical decision: choose the region, then the most likely accent among the accents of that region.",
            ha="center", fontsize=9, color="#4B5563", style="italic")
    ax.text(50, 1.6, "Trained from scratch: random initialisation, no pretrained weights or embeddings.", ha="center", fontsize=9, color="#4B5563", style="italic")
    save(fig, "paper_fig_mlp_architecture.png")


def fig_app():
    fig, ax = canvas(15, 6.0, (0, 100), (0, 40))
    row1 = [("Audio upload\nWAV · MP3 · M4A · OGG", "data"), ("Audio validation\ntype · empty file ·\ndecodable · 1–180 s ·\nnot silent", "warn"), ("Standardisation\nFFmpeg → 16 kHz mono", "proc"),
            ("Feature extraction\nMFCC, Δ, ΔΔ → 78 numbers", "proc"), ("Saved normalisation\ntraining-set mean and std", "proc")]
    row2 = [("Trained MLP\n(final model)", "model"), ("Region and accent\nheads (softmax)", "head"), ("Hierarchical decoding\nregion first, then accent\nwithin that region", "analysis"),
            ("Results display\naccent · region · output\nscores · waveform", "out"), ("Always shown:\nexperimental prototype,\nnot validated on unseen\nspeakers", "warn")]
    w, h, gap = 15.2, 11, 4.6
    for row, y in ((row1, 24), (row2, 5)):
        for i, (t, c) in enumerate(row):
            x = 2 + i * (w + gap)
            box(ax, x, y, w, h, t, C[c], 8.8)
            if i:
                arrow(ax, (x - gap + 0.2, y + h / 2), (x - 0.2, y + h / 2))
    xe = 2 + 4 * (w + gap) + w / 2
    ax.plot([xe, xe], [23.7, 20.5], color="#374151", lw=1.5)
    ax.plot([xe, 2 + w / 2], [20.5, 20.5], color="#374151", lw=1.5)
    arrow(ax, (2 + w / 2, 20.5), (2 + w / 2, 5 + h + 0.3))
    ax.text(50, 39, "End-to-end prediction pipeline of the web application (Phase 7)", ha="center", fontsize=12.5, fontweight="bold", color=INK)
    save(fig, "paper_fig_app_pipeline.png")


if __name__ == "__main__":
    for f in (fig_workflow, fig_preprocessing, fig_mfcc, fig_pca, fig_mlp, fig_app):
        f()
