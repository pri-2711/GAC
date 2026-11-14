"""Plotting helpers shared by the phases that draw accent-coloured figures.

(Notebooks 02 and 03 predate this module and keep their own inline styling so that their
validated figures stay byte-for-byte reproducible; new work should import from here.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import config as cfg

# Colour-blind-safe (Okabe-Ito) colour per configured accent; the same colours as Phases 2-3.
ACCENT_PALETTE = {"Ahmedabad": "#0072B2", "Kutchi": "#E69F00", "Mehsani": "#009E73",
                  "Kathiyawadi": "#CC79A7", "Surti": "#D55E00"}
FALLBACK_COLORS = ["#8E44AD", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

STYLE = {"figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
         "font.size": 10, "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.2}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def accent_colors(accents) -> dict[str, str]:
    """One colour per accent; accents without a preset get a fallback colour."""
    colors, extra = {}, iter(FALLBACK_COLORS)
    for a in accents:
        colors[a] = ACCENT_PALETTE.get(a) or next(extra, "#555555")
    return colors


def format_markers(formats) -> dict[str, str]:
    """One marker per original audio format (e.g. M4A -> circle, OGG -> square)."""
    return {f: m for f, m in zip(sorted(set(formats)), MARKERS)}


def legend_handles(accents, colors, formats, markers):
    """Legend entries for accent colours and format markers."""
    h = [plt.Line2D([], [], marker="o", ls="", color=colors[a], label=a) for a in accents]
    h += [plt.Line2D([], [], marker=markers[f], ls="", color="grey", mfc="none", label=f"{f} file") for f in formats]
    return h


def save_figure(fig, path: Path, show: bool = True) -> Path:
    """Save at the project's standard quality, print the project-relative path, then show/close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print("saved:", path.relative_to(cfg.PROJECT_ROOT).as_posix())
    if show:
        plt.show()
    plt.close(fig)
    return path
