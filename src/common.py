"""Helpers shared by more than one phase module (no phase-specific logic lives here)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

import config as cfg


def load_metadata() -> pd.DataFrame:
    """Phase 1 metadata (dataset/recordings_metadata.csv), successful conversions only."""
    meta = pd.read_csv(cfg.METADATA_CSV)
    return meta[meta["conversion_status"].astype(str).str.startswith("success")].reset_index(drop=True)


def load_quality_metrics(keep_empty_strings: bool = False) -> pd.DataFrame:
    """Phase 2 per-recording quality metrics.

    keep_empty_strings=True keeps an empty `quality_flags` as "" instead of NaN (Phase 3 stores
    the flag text in its feature table and relies on that).
    """
    return pd.read_csv(cfg.PHASE2_METRICS_CSV, keep_default_na=not keep_empty_strings)


class CheckList(list):
    """A list of (name, passed, detail) validation results with a text formatter.

    It *is* a list of tuples, so code that iterates or unpacks checks keeps working.
    """

    def add(self, name: str, ok, detail: str = "") -> None:
        self.append((name, bool(ok), detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self)

    def format_lines(self, indent: str = "  ", arrow: str = " -> ") -> list[str]:
        return [f"{indent}[{'PASS' if ok else 'FAIL'}] {name}" + (f"{arrow}{detail}" if detail else "")
                for name, ok, detail in self]


def hash_files(paths) -> dict[str, str]:
    """SHA-256 of every file under the given files/folders, keyed by project-relative path."""
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(q for q in root.rglob("*") if q.is_file())
        for p in files:
            out[p.relative_to(cfg.PROJECT_ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
