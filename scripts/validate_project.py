"""Run every validation the project defines, in one place (read-only; nothing is modified).

Usage:
    python scripts/validate_project.py            # all checks
    python scripts/validate_project.py --quick    # skip the slower per-file WAV decoding

Sections: imports, Phase 1 (processed audio), Phase 2, Phase 3 (feature arrays and schema),
Phase 4 (cross-phase consistency), notebooks, Markdown links and directory references,
requirements, and a dynamic-accent smoke test (in memory only: no data is created).
Exit status is 1 if any check fails.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import common

QUICK = "--quick" in sys.argv
results = common.CheckList()


def section(title: str) -> None:
    print(f"\n== {title} ==")


def record(name: str, ok, detail: str = "") -> None:
    results.add(name, ok, detail)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
section("1. Imports")
MODULES = ["config", "common", "plot_utils", "audio_utils", "audio_analysis",
           "mfcc_features", "feature_analysis", "feature_plots"]
for m in MODULES:
    try:
        importlib.import_module(m)
        record(f"import {m}", True)
    except Exception as exc:                       # noqa: BLE001 - report every failure
        record(f"import {m}", False, f"{type(exc).__name__}: {exc}")
if not all(ok for _, ok, _ in results):
    print("\nCannot continue: fix the imports first.")
    sys.exit(1)

import audio_utils as au
import config as cfg
import feature_analysis as fa
import mfcc_features as mf

# --------------------------------------------------------------------------- #
section("2. Phase 1 - dataset and standardised audio")
meta = common.load_metadata()
record("metadata CSV loads and has rows", len(meta) > 0, f"{len(meta)} recordings")
record("recording IDs are unique", meta.recording_id.is_unique)
record("every accent in the metadata is configured", set(meta.accent) <= set(cfg.configured_accents()),
       f"found: {sorted(set(meta.accent))}")
missing = [p for p in meta.processed_file_path if not (ROOT / p).is_file()]
record("every processed WAV in the metadata exists", not missing, f"missing: {missing}" if missing else "")
raw_missing = [p for p in meta.original_file_path if not (ROOT / p).is_file()]
record("every original recording in the metadata still exists", not raw_missing, f"missing: {raw_missing}" if raw_missing else "")
if not QUICK:
    bad = {}
    for rid, p in zip(meta.recording_id, meta.processed_file_path):
        problems = au.validate_wav(ROOT / p)["problems"]
        if problems:
            bad[rid] = problems
    record("every WAV is 16 kHz / mono / PCM_16 / readable / non-empty (audio_utils.validate_wav)", not bad, str(bad) if bad else f"{len(meta)} files")
counts = meta.groupby("accent").size().to_dict()
record("recordings per accent", True, ", ".join(f"{a}: {counts.get(a, 0)}" for a in cfg.ordered_accents(list(cfg.configured_accents()) + list(counts))))

# --------------------------------------------------------------------------- #
section("3. Phase 2 - audio quality metrics")
p2 = common.load_quality_metrics()
record("Phase 2 metrics cover every recording", set(meta.recording_id) <= set(p2.recording_id))
record("Phase 2 outputs present", all((cfg.AUDIO_ANALYSIS_DIR / f).is_file() for f in
       ["audio_quality_metrics.csv", "duration_summary_by_accent.csv", "quality_summary.txt"]))

# --------------------------------------------------------------------------- #
section("4. Phase 3 - MFCC arrays and feature table")
feats = pd.read_csv(mf.MFCC_DIR / "mfcc_recording_features.csv", keep_default_na=False)
store = {}
for rid in feats.recording_id:
    z = mf.load_frame_npz(rid)
    store[rid] = {"mfcc": z["mfcc"], "delta": z["delta"], "delta2": z["delta2"], "n_samples": int(z["n_samples"])}
_, missing_flags = mf.load_metadata_with_flags()
phase3_checks, ok3 = mf.validate(meta, feats, store, [], missing_flags)
for name, ok, detail in phase3_checks:
    record(f"[mfcc_features.validate] {name}", ok, detail)
record("mfcc_feature_columns.json matches mfcc_features", fa.load_schema()["features"] == mf.feature_column_names()
       and fa.load_schema()["params"] == mf.PARAMS)

# --------------------------------------------------------------------------- #
section("5. Phase 4 - cross-phase consistency and outputs")
data = fa.load_inputs()
for name, ok, detail in data["checks"]:
    record(f"[feature_analysis.load_inputs] {name}", ok, detail if not ok else "")
expected = ["feature_analysis_summary.csv", "feature_correlations.csv", "accent_feature_statistics.csv",
            "format_comparison.csv", "speech_only_recording_features.csv", "quality_correlations.csv",
            "validation_summary.txt"]
record("Phase 4 output files present", all((cfg.FEATURE_ANALYSIS_DIR / f).is_file() for f in expected))
summary_txt = (cfg.FEATURE_ANALYSIS_DIR / "validation_summary.txt").read_text(encoding="utf-8")
record("Phase 4 validation_summary.txt reports ALL CHECKS PASSED", "ALL CHECKS PASSED" in summary_txt)
fs = pd.read_csv(cfg.FEATURE_ANALYSIS_DIR / "feature_analysis_summary.csv")
record("feature_analysis_summary.csv has one row per schema feature, same order", list(fs.feature) == fa.load_schema()["features"])

# --------------------------------------------------------------------------- #
section("6. Notebooks (stored outputs)")
for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
    j = json.loads(nb.read_text(encoding="utf-8"))
    code = [c for c in j["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    unexecuted = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    record(f"{nb.name}: all code cells executed, no error outputs", not errors and not unexecuted,
           f"{len(code)} code cells; errors={len(errors)}; unexecuted={len(unexecuted)}")

# --------------------------------------------------------------------------- #
section("7. Markdown links and directory references")


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    return {slug(m) for m in re.findall(r"^#{1,6}\s+(.*)$", path.read_text(encoding="utf-8"), re.M)}


md_files = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
link_re = re.compile(r"\]\(([^)\s]+)\)")
n_links = 0
for f in sorted(md_files):
    text = f.read_text(encoding="utf-8")
    broken = []
    for target in link_re.findall(text):
        if re.match(r"^(https?:|mailto:)", target):
            continue
        n_links += 1
        path_part, _, anchor = target.partition("#")
        dest = f if not path_part else (f.parent / path_part).resolve()
        if not dest.exists():
            broken.append(target)
        elif anchor and dest.suffix == ".md" and anchor not in anchors_of(dest):
            broken.append(target)
    record(f"links in {f.relative_to(ROOT).as_posix()}", not broken, f"broken: {broken}" if broken else "")

PATH_TOKEN = re.compile(r"`((?:src|scripts|notebooks|results|reports|dataset)/[^`\s]*)`")
for f in (ROOT / "README.md", ROOT / "reports" / "README.md"):
    if not f.is_file():
        continue
    tokens = set(PATH_TOKEN.findall(f.read_text(encoding="utf-8")))
    missing = [t for t in sorted(tokens)
               if not any(ch in t for ch in "<>*{}") and "Surti" not in t and "..." not in t and not (ROOT / t.rstrip("/")).exists()]
    record(f"directory/file references in {f.relative_to(ROOT).as_posix()}", not missing,
           f"{len(tokens)} references checked; missing: {missing}" if missing else f"{len(tokens)} references checked")

# --------------------------------------------------------------------------- #
section("8. Requirements")
req = ROOT / "requirements.txt"
if req.is_file():
    mismatched = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, version = line.partition("==")
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            mismatched.append(f"{name} not installed")
            continue
        if version and installed != version:
            mismatched.append(f"{name}: pinned {version}, installed {installed}")
    record("installed packages match requirements.txt", not mismatched, "; ".join(mismatched))
else:
    record("requirements.txt exists", False)

# --------------------------------------------------------------------------- #
section("9. Dynamic-accent smoke test (in memory only; no files or data are created)")
original_accents = cfg.ACCENTS
try:
    record("ordered_accents keeps configured order and appends unknown accents",
           cfg.ordered_accents(["Zzz", "Kutchi", "Ahmedabad"]) == ["Ahmedabad", "Kutchi", "Zzz"])
    cfg.ACCENTS = original_accents + (cfg.AccentConfig("TestAccent", "Nowhere", "TST_S01", "TST", ("testaccent",)),)
    recs, unmapped = au.discover_recordings()
    summ = au.summarise([], [])
    record("an extra configured accent with no recordings is reported as 0 and breaks nothing",
           summ["per_accent_valid"].get("TestAccent") == 0 and len(recs) == len(meta), f"{len(recs)} recordings discovered")
    record("no folder was created for the empty test accent", not (cfg.PROCESSED_WAV_DIR / "TestAccent").exists())
finally:
    cfg.ACCENTS = original_accents

T = data["table"]
one = T[T.accent == data["accents"][0]]
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.accent_feature_statistics(one, data["features"], [data["accents"][0]])
        fa.format_comparison(one, one, data["features"])
    record("feature statistics run for a single-accent subset", True)
except Exception as exc:                           # noqa: BLE001
    record("feature statistics run for a single-accent subset", False, str(exc))

# --------------------------------------------------------------------------- #
failed = [name for name, ok, _ in results if not ok]
print("\n" + "=" * 60)
print(f"{len(results) - len(failed)} passed, {len(failed)} failed, {n_links} Markdown links checked")
if failed:
    print("FAILED:", *failed, sep="\n  - ")
    sys.exit(1)
print("ALL VALIDATION CHECKS PASSED")
