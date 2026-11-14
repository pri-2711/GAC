# Project Cleanup Audit

*Gujarati Accent Classifier – project-wide maintenance after Phase 4 (Phase 5 not started)*

This document records (A) what was examined and found **before** any change, (B) the plan, (C) what was changed, and (D) the validation results. Parts A and B were written before editing code; Parts C and D were completed afterwards.

---

## A. Audit (written before making changes)

### A.1 Starting state

- Git repository on `main`, **clean working tree**, last commit `e9f91d9` (Phase 4). All four phases are committed, so `git diff` shows exactly what this cleanup changes.
- 213 tracked files: 58 under `dataset/`, 132 under `results/`, 12 under `src/` (6 source files + **6 tracked `__pycache__/*.pyc` files**), 4 notebooks, 4 reports, 1 script, plus the two original project presentation files (`NLP_researchPaper_project.pdf/.pptx`).
- **No** `README.md`, `requirements.txt` or `.gitignore` at the repository root.
- All four notebooks have been executed and contain **0 error outputs**.

### A.2 Files examined

| Area | Files |
|---|---|
| `src/` | `config.py` (46 lines), `audio_utils.py` (336), `audio_analysis.py` (152), `mfcc_features.py` (244), `feature_analysis.py` (462), `feature_plots.py` (199) |
| `scripts/` | `run_phase1.py` (13) |
| `notebooks/` | `01_audio_standardization`, `02_audio_analysis`, `03_mfcc_extraction`, `04_feature_analysis` (all code cells read) |
| `dataset/` | raw folders, `processed/wav/`, `processed/reports/`, `recordings_metadata.csv` |
| `results/` | `audio_analysis/`, `mfcc/` (+ `frame_features/`, `figures/`), `feature_analysis/` (+ `figures/`) |
| `reports/` | `audio_analysis_report.md`, `mfcc_extraction_report.md`, `feature_analysis_report.md`, `README.md` |

Method: full read of every module; an AST scan for unused imports, unused locals and top-level functions/constants never referenced anywhere (source or notebooks); `grep` for repeated patterns (CSV loading, check lists, palettes, styling, `save` helpers, hard-coded accent names).

### A.3 Redundancies found

**Confirmed duplicates (same task, safe to consolidate):**

| # | Where | What is repeated | Verified same task? |
|---|---|---|---|
| R1 | `mfcc_features.load_metadata_with_flags`, `feature_analysis.load_inputs`, `feature_analysis.compute_speech_frames` | Reading `recordings_metadata.csv`, keeping only `success*` rows; reading Phase 2 `audio_quality_metrics.csv` (twice in `feature_analysis`) | Yes for the metadata. The Phase 2 CSV is read with `keep_default_na=False` in one place and default NA handling in the others, so the shared loader keeps that as an option (the difference matters: empty flag strings vs NaN). |
| R2 | `mfcc_features.validate`, `feature_analysis.load_inputs` | The identical `checks = []` / `add = lambda name, ok, detail="": checks.append((name, bool(ok), detail))` container, plus separately formatted `[PASS]/[FAIL]` text in `mfcc_features.format_validation_summary` and in notebook 04 | Yes (container and text format). The *checks themselves* are different and stay where they are. |
| R3 | `feature_analysis.ordered_accents/configured_accents` vs. `mfcc_features.accent_summary`, `audio_analysis.representative_recordings` | Accent ordering logic. `feature_analysis` handles accents that are not in the config; the two others loop over `cfg.ACCENTS` only and would silently skip an unconfigured accent | Same task, different robustness. Belongs in `config.py`. |
| R4 | `feature_analysis._BASE_COLORS`, `COLORS` dicts in notebooks 02 and 03 | Per-accent palette. Same four colours, but **Surti is grey (`#999999`) in notebooks 02/03 and vermilion (`#D55E00`) in Phase 4** (inconsistent) | Yes (inconsistency confirmed) |
| R5 | `feature_plots.save`, `save()` in notebooks 02, 03, 04; `feature_plots` `rcParams` block vs. notebooks 02/03 | Figure saving and plot style | Similar but **not identical** (different output folders, `axes.grid` and `grid.alpha` differ between notebooks). See "retained". |
| R6 | Results folders | Paths such as `results/audio_analysis`, `results/mfcc`, `results/feature_analysis` are each built inside a different phase module (`aa.RESULTS_DIR`, `mf.MFCC_DIR`, `fa.FA_DIR`) rather than in `config.py` | Yes. One source of truth is better. |
| R7 | `fp._legend_handles` | A private (underscore) helper imported and used by notebook 04 | Naming issue only |

**Dead code (tiny):** `feature_analysis.py` imports `Path` but never uses it; `feature_plots.correlation_heatmap` computes an unused `ticks` list; `mfcc_features.validate` creates an unused `ok_frames` flag. The AST scan found **no** unreferenced top-level functions or constants.

**Repository hygiene:**
- `src/__pycache__/*.pyc` (6 files) are **tracked in git** and there is no `.gitignore`.
- No `requirements.txt` (dependencies are currently only implied by imports).
- Duplicated generated documents: `results/audio_analysis/audio_analysis_report.md` and `results/mfcc/README.md` have copies in `reports/`. (Retained: see below.)

### A.4 Code that looks similar but is genuinely different (kept separate)

| Code | Why it stays separate |
|---|---|
| `audio_utils.validate_wav` (SoundFile) vs. `audio_analysis.load_audio` (Librosa) | Different jobs. `validate_wav` fully decodes with SoundFile to check the **file header and encoding** (Phase 1 contract); `load_audio` loads a float signal **unchanged at the native rate** for analysis. Only one audio-loading implementation exists for analysis (`load_audio`), and Phases 3 and 4 already reuse it. |
| `audio_utils.validate_wav`, `mfcc_features.validate`, `feature_analysis.load_inputs` checks | They validate different objects (audio files, feature arrays, cross-phase consistency). Only the check container and text formatting are shared. |
| `mfcc_features.accent_summary` vs. `feature_analysis.accent_feature_statistics` | Different statistics and schemas (Phase 3: recording-level means plus pooled frame statistics in long format; Phase 4: n/mean/std/min/median/max of recording-level values). Both CSVs are validated deliverables referenced by reports. |
| `audio_analysis.speech_activity` vs. `feature_analysis.compute_speech_frames` | The second *calls* the first and adds the alignment checks. No duplication. |
| Phase modules (`audio_utils`, `audio_analysis`, `mfcc_features`, `feature_analysis`) | Each owns one phase's logic and is imported by the notebooks under those names. Merging them would break notebooks and create one very large module. |
| `PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/"dataset").is_dir())` in every notebook | A deliberate bootstrap: it runs **before** `src/` is on the import path, so it cannot be moved into `src/`. |
| Plot-style and `save()` boilerplate inside notebooks 02 and 03 | Left unchanged on purpose: those notebooks produced validated figures. Re-writing them risks changing published PNGs for little gain. Recorded as technical debt. |
| `reports/` copies of the Phase 2/3 reports | You asked that the original phase reports not be deleted. Both copies are kept (different image paths). `reports/` is the canonical location. |

### A.5 Proposed changes

1. **New `src/common.py`** (shared, phase-independent): metadata/quality loaders (R1), `CheckList` container and formatter (R2), file hashing helper.
2. **New `src/plot_utils.py`**: one palette incl. a Surti colour, marker map, plot style, `save_figure`, `legend_handles` (R4, R5 for new code, R7).
3. **`config.py`**: add `ordered_accents()` and `configured_accents()` (R3) and central results-folder constants (R6). Existing names stay valid (re-exported / aliased).
4. Update `mfcc_features`, `feature_analysis`, `feature_plots`, `audio_analysis` to use these; delete dead code. Keep **every public name** that notebooks use.
5. Update notebook 04's builder to drop its own duplicates; re-run it.
6. Add `scripts/validate_project.py` (runs every existing validation in one place, plus a link checker, an import check and a dynamic-accent smoke test) and `scripts/run_notebooks.py` (headless execution of the notebooks).
7. Add `requirements.txt`, `.gitignore`, root `README.md`; remove `__pycache__` folders.
8. Do **not** touch: raw or processed audio, any CSV/PNG/NPZ result, the Phase 2–4 reports, notebooks 01–03.

---

## C. Changes made

### C.1 New files

| File | Purpose |
|---|---|
| `src/common.py` | Shared helpers used by more than one phase: `load_metadata()`, `load_quality_metrics()`, the `CheckList` validation container (it *is* a list of `(name, passed, detail)` tuples, so existing code keeps working) and `hash_files()`. Resolves R1 and R2. |
| `src/plot_utils.py` | One accent palette (now including a Surti colour), format markers, plot style, `legend_handles()` and `save_figure()`. Resolves R4, R7 and the Phase 4 side of R5. |
| `scripts/validate_project.py` | Runs every existing validation in one place (imports, Phase 1 audio check, Phase 2 and 3 outputs, Phase 3 `validate()`, Phase 4 cross-phase checks, stored notebook outputs, Markdown links and directory references, `requirements.txt`, and a dynamic-accent smoke test that creates no data). Exit status 1 on any failure. |
| `scripts/run_notebooks.py` | Executes the phase notebooks headlessly (`--only`, `--no-save`). |
| `README.md`, `requirements.txt`, `.gitignore` | Project documentation, pinned dependencies (exactly the versions the project was validated with), and ignore rules for caches. |

### C.2 Modified files

| File | Change |
|---|---|
| `src/config.py` | Added the results-folder constants (`RESULTS_DIR`, `AUDIO_ANALYSIS_DIR`, `MFCC_DIR`, `FEATURE_ANALYSIS_DIR`, `PHASE2_METRICS_CSV`, `PROJECT_REPORTS_DIR`) and the accent helpers `configured_accents()` / `ordered_accents()` (R3, R6). No existing name or value changed. |
| `src/audio_analysis.py` | `RESULTS_DIR` now points at the config constant (same value); `representative_recordings` uses `ordered_accents` so an accent outside the config list cannot be skipped silently. |
| `src/mfcc_features.py` | Uses the shared loaders and `CheckList`; `accent_summary` uses `ordered_accents`; removed the unused `ok_frames`. Output text keeps its original layout. |
| `src/feature_analysis.py` | Uses the shared loaders, `CheckList` and `hash_files`; accent/colour/marker helpers are now re-exports from `config` / `plot_utils` (so `fa.ordered_accents`, `fa.accent_colors`, ... still work); removed the unused `Path` import and the duplicated palette. |
| `src/feature_plots.py` | Uses `plot_utils` for style, saving and legend handles; `_legend_handles` kept as an alias of the new public `legend_handles`; removed an unused local. |
| `notebooks/04_feature_analysis.ipynb` | Rebuilt and re-executed: uses `fp.` instead of a second import of the same module, `common.CheckList` for its validation summary, and no unused variable. All results identical (see D.2). |
| `reports/mfcc_extraction_report.md`, `results/mfcc/README.md` | Corrected a factual error: "14 summary figures" is **13** (the `per_recording/` folder had been counted). |
| `reports/README.md` | Added this audit to the index (the Phase 4 report was already linked). |

### C.3 Removed

- `src/__pycache__/*.pyc` (6 files that were tracked in git). They are generated files; `.gitignore` now excludes them. In `git status` they appear as deleted.

### C.4 One bug found and fixed during the cleanup

The first run of `scripts/validate_project.py` failed with `UnboundLocalError` in `feature_analysis.load_inputs`: an existing local variable named `common` shadowed the new `common` module. It was renamed (`shared_ids`), and a scan of all functions for locals that shadow imported modules found no others. This is a good example of why the validation script was added.

### C.5 Intentionally retained (and why)

| Item | Reason |
|---|---|
| Notebooks 01, 02, 03 (code unchanged) | Backward compatibility, and their inline colour/style/`save()` boilerplate produced validated figures. They were **executed after the refactor** (D.2) and still work. |
| `PROJECT_ROOT = next(...)` bootstrap in each notebook | Runs before `src/` is importable, so it cannot move into `src/`. |
| Separate phase modules (`audio_utils`, `audio_analysis`, `mfcc_features`, `feature_analysis`) | They own genuinely different logic and are imported by name from the notebooks. |
| `audio_utils.validate_wav` and `audio_analysis.load_audio` | Different tasks (header/encoding validation vs. loading a float signal at the native rate). |
| `mfcc_features.accent_summary` and `feature_analysis.accent_feature_statistics` | Different statistics and schemas; both CSVs are validated deliverables. |
| `results/audio_analysis/audio_analysis_report.md`, `results/mfcc/README.md` | Original phase reports (you asked that they not be deleted). `reports/` holds the canonical copies. |
| `NLP_researchPaper_project.pdf/.pptx` | The project presentation; not part of the code or results. |
| `results/audio_analysis/image.png` | See E.1. |

---

## D. Validation results

### D.1 `python scripts/validate_project.py`

**67 checks passed, 0 failed, 217 Markdown links checked.** It covers:

- all 8 modules import;
- all 28 processed WAV files are 16 kHz, mono, PCM_16, readable and non-empty (`audio_utils.validate_wav`);
- all 28 original recordings still exist; IDs unique; every accent in the metadata is configured;
- Phase 3: the 11 checks of `mfcc_features.validate` pass on the stored NPZ arrays, and the feature-column file matches the code;
- Phase 4: the 16 cross-phase consistency checks pass, and its saved validation summary reports all checks passed;
- all 4 notebooks have every code cell executed and no error outputs;
- every relative link and anchor in all Markdown files, and every directory/file reference in `README.md` and `reports/README.md`;
- installed packages match `requirements.txt`;
- dynamic-accent smoke test: an extra configured accent with no recordings is reported as 0 and creates no folder; a single-accent subset runs.

### D.2 Notebooks re-executed after the refactor

All four notebooks were executed headlessly with `scripts/run_notebooks.py` (Phase 1: 9 s, Phase 2: 42 s, Phase 3: 37 s, Phase 4: 217 s). No errors. Phase 4's own 26 validation checks passed, including "Phase 1-3 outputs unchanged".

All 133 regenerated files under `results/` plus `dataset/recordings_metadata.csv` were compared with a backup taken before any change: **133 byte-identical, 0 different, 0 missing, 0 new**. The raw recordings and processed WAV files were fingerprinted before and after: **0 changed**.

### D.3 Git status after cleanup

| Status | Files |
|---|---|
| Modified (9) | `notebooks/04_feature_analysis.ipynb`, `reports/README.md`, `reports/mfcc_extraction_report.md`, `results/mfcc/README.md`, `src/audio_analysis.py`, `src/config.py`, `src/feature_analysis.py`, `src/feature_plots.py`, `src/mfcc_features.py` |
| Added (8) | `.gitignore`, `README.md`, `requirements.txt`, `reports/project_cleanup_audit.md`, `scripts/run_notebooks.py`, `scripts/validate_project.py`, `src/common.py`, `src/plot_utils.py` |
| Deleted (6) | `src/__pycache__/*.pyc` |

No file under `dataset/` and no data file under `results/` changed. Nothing was committed.

---

## E. Remaining technical debt and open items

1. **`results/audio_analysis/image.png`** is a tracked screenshot of an earlier chat's Tech Stack table. No code, notebook or report references it (the only search hits were base64 image data inside notebook outputs). It was **not** created by the pipeline, so I left it alone. It can be removed with `git rm results/audio_analysis/image.png` if you do not need it.
2. **Notebooks 02 and 03** still define their own colour dictionary, plot style and `save()` helper (the Surti colour there is grey, whereas `plot_utils` uses vermilion; Surti is not plotted yet, so no figure is affected). Migrating them to `plot_utils` would be a small change but would regenerate validated figures, so it was deferred.
3. **Duplicate report copies** in `results/` and `reports/` can drift apart. Consider keeping only the `reports/` versions once you are comfortable removing the originals.
4. **Modules are imported by bare name** (the notebooks and scripts add `src/` to `sys.path`). A `pyproject.toml` and an installable package would be cleaner when the project grows.
5. **Adding a new accent still needs one edit** in `src/config.py` (`ACCENTS`) so that its folder name, speaker ID and region are known. Everything after that (discovery, features, analysis, plots) picks it up from the data. Surti is already configured.
6. **The written reports contain numbers from the four-class dataset** and must be refreshed after Surti is added.
7. **No automated unit tests** beyond the validations. `scripts/validate_project.py` is the project's test entry point; real unit tests for the statistics (eta-squared, correlations) would be a good addition before Phase 6.
8. The notebooks' stored outputs are large (notebook 02 is about 3.5 MB, notebook 04 about 4 MB). Clearing outputs before commits would reduce the repository size if that becomes a concern.
9. Git reports a line-ending warning for `src/feature_plots.py` (LF will become CRLF). A `.gitattributes` file would make line endings consistent across machines.
