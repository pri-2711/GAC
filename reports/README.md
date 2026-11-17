# Reports

Beginner-friendly visual analysis reports for each phase of the Gujarati Accent Classifier. Figures are loaded from the `results/` folders using relative paths.

| Phase | Report | Based on |
|---|---|---|
| 2 – Audio quality analysis | [audio_analysis_report.md](audio_analysis_report.md) | `notebooks/02_audio_analysis.ipynb`, `results/audio_analysis/` |
| 3 – MFCC extraction | [mfcc_extraction_report.md](mfcc_extraction_report.md) | `notebooks/03_mfcc_extraction.ipynb`, `results/mfcc/` |
| 4 – Feature analysis and comparison | [feature_analysis_report.md](feature_analysis_report.md) | `notebooks/04_feature_analysis.ipynb`, `results/feature_analysis/` |
| 5 – Dimensionality reduction (PCA) | [pca_report.md](pca_report.md) | `scripts/run_phase5.py`, `results/pca/` |
| 6 – Model development (dual-branch CNN–MLP, experiments, final model) | [model_development_report.md](model_development_report.md) | `scripts/run_phase6.py`, `results/model_development/`, `figures/phase6_hierarchical_model_architecture.png` |
| Maintenance - code audit and cleanup after Phase 4 | [project_cleanup_audit.md](project_cleanup_audit.md) | `src/`, `scripts/`, `README.md` |

**Surti update.** The Phase 2, 3 and 4 reports were updated after genuine Surti recordings were added: they now cover **five accents and 35 recordings**. The original four-accent findings are kept; anything new is marked **[NEW – Surti]** and anything that changed is marked **[CHANGED]**, and each report starts with a "What Changed After Adding Surti" section. The last four-accent versions are in the git history (commit `644befb`).
