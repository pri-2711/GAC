# Research Paper: Source and Reference Audit

This note records how `reports/research_paper.md` was checked. The paper is built by `python scripts/build_paper.py` from the prose in `scripts/paper_template.md`; every number is read from the saved project results (the script stops with an error if a statistic or reference is missing).

## 1. The literature review

**No existing literature review or reference list was found in the project.** The repository contains only the Phase 1 topic presentation (`NLP_researchPaper_project.pdf` and `.pptx`: introduction, motivation, problem statement, research gap, objectives, planned methods), which has no citations; the reports, notebooks and README contain none; and a filename and keyword search of the parent folder's Word documents and PDF found none. On the project owner's instruction, Section 2 was therefore written from references **found and checked by web search during this task**. They are *new* references, not a preserved review, and should be checked by the authors. The Phase 1 presentation was used for the project's own motivation, objectives and planned methods (Sections 1.3 and 1.4), without citation.

## 2. References

18 references, IEEE numeric style (by order of first citation). Every in-text number has an entry and every entry is cited (checked by `scripts/validate_project.py`). No reference was invented; where a detail could not be confirmed it is flagged below.

| # | Reference | Times cited | Verification notes |
|---|---|---|---|
| 1 | G. Cardona and B. Suthar, "Gujarati," in *The Indo-Aryan Languages*, G. Cardona and D. Jain, Eds. London, U.K.: Routledge, 2003, pp. 659–697. | 2 | Existence, editors, publisher and page range confirmed. Chapter content not read; cited only for the general statement that standard references describe Gujarati. |
| 2 | R. Mesthrie and V. Chavda, "Cape Town Gujarati and its relation to Gujarati dialectology: A study of retroflex boosting," *Journal of South Asian Languages and Linguistics*, vol. 7, no. 1, pp. 1–14, 2021, doi: 10.1515/jsall-2020-2022. | 2 | Title, authors, journal, volume/pages and DOI confirmed. **Year differs between sources** (catalogue 2021, publisher page copyright 2020); 2021 used. Only abstract-level content (Cape Town Gujarati, retroflex boosting, relation to Gujarati dialectology) was used. |
| 3 | M. M. Shah and H. R. Kavathiya, "Voice recognition for Gujarati dialects: An in-depth survey," *International Journal of Computer Applications*, vol. 186, no. 5, pp. 1–4, Jan. 2024, doi: 10.5120/ijca2024923112. | 1 | Title, authors, journal, volume/issue/pages, month and DOI confirmed from the journal listing. Full text not read; cited only as a survey of Gujarati voice recognition that discusses problems of such systems. |
| 4 | A. Chittora and H. A. Patil, "Classification of phonemes using modulation spectrogram based features for Gujarati language," in *Proc. Int. Conf. Asian Language Processing (IALP)*, Kuching, Malaysia, 2014, pp. 46–49, doi: 10.1109/IALP.2014.6973506. | 1 | Authors, title, venue, pages and DOI taken from a search result; the method and results (modulation spectrogram, SVM, MFCC fusion helps) were confirmed from that result's summary. Full text not read. |
| 5 | S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, Aug. 1980, doi: 10.1109/TASSP.1980.1163420. | 2 | Authors, journal, volume/pages, date and DOI confirmed; result (MFCC performed best) confirmed from the abstract. |
| 6 | S. Furui, "Speaker-independent isolated word recognition using dynamic features of speech spectrum," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 34, no. 1, pp. 52–59, Feb. 1986, doi: 10.1109/TASSP.1986.1164788. | 2 | Authors, journal, volume/pages, date and DOI confirmed; abstract content (dynamic features, speaker-independent word recognition) confirmed. |
| 7 | S. Shon, A. Ali, and J. Glass, "Convolutional neural networks and language embeddings for end-to-end dialect recognition," in *Proc. Odyssey 2018: The Speaker and Language Recognition Workshop*, Les Sables d'Olonne, France, 2018, pp. 98–104, doi: 10.21437/Odyssey.2018-14. | 1 | Authors, title, venue, pages and DOI confirmed; findings used (5 Arabic dialects, MFCC vs filter-bank, speed perturbation) confirmed from the abstract/summary. |
| 8 | S. Darshana *et al.*, "MARS: A hybrid deep CNN-based multi-accent recognition system for English language," in *Proc. Int. Conf. Advances in Information Technology and Pattern Recognition (ICAITPR)*, 2022, doi: 10.1109/ICAITPR51569.2022.9844177. | 1 | **Author list is incomplete**: the search returned only 'S. Darshana et al.'; the first author's initial and the other authors were not verified. Title, venue and DOI confirmed. Findings used (six Indian states, classifier list, residual network best) come from a summary, not the full text. |
| 9 | C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995, doi: 10.1007/BF00994018. | 2 | Authors, journal, volume/pages, year and DOI confirmed. |
| 10 | S. Hershey *et al.*, "CNN architectures for large-scale audio classification," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2017, arXiv:1609.09430. | 1 | Title and arXiv number confirmed; author list (first author S. Hershey and co-authors) and ICASSP 2017 venue confirmed from arXiv/Google Research listings; proceedings pages not verified. |
| 11 | D. S. Park *et al.*, "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech 2019*, 2019, pp. 2613–2617, arXiv:1904.08779. | 2 | Authors, venue, pages (2613-2617) and arXiv number confirmed. |
| 12 | I. R. Titze, "Physiologic and acoustic differences between male and female voices," *J. Acoust. Soc. Am.*, vol. 85, no. 4, pp. 1699–1707, Apr. 1989, doi: 10.1121/1.397959. | 2 | Author, journal, volume/issue/pages, date and DOI confirmed; only the title-level statement that male and female voices differ physiologically and acoustically is used. |
| 13 | R. Tatman, "Gender and dialect bias in YouTube's automatic captions," in *Proc. First ACL Workshop on Ethics in Natural Language Processing*, Valencia, Spain, Apr. 2017, pp. 53–59, doi: 10.18653/v1/W17-1606. | 2 | Author, title, workshop, pages, date and DOI confirmed (ACL Anthology W17-1606). |
| 14 | B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf. (SciPy)*, Austin, TX, USA, 2015, doi: 10.25080/Majora-7b98e3ed-003. | 1 | Authors, title, conference and DOI confirmed. **Page numbers conflict between sources (18-24 vs 18-25) and are omitted.** |
| 15 | I. T. Jolliffe and J. Cadima, "Principal component analysis: A review and recent developments," *Phil. Trans. R. Soc. A*, vol. 374, no. 2065, art. 20150202, Apr. 2016, doi: 10.1098/rsta.2015.0202. | 1 | Authors, journal, volume/issue, article number, date and DOI confirmed. |
| 16 | S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, PMLR vol. 37, 2015, pp. 448–456. | 1 | Authors, title, conference, volume and pages confirmed (PMLR 37:448-456). |
| 17 | N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014. | 1 | Authors, journal, volume, pages and year confirmed. |
| 18 | D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015, arXiv:1412.6980. | 1 | Authors, title, venue (ICLR 2015) and arXiv number confirmed. |

## 3. Figures

| Figure | Source image | Caption (abridged) |
|---|---|---|
| 1 | `figures/paper_fig_workflow.png` | Overall research workflow, from data collection to the prediction application. The short notes under each stag… |
| 2 | `figures/paper_fig_preprocessing.png` | Audio preprocessing pipeline. Only the file format is standardised; the content of the signal is left as recor… |
| 3 | `../results/audio_analysis/09_quality_summary.png` | Audio quality summary for all recordings (colour = accent). Left: peak level. Middle: speech loudness, with th… |
| 4 | `../results/audio_analysis/07_speech_activity_timelines.png` | Energy-based speech activity for one representative recording per accent. The solid line is frame energy, the … |
| 5 | `figures/paper_fig_mfcc_pipeline.png` | From a recording to 78 MFCC-based features. The first row produces frame-level spectra, the second row produce… |
| 6 | `../results/feature_analysis/figures/10_accent_separation_heatmaps_all_frames.png` | Accent separation for all 78 features (whole recordings). Left: η² of the accent grouping for each coefficient… |
| 7 | `../results/feature_analysis/figures/11_feature_correlation_heatmap.png` | Correlation between the 78 features across the 35 recordings (red = move together, blue = move in opposite dir… |
| 8 | `../results/feature_analysis/figures/18_speechshare_vs_mfcc12_all_vs_speech.png` | Mean MFCC 1 (top) and MFCC 2 (bottom) against the share of each recording that is speech, using all frames (le… |
| 9 | `../results/feature_analysis/figures/21_format_vs_accent_eta2.png` | Is a feature mostly an M4A-versus-OGG split? Left: η² explained by original format. Right: η² of accent that r… |
| 10 | `figures/paper_fig_pca_workflow.png` | Principal component analysis workflow used in this study. |
| 11 | `../results/pca/figures/scree_plot.png` | Scree plot: the percentage of variance explained by each of the 34 principal components. The dashed line marks… |
| 12 | `figures/paper_fig_mlp_architecture.png` | Architecture of the selected final model: an MLP with a shared 64-unit layer and two softmax output heads. Par… |
| 13 | `figures/phase6_hierarchical_model_architecture.png` | The CNN–MLP fusion network evaluated as an experimental candidate. Branch A (top) reads the Log-Mel spectrogra… |
| 14 | `figures/paper_fig_app_pipeline.png` | End-to-end prediction pipeline of the web application. |
| 15 | `../results/mfcc/figures/comparison_mfcc_accent_averages.png` | Accent-average MFCC values (each cell is the mean of the seven recording means of that accent). MFCC 1 is plot… |
| 16 | `../results/mfcc/figures/delta_variability_by_accent.png` | Mean within-recording standard deviation of the delta (left) and delta-delta (right) coefficients for each acc… |
| 17 | `../results/pca/figures/cumulative_variance.png` | Cumulative explained variance. The dots mark the number of components needed to reach 80, 90 and 95% of the va… |
| 18 | `../results/pca/figures/pc1_pc2_scatter.png` | Recordings in the plane of the first two principal components (colour = accent, marker = original format, labe… |
| 19 | `../results/pca/figures/loadings_top_features.png` | The ten features with the largest loadings on components 1 to 3, coloured by feature family (MFCC, delta, delt… |
| 20 | `../results/pca/figures/pc_condition_associations.png` | Left: how much of each component's variation is explained by accent, original format and original sample rate … |
| 21 | `../results/pca/figures/surti_leave_out_pc1_pc2.png` | Left: the full PCA with Surti highlighted. Right: a PCA fitted without Surti, into which the Surti recordings … |
| 22 | `../results/pca/figures/sensitivity_pc1_pc2.png` | Components 1 and 2 for the primary analysis and the three sensitivity analyses. |
| 23 | `../results/model_development/evaluation/figures/fold_accuracy_heatmap.png` | Accent accuracy of each held-out fold (five recordings per fold, one per accent) for every completed run. |
| 24 | `../results/model_development/evaluation/figures/experiment_comparison.png` | Pooled accuracy (bars) and fold-wise mean ± standard deviation (points) for all completed runs, with the chanc… |
| 25 | `../results/model_development/evaluation/figures/confusion_matrices_accent_A_to_D.png` | Accent confusion matrices pooled over the seven test folds for experiments A to D (rows: true class; columns: … |
| 26 | `../results/model_development/fusion_model/training_curves.png` | Training and validation curves of the fusion network (patience 10). Thin lines are individual folds, thick lin… |
| 27 | `../results/model_development/evaluation/figures/train_val_test_accuracy.png` | Mean accuracy on the training recordings (without augmentation), the validation recordings and the held-out te… |
| 28 | `../results/model_development/evaluation/figures/cnn_batchnorm_diagnostic.png` | Stored BatchNorm running variance relative to the true activation variance, in the last convolution block, aga… |

The workflow, preprocessing, MFCC, PCA, final-MLP and application-pipeline diagrams are new schematic figures drawn by `scripts/make_paper_figures.py`; the fusion-network diagram comes from Phase 6; all other figures are existing results of Phases 2 to 6. **No frontend screenshot exists in the project, so none is included.**

## 4. Tables

| Table | Caption |
|---|---|
| 1 | The dataset: five accents, five speakers, 35 recordings. Original sample rates were read from the original files; all analysis uses the stan… |
| 2 | Recording quality by accent (means over the seven recordings of each accent). Levels are in decibels relative to full scale (dBFS). |
| 3 | Composition of the 78 recording-level features. |
| 4 | Feature-extraction configuration. |
| 5 | The four model types compared (trainable parameter counts for the neural models; the CNN-only variant, which has the CNN branch plus the sha… |
| 6 | Evaluation protocol. |
| 7 | The first five principal components: variance explained, the chance level from shuffled data, and η² of each component's scores with accent … |
| 8 | Sensitivity analyses. The primary analysis is the first row; the last column gives the absolute correlation between the primary scores and t… |
| 9 | Results of all experiments on identical seven-fold splits (accuracy is for the accent head, pooled over the 35 held-out recordings). Post-ho… |
| 10 | Frontend tests that were run (all passed). |

## 5. Numerical results

Dataset, quality, feature, PCA and model numbers and all tables are computed from `dataset/recordings_metadata.csv` and the files under `results/` when the paper is built. A few statements are not regenerated because they come from figures, logs or one-off checks; each was checked against its source:

- mean MFCC 1 of Surti (about -353) and Kathiyawadi (about -494): computed from `mfcc_recording_features.csv` when the paper was written;
- the 18 and 232 strongly correlated feature pairs, 75.5% speech frames, the 51/20/7 separation classes and the 10 Surti-distinct features: recomputed from the Phase 3 and 4 tables;
- the format eta-squared of 0.95 for MFCC 2 std and the shape of the validation-loss curves (plateau in the first ~10 epochs, steep fall between epochs 15 and 40): read from the corresponding figures;
- agreement of the PCA implementation with scikit-learn (about 4e-16): a one-off check made during Phase 5, not stored as a file;
- training times (83 s, 443 s and 2,092 s for seven folds): from `results/model_development/evaluation/training_logs/phase6_train.log`;
- frontend tests: `results/frontend/frontend_test_results.txt`.

## 6. Information that could not be verified

- The original literature review (none could be found); the new references were mostly read at abstract or catalogue level, not in full text.
- The author list of reference [8]; the page numbers of the librosa paper; the year of the Mesthrie and Chavda article (2020 or 2021).
- Speaker gender, age and recording details (not in the metadata); the cause of the very clean OGG recordings (stated as an untested hypothesis); the cause of the CNN training problem (a hypothesis; the controlled test was not run).
- That the application displays correctly in real browsers (not tested; no screenshot).
- The authors of the paper: the byline is a placeholder.

## 7. Size

About 12,845 words including tables and references (12,249 before the reference list); 28 figures; 10 tables; 18 references.
