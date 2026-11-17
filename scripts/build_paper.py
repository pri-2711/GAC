"""Build reports/research_paper.md from scripts/paper_template.md (run: python scripts/build_paper.py).

Every number in the paper is read from the saved project results; the template holds only the prose.

Template markers
  @@name@@                     -> a number/string computed from the saved project results (facts below)
  %%FIG key | path | caption%%  -> numbered figure with caption (path relative to reports/)
  %%TBL key | caption%%         -> numbered table (content from TABLES[key]) with caption
  @fig:key / @tbl:key           -> "Figure N" / "Table N"
  [@a] or [@a; @b]              -> numbered citations (IEEE, by first appearance); REFS holds the verified bibliography
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
RES = R / "results"
TEMPLATE = Path(__file__).with_name("paper_template.md")

# ----------------------------------------------------------------------------- verified bibliography
REFS = {
    "cardona": 'G. Cardona and B. Suthar, "Gujarati," in *The Indo-Aryan Languages*, G. Cardona and D. Jain, Eds. London, U.K.: Routledge, 2003, pp. 659–697.',
    "mesthrie": 'R. Mesthrie and V. Chavda, "Cape Town Gujarati and its relation to Gujarati dialectology: A study of retroflex boosting," *Journal of South Asian Languages and Linguistics*, vol. 7, no. 1, pp. 1–14, 2021, doi: 10.1515/jsall-2020-2022.',
    "shah": 'M. M. Shah and H. R. Kavathiya, "Voice recognition for Gujarati dialects: An in-depth survey," *International Journal of Computer Applications*, vol. 186, no. 5, pp. 1–4, Jan. 2024, doi: 10.5120/ijca2024923112.',
    "chittora": 'A. Chittora and H. A. Patil, "Classification of phonemes using modulation spectrogram based features for Gujarati language," in *Proc. Int. Conf. Asian Language Processing (IALP)*, Kuching, Malaysia, 2014, pp. 46–49, doi: 10.1109/IALP.2014.6973506.',
    "davis": 'S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, Aug. 1980, doi: 10.1109/TASSP.1980.1163420.',
    "furui": 'S. Furui, "Speaker-independent isolated word recognition using dynamic features of speech spectrum," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 34, no. 1, pp. 52–59, Feb. 1986, doi: 10.1109/TASSP.1986.1164788.',
    "shon": 'S. Shon, A. Ali, and J. Glass, "Convolutional neural networks and language embeddings for end-to-end dialect recognition," in *Proc. Odyssey 2018: The Speaker and Language Recognition Workshop*, Les Sables d\'Olonne, France, 2018, pp. 98–104, doi: 10.21437/Odyssey.2018-14.',
    "darshana": 'S. Darshana *et al.*, "MARS: A hybrid deep CNN-based multi-accent recognition system for English language," in *Proc. Int. Conf. Advances in Information Technology and Pattern Recognition (ICAITPR)*, 2022, doi: 10.1109/ICAITPR51569.2022.9844177.',
    "cortes": 'C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995, doi: 10.1007/BF00994018.',
    "hershey": 'S. Hershey *et al.*, "CNN architectures for large-scale audio classification," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2017, arXiv:1609.09430.',
    "park": 'D. S. Park *et al.*, "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech 2019*, 2019, pp. 2613–2617, arXiv:1904.08779.',
    "tatman": 'R. Tatman, "Gender and dialect bias in YouTube\'s automatic captions," in *Proc. First ACL Workshop on Ethics in Natural Language Processing*, Valencia, Spain, Apr. 2017, pp. 53–59, doi: 10.18653/v1/W17-1606.',
    "titze": 'I. R. Titze, "Physiologic and acoustic differences between male and female voices," *J. Acoust. Soc. Am.*, vol. 85, no. 4, pp. 1699–1707, Apr. 1989, doi: 10.1121/1.397959.',
    "jolliffe": 'I. T. Jolliffe and J. Cadima, "Principal component analysis: A review and recent developments," *Phil. Trans. R. Soc. A*, vol. 374, no. 2065, art. 20150202, Apr. 2016, doi: 10.1098/rsta.2015.0202.',
    "mcfee": 'B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf. (SciPy)*, Austin, TX, USA, 2015, doi: 10.25080/Majora-7b98e3ed-003.',
    "ioffe": 'S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, PMLR vol. 37, 2015, pp. 448–456.',
    "srivastava": 'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014.',
    "kingma": 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015, arXiv:1412.6980.',
}


# ----------------------------------------------------------------------------- facts
def fmt(v, nd=3):
    return f"{v:.{nd}f}"


meta = pd.read_csv(R / "dataset" / "recordings_metadata.csv")
meta = meta[meta.conversion_status.astype(str).str.startswith("success")]
q = pd.read_csv(RES / "audio_analysis" / "audio_quality_metrics.csv")
feat = pd.read_csv(RES / "mfcc" / "mfcc_recording_features.csv")
schema = json.loads((RES / "mfcc" / "mfcc_feature_columns.json").read_text(encoding="utf-8"))
cols = schema["acoustic_feature_columns"]
params = schema["extraction_parameters"]
fs = pd.read_csv(RES / "feature_analysis" / "feature_analysis_summary.csv")
sp = pd.read_csv(RES / "feature_analysis" / "speech_only_recording_features.csv")
qc = pd.read_csv(RES / "feature_analysis" / "quality_correlations.csv")
ev = pd.read_csv(RES / "pca" / "explained_variance.csv")
pm = json.loads((RES / "pca" / "pca_manifest.json").read_text(encoding="utf-8"))
eta = pd.read_csv(RES / "pca" / "pc_grouping_eta2.csv")
ovr = pd.read_csv(RES / "pca" / "pc_accent_one_vs_rest_eta2.csv").set_index("component")
chance = pd.read_csv(RES / "pca" / "permutation_baseline.csv").set_index("component")
sens = pd.read_csv(RES / "pca" / "sensitivity_summary.csv").set_index("variant")
loo = pd.read_csv(RES / "pca" / "leave_one_accent_out.csv").set_index("left_out_accent")
gc = pd.read_csv(RES / "pca" / "loading_group_contributions.csv")
cond = pd.read_csv(RES / "pca" / "pc_condition_associations.csv")
sep = pd.read_csv(RES / "pca" / "separation_metrics.csv")
pairs = pd.read_csv(RES / "pca" / "accent_pair_separation.csv")
comp = pd.read_csv(RES / "model_development" / "evaluation" / "experiment_comparison.csv").set_index("run")
status = pd.read_csv(RES / "model_development" / "evaluation" / "run_status.csv").set_index("run")
fm = pd.read_csv(RES / "model_development" / "evaluation" / "fold_accuracy_matrix.csv", index_col=0)
shuf = pd.read_csv(RES / "model_development" / "svm_baseline" / "label_shuffle_control.csv")
extra = pd.read_csv(RES / "model_development" / "additional_features" / "additional_acoustic_features.csv")
final_info = json.loads((RES / "model_development" / "final_model" / "final_model_info.json").read_text(encoding="utf-8"))
bn = pd.read_csv(RES / "model_development" / "evaluation" / "cnn_batchnorm_statistics_diagnostic.csv")
fe_tests = (RES / "frontend" / "frontend_test_results.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
scores = pd.read_csv(RES / "pca" / "pca_scores.csv")
log = (RES / "model_development" / "evaluation" / "training_logs" / "phase6_train.log").read_text(encoding="utf-8")
t_mlp = re.search(r"\[done\] mlp_only: .*\((\d+)s\)", log).group(1)
t_fus = re.search(r"\[done\] fusion_model: .*\((\d+)s\)", log).group(1)
t_cnn = re.search(r"\[done\] cnn_only: .*\((\d+)s\)", log).group(1)

acc_order = list(dict.fromkeys(meta.accent))
qa = q.groupby("accent").mean(numeric_only=True).loc[acc_order]
rate = scores.groupby("accent").original_sample_rate_hz.first()

F = {}
F["n_rec"] = len(meta)
F["n_acc"] = meta.accent.nunique()
F["n_spk"] = meta.speaker_id.nunique()
F["per_acc"] = int(meta.groupby("accent").size().iloc[0])
F["total_min"] = f"{meta.duration_seconds.sum() / 60:.1f}"
F["dur_min"] = f"{meta.duration_seconds.min():.1f}"
F["dur_max"] = f"{meta.duration_seconds.max():.1f}"
F["dur_mean"] = f"{meta.duration_seconds.mean():.1f}"
F["n_flag"] = int((q.quality_flags.fillna("") != "").sum())
F["n_clip"] = int(q.flag_clipping.sum())
F["clip_meh"] = int(q[q.accent == "Mehsani"].flag_clipping.sum())
F["clip_kat"] = int(q[q.accent == "Kathiyawadi"].flag_clipping.sum())
F["n_level"] = int(q.flag_level_inconsistent.sum())
for a in acc_order:
    k = a[:3].lower()
    F[f"nf_{k}"] = f"{qa.loc[a, 'noise_floor_dbfs']:.1f}"
    F[f"snr_{k}"] = f"{qa.loc[a, 'approx_snr_db']:.1f}"
    F[f"sr_{k}"] = f"{qa.loc[a, 'speech_ratio']:.2f}"
    F[f"dur_{k}"] = f"{qa.loc[a, 'duration_s']:.1f}"
F["frames_total"] = f"{int(feat.shape[0] and np.nan_to_num(sp.n_frames_total.sum())):,}"
F["frames_min"] = int(sp.n_frames_total.min())
F["frames_max"] = int(sp.n_frames_total.max())
F["speech_pct"] = f"{sp.n_speech_frames.sum() / sp.n_frames_total.sum() * 100:.1f}"
F["speech_frames"] = f"{int(sp.n_speech_frames.sum()):,}"
for key, val in params.items():
    F[f"p_{key}"] = val
cm = feat[cols].corr().abs().values
iu = np.triu_indices(len(cols), 1)
F["pairs_tot"] = f"{len(iu[0]):,}"
F["pairs_09"] = int((cm[iu] >= 0.9).sum())
F["pairs_07"] = int((cm[iu] >= 0.7).sum())
F["eta_hi_all"] = int((fs.eta2_accent_all >= .6).sum())
F["eta_mod_all"] = int(((fs.eta2_accent_all >= .3) & (fs.eta2_accent_all < .6)).sum())
F["eta_lo_all"] = int((fs.eta2_accent_all < .3).sum())
F["eta_hi_sp"] = int((fs.eta2_accent_speech >= .6).sum())
F["eta_mod_sp"] = int(((fs.eta2_accent_speech >= .3) & (fs.eta2_accent_speech < .6)).sum())
F["eta_lo_sp"] = int((fs.eta2_accent_speech < .3).sum())
top = fs.sort_values("eta2_accent_all", ascending=False).head(3)
F["top_eta"] = "; ".join(f"{r.label} ({r.eta2_accent_all:.2f})" for r in top.itertuples())
F["n_cand"] = int(fs.candidate_for_follow_up.sum())
def wr(f, s):
    return qc[(qc.feature == f) & (qc.quality_metric == "speech_ratio") & (qc.feature_set == s)].pearson_within_accent.iloc[0]
F["r1_all"], F["r2_all"] = f"{wr('mfcc_01_mean', 'all_frames'):.2f}", f"{wr('mfcc_02_mean', 'all_frames'):.2f}"
F["r1_sp"], F["r2_sp"] = f"{wr('mfcc_01_mean', 'speech_only'):.2f}", f"{wr('mfcc_02_mean', 'speech_only'):.2f}"
dl = feat.assign(d=feat[[f"delta_{i:02d}_std" for i in range(1, 14)]].mean(axis=1)).groupby("accent").d.mean()
for a in acc_order:
    F[f"dl_{a[:3].lower()}"] = f"{dl[a]:.2f}"
# PCA
F["pc1"], F["pc2"], F["pc3"], F["pc4"], F["pc5"] = [f"{ev.explained_variance_ratio[i] * 100:.1f}" for i in range(5)]
F["cum2"] = f"{ev.cumulative_explained_variance[1] * 100:.1f}"
F["cum4"] = f"{ev.cumulative_explained_variance[3] * 100:.1f}"
F["cum5"] = f"{ev.cumulative_explained_variance[4] * 100:.1f}"
F["k80"], F["k90"], F["k95"] = pm["components_for_variance"]["80%"], pm["components_for_variance"]["90%"], pm["components_for_variance"]["95%"]
F["ncomp"] = pm["n_components"]
F["ch1"], F["ch2"], F["ch3"], F["ch4"], F["ch5"] = [f"{chance.chance_mean.iloc[i] * 100:.1f}" for i in range(5)]
F["ch1_95"] = f"{chance.chance_95th_percentile.iloc[0] * 100:.1f}"
for i in range(1, 6):
    F[f"eta_pc{i}"] = f"{eta.eta2_accent[i - 1]:.2f}"
F["eta_pc1_fmt"] = f"{eta.eta2_original_format[0]:.2f}"
F["ovr_kat_pc1"] = f"{ovr.loc['PC1', 'Kathiyawadi']:.2f}"
F["ovr_kut_pc2"] = f"{ovr.loc['PC2', 'Kutchi']:.2f}"
F["ovr_ahm_pc3"] = f"{ovr.loc['PC3', 'Ahmedabad']:.2f}"
F["ovr_meh_pc4"] = f"{ovr.loc['PC4', 'Mehsani']:.2f}"
F["ovr_sur_pc2"] = f"{ovr.loc['PC2', 'Surti']:.2f}"
g1 = gc[gc.component == "PC1"]
F["pc1_std_share"] = f"{g1[g1.statistic == 'std'].share_of_squared_loadings.sum() * 100:.0f}"
g3 = gc[gc.component == "PC3"]
F["pc3_mean_share"] = f"{g3[g3.statistic == 'mean'].share_of_squared_loadings.sum() * 100:.0f}"
F["sil2"], F["sil5"] = f"{sep.silhouette_accent[0]:.2f}", f"{sep.silhouette_accent[1]:.2f}"
F["sil2_fmt"] = f"{sep.silhouette_original_format[0]:.2f}"
p_ahm_meh = pairs[(pairs.accent_a == "Ahmedabad") & (pairs.accent_b == "Mehsani")].iloc[0]
F["d_ahm_meh"] = f"{p_ahm_meh['centroid_distance_over_spread_PC1-2']:.1f}"
F["sens_s30_pc1"], F["sens_s20_pc1"], F["sens_no12_pc1"] = [f"{sens.loc[v, 'PC1_variance'] * 100:.1f}" for v in ("speech_only_30db", "speech_only_20db", "without_mfcc1_mfcc2")]
F["no12_r1"], F["no12_r2"], F["no12_r3"] = [f"{sens.loc['without_mfcc1_mfcc2', f'abs_r_primary_PC{i}_vs_same_PC']:.2f}" for i in (1, 2, 3)]
F["s30_r2"] = f"{sens.loc['speech_only_30db', 'abs_r_primary_PC2_vs_same_PC']:.2f}"
F["s30_r2_best"] = f"{sens.loc['speech_only_30db', 'abs_r_primary_PC2_vs_best_PC']:.2f}"
F["s30_r1"] = f"{sens.loc['speech_only_30db', 'abs_r_primary_PC1_vs_same_PC']:.2f}"
F["loo_sur_pc1"], F["loo_sur_pc2"], F["loo_sur_pc3"] = [f"{loo.loc['Surti', f'PC{i}_loading_cosine_with_primary']:.2f}" for i in (1, 2, 3)]
F["loo_sur_dist"] = f"{loo.loc['Surti', 'held_out_centroid_distance_over_fit_spread_PC1-3']:.2f}"
F["loo_kat_dist"] = f"{loo.loc['Kathiyawadi', 'held_out_centroid_distance_over_fit_spread_PC1-3']:.2f}"
def cr(pc, c):
    return cond[(cond.component == pc) & (cond.condition == c)].pearson_overall.iloc[0]
F["pc1_sr"] = f"{cr('PC1', 'original_sample_rate_hz'):.2f}"
F["pc1_nf"] = f"{cr('PC1', 'noise_floor_dbfs'):.2f}"
F["pc1_spr"] = f"{cr('PC1', 'speech_ratio'):.2f}"
wi = cond[(cond.component == "PC1") & (cond.condition == "speech_ratio")].pearson_within_accent.iloc[0]
F["pc1_spr_within"] = f"{wi:.2f}"
# models
def c(run, col, nd=3):
    return f"{comp.loc[run, col]:.{nd}f}"
for run, k in (("svm_baseline", "A"), ("cnn_only", "B"), ("mlp_only", "C"), ("fusion_model", "D"), ("cnn_only_patience25", "Bp"), ("fusion_patience25", "Dp"), ("fusion_noaug", "Dn")):
    F[f"acc_{k}"] = c(run, "pooled_accent_accuracy")
    F[f"sd_{k}"] = c(run, "fold_sd")
    F[f"f1_{k}"] = c(run, "macro_f1")
    F[f"reg_{k}"] = c(run, "region_accuracy")
F["shuf_mean"] = f"{shuf.test_accent_accuracy.mean():.3f}"
F["shuf_n"] = len(shuf)
F["t_mlp"], F["t_fus"], F["t_cnn"] = f"{int(t_mlp):,}", f"{int(t_fus):,}", f"{int(t_cnn):,}"
F["n_fit"] = final_info["n_fit_recordings"]
F["n_val_final"] = len(final_info["validation_recordings"])
F["fe_tests"] = fe_tests.split()[0]
F["f0_min"] = f"{extra.f0_mean_hz.min():.0f}"
F["f0_max"] = f"{extra.f0_mean_hz.max():.0f}"
f0 = extra.assign(a=extra.recording_id.str[:3]).groupby("a").f0_mean_hz.mean()
F["f0_lo_spk"], F["f0_hi_spk"] = f"{f0.min():.0f}", f"{f0.max():.0f}"
bn3 = bn[bn.bn_layer == "conv3_bn"]
early, late = bn3[bn3.best_epoch <= 12], bn3[bn3.best_epoch > 40]
F["bn_early_lo"], F["bn_early_hi"] = f"{early.median_var_ratio_running_over_true.min():.1f}", f"{early.median_var_ratio_running_over_true.max():.1f}"
F["bn_late_lo"], F["bn_late_hi"] = f"{late.median_var_ratio_running_over_true.min():.2f}", f"{late.median_var_ratio_running_over_true.max():.2f}"
F["bn_n_early"], F["bn_n_late"] = len(early), len(late)
inoaug = pd.read_csv(RES / "model_development" / "fusion_model" / "supplementary_patience25_no_augmentation" / "run_info.csv")
iaug = pd.read_csv(RES / "model_development" / "fusion_model" / "supplementary_patience25" / "run_info.csv")
F["noaug_short"] = int((inoaug.epochs_run <= 30).sum())
F["aug_short"] = int((iaug.epochs_run <= 30).sum())
ifus = pd.read_csv(RES / "model_development" / "fusion_model" / "run_info.csv")
F["fus_short"] = int((ifus.epochs_run <= 20).sum())
icnn = pd.read_csv(RES / "model_development" / "cnn_only" / "run_info.csv")
F["cnn_ep_lo"], F["cnn_ep_hi"] = int(icnn.epochs_run.min()), int(icnn.epochs_run.max())
F["pc1_min_r"] = f"{sens.abs_r_primary_PC1_vs_same_PC.min():.2f}"
_o = pd.read_csv(RES / "model_development" / "cnn_only" / "overfit_summary.csv")
F["cnn_tr_lo"], F["cnn_tr_hi"] = f"{_o.train_accent_accuracy.min():.2f}", f"{_o.train_accent_accuracy.max():.2f}"
_f = pd.read_csv(RES / "model_development" / "fusion_model" / "overfit_summary.csv"); _f = _f[_f.best_epoch <= 12]
F["fus_tr_lo"], F["fus_tr_hi"] = f"{_f.train_accent_accuracy.min():.2f}", f"{_f.train_accent_accuracy.max():.2f}"
_im = pd.read_csv(RES / "model_development" / "mlp_only" / "run_info.csv")
F["mlp_ep_lo"], F["mlp_ep_hi"] = int(_im.epochs_run.min()), int(_im.epochs_run.max())
F["p_mlp"] = f"{int(pd.read_csv(RES / 'model_development' / 'mlp_only' / 'architecture_layer_table.csv').trainable_params.sum()):,}"
F["p_fus"] = f"{int(pd.read_csv(RES / 'model_development' / 'fusion_model' / 'architecture_layer_table.csv').trainable_params.sum()):,}"
F["p_cnn"] = f"{int(pd.read_csv(RES / 'model_development' / 'cnn_only' / 'architecture_layer_table.csv').trainable_params.sum()):,}"
_pm = int(pd.read_csv(RES / 'model_development' / 'mlp_only' / 'architecture_layer_table.csv').trainable_params.sum())
_pf = int(pd.read_csv(RES / 'model_development' / 'fusion_model' / 'architecture_layer_table.csv').trainable_params.sum())
F["pct_mlp_fus"] = f"{_pm / _pf * 100:.0f}"
F["mlp_top"] = f"{comp.loc['mlp_only', 'mean_top_probability']:.3f}"
_s = sp[sp.accent == "Surti"]; _o = sp[sp.accent != "Surti"]
F["surti_apart"] = sum(bool((_s[c_] > _o[c_].max()).all() or (_s[c_] < _o[c_].min()).all()) for c_ in cols)


# ----------------------------------------------------------------------------- tables
def md_table(df, floatfmt="{:.3f}"):
    cols_ = list(df.columns)
    rows = ["| " + " | ".join(str(c) for c in cols_) + " |", "|" + "---|" * len(cols_)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(rows)


T = {}
d = meta.groupby(["region", "accent", "speaker_id"], sort=False).agg(n=("recording_id", "count"), fmt=("original_format", lambda s: "/".join(sorted(set(s)))),
                                                                   mn=("duration_seconds", "min"), mx=("duration_seconds", "max"), mean=("duration_seconds", "mean")).reset_index()
d["sr"] = d.accent.map(lambda a: f"{rate[a] / 1000:g} kHz")
d["dur"] = d.apply(lambda r: f"{r.mn:.1f}–{r.mx:.1f} (mean {r['mean']:.1f})", axis=1)
T["dataset"] = md_table(pd.DataFrame({"Region": d.region, "Accent": d.accent, "Speaker ID": d.speaker_id, "Recordings": d.n, "Original format": d.fmt,
                                       "Original sample rate": d.sr, "Duration, s (range)": d.dur}))
qt = pd.DataFrame({"Accent": acc_order, "Peak, dBFS": [f"{qa.loc[a, 'peak_dbfs']:.1f}" for a in acc_order], "Noise floor, dBFS": [f"{qa.loc[a, 'noise_floor_dbfs']:.1f}" for a in acc_order],
                   "Approx. SNR, dB": [f"{qa.loc[a, 'approx_snr_db']:.1f}" for a in acc_order], "Speech share": [f"{qa.loc[a, 'speech_ratio']:.2f}" for a in acc_order],
                   "Internal pauses": [f"{qa.loc[a, 'n_internal_pauses']:.1f}" for a in acc_order],
                   "Clipping flags": [int(q[q.accent == a].flag_clipping.sum()) for a in acc_order]})
T["quality"] = md_table(qt)
T["mfccparams"] = md_table(pd.DataFrame([("Sampling rate", f"{params['sample_rate']:,} Hz"), ("MFCC coefficients", params["n_mfcc"]), ("FFT size (n_fft)", params["n_fft"]),
                                          ("Window length", f"{params['win_length']} samples (25 ms), {params['window']} window"), ("Hop length", f"{params['hop_length']} samples (10 ms)"),
                                          ("Mel filter bands", params["n_mels"]), ("DCT", f"type {params['dct_type']}, norm = {params['norm']}"), ("Frame centring", params["center"]),
                                          ("Delta width", f"{params['delta_width']} frames (about 90 ms)")], columns=["Setting", "Value"]))
T["features78"] = md_table(pd.DataFrame([("MFCC 1–13", "mean, standard deviation", 26), ("Delta MFCC 1–13", "mean, standard deviation", 26), ("Delta-delta MFCC 1–13", "mean, standard deviation", 26), ("**Total**", "", 78)],
                                         columns=["Feature group", "Statistics per coefficient", "Features"]))
T["pcavar"] = md_table(pd.DataFrame({"Component": [f"PC{i}" for i in range(1, 6)], "Variance explained (%)": [F[f"pc{i}"] for i in range(1, 6)],
                                      "Chance level, mean of shuffled data (%)": [F[f"ch{i}"] for i in range(1, 6)], "η² with accent": [F[f"eta_pc{i}"] for i in range(1, 6)],
                                      "η² with original format": [f"{eta.eta2_original_format[i]:.2f}" for i in range(5)]}))
T["pcasens"] = md_table(pd.DataFrame({"Variant": [sens.loc[v, "description"] for v in sens.index], "Features": [int(sens.loc[v, "n_features"]) for v in sens.index],
                                       "PC1 (%)": [f"{sens.loc[v, 'PC1_variance'] * 100:.1f}" for v in sens.index], "PC2 (%)": [f"{sens.loc[v, 'PC2_variance'] * 100:.1f}" for v in sens.index],
                                       "Components for 80 / 90 / 95%": [f"{int(sens.loc[v, 'components_for_80pct'])} / {int(sens.loc[v, 'components_for_90pct'])} / {int(sens.loc[v, 'components_for_95pct'])}" for v in sens.index],
                                       "Absolute correlation with primary PC1 / PC2 / PC3": [f"{sens.loc[v, 'abs_r_primary_PC1_vs_same_PC']:.2f} / {sens.loc[v, 'abs_r_primary_PC2_vs_same_PC']:.2f} / {sens.loc[v, 'abs_r_primary_PC3_vs_same_PC']:.2f}" for v in sens.index]}))
T["models"] = md_table(pd.DataFrame([
    ("A", "SVM (RBF kernel, C = 1)", "78 numerical features", "Support-vector classifier on standardised features", "not applicable"),
    ("B", "CNN only", "Log-Mel spectrogram (64 × 800)", "3 convolution blocks → 128-d → shared Dense(64) → two heads", F["p_cnn"]),
    ("C", "MLP only (selected)", "78 numerical features", "Dense 64 → Dense 32 → shared Dense 64 → two heads", F["p_mlp"]),
    ("D", "CNN–MLP fusion", "Log-Mel spectrogram and 78 features", "CNN 128-d and MLP 32-d → concatenate (160-d) → shared Dense(64) → two heads", F["p_fus"])],
    columns=["Exp.", "Model", "Input", "Structure", "Trainable parameters"]))
rows = []
def addrow(run, label, kind):
    r = comp.loc[run]
    rows.append((label, kind, "complete (7 of 7 folds)", f"{r.pooled_accent_accuracy:.3f}", f"{r.fold_mean:.3f} ± {r.fold_sd:.3f}", f"{r.macro_f1:.3f}", f"{r.region_accuracy:.3f}"))
addrow("svm_baseline", "A. SVM, 78 features", "Original")
addrow("mlp_only", "C. MLP, 78 features (**selected**)", "Original")
addrow("cnn_only", "B. CNN, Log-Mel, patience 10", "Original")
addrow("fusion_model", "D. CNN–MLP fusion, patience 10", "Original")
addrow("cnn_only_patience25", "B′. CNN, patience 25", "Post-hoc")
addrow("fusion_patience25", "D′. Fusion, patience 25", "Post-hoc")
addrow("fusion_noaug", "D″. Fusion, patience 25, no augmentation", "Post-hoc")
for label in ("MLP with the 13 additional spectral/F0 features", "MLP without augmentation", "CNN without augmentation", "BatchNorm-momentum / minimum-epoch test", "Speaker-independent evaluation"):
    rows.append((label, "Not run", "not run", "–", "–", "–", "–"))
T["comparison"] = md_table(pd.DataFrame(rows, columns=["Run", "Type", "Status", "Pooled accent accuracy", "Fold-wise mean ± SD", "Macro F1", "Region accuracy"]))
T["cv"] = md_table(pd.DataFrame([("Folds", "7, stratified by accent: each fold holds out one recording per accent (5 test recordings)"), ("Per fold", "25 training, 5 validation (one per accent), 5 test recordings"),
                                  ("Coverage", "every one of the 35 recordings is tested exactly once"), ("Validation use", "early stopping and checkpoint selection only"),
                                  ("Normalisation", "fitted on the training recordings of each fold only"), ("Augmentation", "training recordings only, applied on the fly inside the training fold"),
                                  ("Unit of evaluation", "recording (not speaker): test recordings share speaker, session and device with the training recordings")], columns=["Aspect", "Setting"]))
T["frontendtests"] = md_table(pd.DataFrame([
    ("Valid files", "WAV, original M4A, original OGG, MP3, 44.1 kHz stereo WAV, 8 kHz mono WAV: decoded, standardised and classified"),
    ("Consistency", "same result as the command-line script `scripts/predict_accent.py` (probability difference 0.0) and as the stored Phase 6 prediction"),
    ("Rejected with a message", "unsupported type, empty file, text file named .wav, corrupted M4A, zero-sample WAV, 0.3 s clip, digital silence, 190 s clip"),
    ("Accepted with a warning", "near-silent noise, 3 s clip"),
    ("Interface", "all four pages render; each sidebar tab switches the page; no radio widget; the gender-related limitation is displayed")], columns=["Test group", "What was checked"]))

# ----------------------------------------------------------------------------- build
text = TEMPLATE.read_text(encoding="utf-8")


def sub_facts(m):
    key = m.group(1)
    if key not in F:
        raise KeyError(f"unknown fact @@{key}@@")
    return str(F[key])


text = re.sub(r"@@([A-Za-z0-9_]+)@@", sub_facts, text)

fig_re = re.compile(r"%%FIG\s+([a-z0-9_]+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)%%", re.S)
tbl_re = re.compile(r"%%TBL\s+([a-z0-9_]+)\s*\|\s*(.+?)%%", re.S)
figs, tbls = {}, {}
for m in fig_re.finditer(text):
    figs[m.group(1)] = len(figs) + 1
for m in tbl_re.finditer(text):
    tbls[m.group(1)] = len(tbls) + 1
text = re.sub(r"@fig:([a-z0-9_]+)", lambda m: f"Figure {figs[m.group(1)]}", text)
text = re.sub(r"@tbl:([a-z0-9_]+)", lambda m: f"Table {tbls[m.group(1)]}", text)
text = fig_re.sub(lambda m: f"![Figure {figs[m.group(1)]}]({m.group(2).strip()})\n\n*Figure {figs[m.group(1)]}.* {m.group(3).strip()}\n", text)
text = tbl_re.sub(lambda m: f"*Table {tbls[m.group(1)]}.* {m.group(2).strip()}\n\n{T[m.group(1)]}\n", text)

order = []


def sub_cite(m):
    keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(f"unknown reference {k}")
        if k not in order:
            order.append(k)
        nums.append(order.index(k) + 1)
    return ", ".join(f"[{n}]" for n in sorted(nums))


text = re.sub(r"\[(@[a-z0-9_]+(?:\s*;\s*@[a-z0-9_]+)*)\]", sub_cite, text)
refs_md = "\n\n".join(f"[{i + 1}] {REFS[k]}" for i, k in enumerate(order))
unused = [k for k in REFS if k not in order]
assert not unused, f"unused references: {unused}"
text = text.replace("%%REFERENCES%%", refs_md)
left = re.findall(r"@@|%%|@fig:|@tbl:", text)
assert not left, f"unresolved markers: {left}"
out = R / "reports" / "research_paper.md"
out.write_text(text, encoding="utf-8")
words = len(re.findall(r"\b[\w'’\-]+\b", re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)))
print(f"written {out}  words={words}  figures={len(figs)}  tables={len(tbls)}  references={len(order)}")
