"""Phase 6 diagnostic (inference only, no training): are the BatchNorm running statistics stored in each saved checkpoint
consistent with the real activation statistics of the training recordings?

For every fold of cnn_only, cnn_only/supplementary_patience25 and fusion_model, compare each BatchNorm layer's stored moving
variance with the true variance of its input on the fold's training recordings. A ratio >> 1 means the checkpoint was
saved before the running statistics had converged. Writes results/model_development/evaluation/cnn_batchnorm_statistics_diagnostic.csv
"""
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
import model_data as md, model_dev as mdl
from tensorflow import keras
meta=md.load_recordings(); maps=md.class_maps(meta); specs=md.load_spectrograms(); ids=list(meta.recording_id)
yr,ya=md.encode_labels(meta,maps)
ctx={"specs":specs,"y_region":pd.Series(yr,index=ids),"y_accent":pd.Series(ya,index=ids),"X":None}
folds=md.make_folds(meta); rows=[]
for run,kind in (("cnn_only","cnn"),("cnn_only/supplementary_patience25","cnn"),("fusion_model","fusion")):
    hist=pd.read_csv(Path("results/model_development")/run/"history_all_folds.csv")
    for f in folds:
        d=Path("results/model_development")/run/f"fold_{f['fold']}"
        pr=mdl.AccentPredictor(d); m=pr.model
        sp=np.stack([md.fit_crop(md.normalise_spec(specs[r],pr.sc["spec"]),md.T_FRAMES) for r in f["train"]])[...,None].astype("float32")
        best=int(hist[hist.fold==f["fold"]].loc[lambda x:x.val_loss.idxmin()].epoch)
        for k,(conv,bn) in enumerate((("conv1_32","conv1_bn"),("conv2_64","conv2_bn"),("conv3_128","conv3_bn")),1):
            sub=keras.Model(m.inputs[0] if kind=="cnn" else m.get_layer("spectrogram").output, m.get_layer(conv).output)
            a=sub.predict(sp,verbose=0,batch_size=5); mu=a.mean((0,1,2)); var=a.var((0,1,2))
            L=m.get_layer(bn); mm,mv=L.moving_mean.numpy(),L.moving_variance.numpy()
            rows.append(dict(run=run,fold=f["fold"],best_epoch=best,bn_layer=bn,median_var_ratio_running_over_true=float(np.median(mv/var)),
                             median_abs_mean_shift_in_sd=float(np.median(np.abs(mu-mm)/np.sqrt(var)))))
        keras.backend.clear_session()
df=pd.DataFrame(rows); df.to_csv("results/model_development/evaluation/cnn_batchnorm_statistics_diagnostic.csv",index=False)
pd.set_option("display.width",200)
print(df.groupby(["run","bn_layer"])[["median_var_ratio_running_over_true","median_abs_mean_shift_in_sd"]].median().round(2))
print(df[df.bn_layer=="conv3_bn"][["run","fold","best_epoch","median_var_ratio_running_over_true","median_abs_mean_shift_in_sd"]].round(2).to_string(index=False))
