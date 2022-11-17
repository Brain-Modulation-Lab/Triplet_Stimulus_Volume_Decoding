import py_neuromodulation as nm
import xgboost
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_stats
)
from sklearn import metrics, model_selection, preprocessing
from sklearn import linear_model
import json
import matplotlib.pyplot as plt
import numpy as np
import PSID
from scipy import stats
import pandas as pd
import seaborn as sb

PATH_OUT = "/home/timonmerk/Document/RUN_SPEECHDECODING_PSID/OUT"
RUN_NAME = "Speech"

import pickle

with open('comb_out.p', 'rb') as handle:
    d_out = pickle.load(handle)

vol = np.array(d_out["volume"])[1000:][::10]
msk_vowels = np.logical_or.reduce((
    d_out["stimulus_class"] == "ee",
    d_out["stimulus_class"] == "oo",
    d_out["stimulus_class"] == "ah"
))[1000:][::10]

msk_vowels = np.logical_or.reduce((
    d_out["stimulus_class"] == "t",
    d_out["stimulus_class"] == "s",
    d_out["stimulus_class"] == "v",
    d_out["stimulus_class"] == "gh"
))[1000:][::10]

msk_vowels = np.logical_or.reduce((
    d_out["stimulus_class"] == "t",
    d_out["stimulus_class"] == "s",
    d_out["stimulus_class"] == "v",
    d_out["stimulus_class"] == "gh",
    d_out["stimulus_class"] == "ee",
    d_out["stimulus_class"] == "oo",
    d_out["stimulus_class"] == "ah"
))[1000:][::10]


stims = d_out["stimulus_class"][1000:][::10][msk_vowels]
vol_use = vol[msk_vowels]
time_use = d_out["time"][1000:][::10][msk_vowels]

feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
)

arr = feature_reader.feature_arr[msk_vowels]

idx_last = 0
epochs = []
vol_ = []

for idx, diff in enumerate(np.diff(time_use)):
    if np.round(diff, 2) != 0.01:

        arr_to_append = arr.iloc[idx_last:idx, :]
        if arr_to_append.shape[0] >=20:
            epochs.append(arr.iloc[idx_last:idx_last+20, :])
            vol_.append(vol_use[idx])

        idx_last = idx

concat_data = np.concatenate([np.expand_dims(np.array(e), axis=0) for e in epochs], axis=0)
vol_ = np.array(vol_)

vol_vowels = vol_
plt.figure(figsize=(5,3), dpi=300)
plt.hist(vol_vowels, bins=20, label="Vowels", alpha=0.5)
plt.hist(vol_, bins=20, label="Consonants",  alpha=0.5)
plt.legend()
plt.xlabel("Volume [dB]")
plt.ylabel("Count")
plt.title("Volume for vowels and consonants")
plt.savefig("volume_modulation.pdf", bbox_inches='tight')

vol_high = vol_> np.percentile(vol_, 75)
vol_low = vol_< np.percentile(vol_, 25)

arr_vh = np.nanmean(concat_data[vol_high, :, :], axis=0)
arr_vl = np.nanmean(concat_data[vol_low, :, :], axis=0)

#arr_vh = np.nanstd(concat_data[vol_high, :, :], axis=0)
#arr_vl = np.nanstd(concat_data[vol_low, :, :], axis=0)

cols_use = [idx for idx, i in enumerate(arr_to_append.columns) if "HFA" in i]
col_names = [i[:8] for idx, i in enumerate(arr_to_append.columns) if "HFA" in i]

plt.figure(figsize=(10, 6), dpi=300)
plt.subplot(121)
plt.imshow(stats.zscore(arr_vh[:, cols_use].T), aspect="auto")
plt.title("High Volume HFA Power")
plt.xticks(np.arange(0, 20, 2), np.arange(0, 200, 20))
plt.xlabel("Time [ms]")
plt.yticks(np.arange(0, len(cols_use), 4), col_names[::4])
plt.colorbar()
plt.subplot(122)
plt.imshow(stats.zscore(arr_vl[:, cols_use].T), aspect="auto")
plt.xticks(np.arange(0, 20, 2), np.arange(0, 200, 20))
plt.yticks(np.arange(0, len(cols_use), 4), col_names[::4])
plt.xlabel("Time [ms]")
plt.title("Low Volume HFA Power")
plt.colorbar()
plt.tight_layout()
plt.savefig("HFA_Vowel_Activateions.pdf", bbox_inches='tight')

mean_time = np.nanmean(concat_data, axis=1)

features = ["theta", "alpha", "low beta", "high beta", "beta", "low gamma", "high gamma", "HFA", "gamma", "fft"]
balanced_per = []
r2_scores_ = []

l_pd_out = []

for f in features:
    cols_use = [idx for idx, i in enumerate(arr_to_append.columns) if f in i]
    out_ = model_selection.cross_val_predict(
        estimator=linear_model.LogisticRegression(),
        X=mean_time[:800, cols_use], y=np.array(vol_>-10)[:800],
        cv=model_selection.KFold(n_splits=5, shuffle=True)
    )

    ba = metrics.balanced_accuracy_score(np.array(vol_>-10)[:800], out_)

    out_ = model_selection.cross_val_predict(
        estimator=linear_model.LinearRegression(),
        X=mean_time[:800, cols_use], y=vol_[:800],
        cv=model_selection.KFold(n_splits=5, shuffle=True)
    )

    l_pd_out.append({
        "spearman's rho" : stats.spearmanr(vol_[:800], out_)[0],
        "balanced_acc" : ba,
        "feature" : f
    })

df = pd.DataFrame(l_pd_out)

plt.figure(figsize=(5, 3), dpi=300)
plt.subplot(121)
plt.title("Volume Regression")
sb.barplot(x="feature", y="spearman's rho", data=df, palette="viridis")
plt.ylim(0, )
plt.xticks(rotation=90)

plt.subplot(122)
plt.title("High vs Low Classification")
sb.barplot(x="feature", y="balanced_acc", data=df, palette="viridis")
plt.ylim(0.5, )
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("PerformancePredictions.pdf", bbox_inches='tight')


# Run now the classification for every channel individually
chs = np.unique([i[:8] for i in arr_to_append.columns])[:-2]
per_ = []
ba_ = []

for ch in chs:

    cols = [idx for idx, i in enumerate(arr_to_append.columns) if ch in i]
    out_ = model_selection.cross_val_predict(
        estimator=linear_model.LinearRegression(),
        X=mean_time[:800, cols], y=vol_[:800],
        cv=model_selection.KFold(n_splits=5, shuffle=True)
    )
    rho_ = stats.spearmanr(vol_[:800], out_)[0]
    per_.append(rho_)

    out_ = model_selection.cross_val_predict(
        estimator=linear_model.LogisticRegression(),
        X=mean_time[:800, cols], y=np.array(vol_>-10)[:800],
        cv=model_selection.KFold(n_splits=5, shuffle=True)
    )

    ba_.append(metrics.balanced_accuracy_score(np.array(vol_>-10)[:800], out_))

plt.figure(figsize=(5, 3), dpi=300)
plt.subplot(121)
plt.bar(np.arange(chs.shape[0]), per_)
plt.xticks(np.arange(chs.shape[0])[::4], chs[::4], rotation=90)
plt.ylabel("Spearman's Rho")
plt.title("Regression")
plt.subplot(122)
plt.bar(np.arange(chs.shape[0]), ba_)
plt.xticks(np.arange(chs.shape[0])[::4], chs[::4], rotation=90)
plt.ylabel("Balanced Accuracy")
plt.ylim(0.5, )
plt.title("Classification")
plt.tight_layout()
plt.savefig("Performance_by_contact.pdf", bbox_inches='tight')


plt.plot(vol_, label='true vol')
plt.plot(out_, label='predicted vol')

metrics.mean_absolute_error(vol_, out_)
# now run the fianl prediction




# check one 

# the only plot that kinda makes sense is to plot averaged features
# for each vowel the volume is the same!

