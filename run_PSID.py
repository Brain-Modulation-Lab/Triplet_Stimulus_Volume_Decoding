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

# init analyzer
PATH_OUT = "/home/timonmerk/Document/RUN_SPEECHDECODING_PSID/OUT"
RUN_NAME = "Speech"

import pickle

with open('comb_out.p', 'rb') as handle:
    d_out = pickle.load(handle)


feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
)

vol = np.array(d_out["volume"])[1000:][::10]

plt.title("Volume encoded for concatenated -1 till +1 epochs")
plt.plot(np.arange(10000), d_out["volume"][20000:30000])
plt.xlabel("Time [ms]")
plt.ylabel("Volume [dB]")

plt.hist(vol[vol != np.array(None)], bins=50)

feature_reader.feature_arr["volume"] = vol


arr = feature_reader.feature_arr

df_orig = feature_reader.feature_arr.copy()

# first use case: make all labels -50 except classes where the volume passes
# -13 dB

feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
)

vol = np.array(d_out["volume"])[1000:][::10]

feature_reader.feature_arr["volume"] = vol
arr = feature_reader.feature_arr
label = arr["volume"]

label_ = np.nan_to_num(np.array(label)).astype(float)
#label_[label_ < -13] = -50
label_ = np.nan_to_num(label_)
msk_ = label_ != 0
label_use = label_[msk_]
feature_reader.feature_arr = feature_reader.feature_arr[msk_]

feature_reader.label = label_use
features_to_plt = [col for col in feature_reader.feature_arr.columns if "beta" in col]
feature_reader.plot_target_averaged_channel(
    ch=None, features_to_plt=features_to_plt, epoch_len=1, threshold=-13,
    normalize_data=True, show_plot=True
)


# Low Volume
feature_reader = nm_analysis.Feature_Reader(
    feature_dir=PATH_OUT, feature_file=RUN_NAME, binarize_label=False
)
feature_reader.feature_arr["volume"] = vol
arr = feature_reader.feature_arr
label = arr["volume"]

label_ = np.nan_to_num(np.array(label)).astype(float)
#label_[label_ > -13] = -50
label_ = np.nan_to_num(label_)
msk_ = label_ != 0
label_use = label_[msk_]
feature_reader.feature_arr = feature_reader.feature_arr[msk_]

feature_reader.label = label_use

features_to_plt = [col for col in feature_reader.feature_arr.columns if "HFA" in col]

feature_reader.plot_target_averaged_channel(
    ch=None, features_to_plt=features_to_plt, epoch_len=1, threshold=-20,
    normalize_data=True, show_plot=True
)



# Now repeat same for low volume modulation
label = arr["volume"].copy()

label_ = np.nan_to_num(np.array(label)).astype(float)
label_[label_ > -13] = -50
label_ = np.nan_to_num(label_)
label_[label_ == 0] = -50

feature_reader.label = label_

features_to_plt = [col for col in feature_reader.feature_arr.columns if "fft" in col]


feature_reader.plot_target_averaged_channel(
    ch=None, features_to_plt=features_to_plt, epoch_len=1, threshold=-30,
    normalize_data=True, show_plot=True
)


# Check now high vs low speech decoding
label = arr["volume"].copy()
label_ = np.nan_to_num(np.array(label)).astype(float)
label_[label_ > -13] = 1
label_[label_ < -13] = -1
label_ = np.nan_to_num(label_)
msk_use = label_ != 0

X = feature_reader.feature_arr[msk_use]
X_ = X[[col for col in feature_reader.feature_arr.columns if "gamma" in col]]
X_zs = X_.apply(stats.zscore)

y  = np.array(label[msk_use]).astype(float)

out_ = model_selection.cross_val_predict(
    estimator=linear_model.LinearRegression(),
    X=X_zs, y=y, cv=model_selection.KFold(n_splits=5, shuffle=False)
)
print(metrics.mean_absolute_error(y, out_))
print(metrics.r2_score(y, out_))
print(np.corrcoef(y, out_)[0, 1])

plt.plot(np.arange(4633), y[:5000], label="True Volume")
plt.plot(np.arange(4633), out_[:5000], label="Predicted Volume")
plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("Volume [dB]")
plt.title("5 Fold CrossValidation Speech Volume Regression Prediction")
plt.show()


y  = np.array(label[msk_use]).astype(float)>-13

out_ = model_selection.cross_val_predict(
    estimator=linear_model.LogisticRegression(),
    X=X_zs, y=y, cv=model_selection.KFold(n_splits=5, shuffle=False)
)
metrics.balanced_accuracy_score(y, out_)


metrics.balanced_accuracy_score(y, out_)


arr[arr["volume"] < -13]["volume"] = -50


# pick a label and plot averaged features
high_vol = arr[arr["volume"] > -13]
low_vol = arr[arr["volume"] < -13]

feature_reader.feature_arr = low_vol
feature_reader.label 






import pickle

with open('comb_out.p', 'rb') as handle:
    d_out = pickle.load(handle)

stim_class = np.array(d_out["stimulus_class"])[1000:][::10]
z = np.expand_dims(stim_class, axis=1)
z_enc = preprocessing.OrdinalEncoder().fit_transform(z)
idx_dat = z_enc != 7

z_here = z_enc[idx_dat]  # l
arr = feature_reader.feature_arr
col = [i for i in arr.columns if i.startswith("ecog")]#[:20]  "high gamma"
#col = [i for i in arr.columns if "gamma" in i]

y = np.array(arr[col])[idx_dat[:, 0], :]

index_train = 4000 #40000 #
index_end = 4600 #50000 #12600
y_train = stats.zscore(y[:index_train, :], axis=0)  # 
y_test = stats.zscore(y[index_train:index_end, :], axis=0) #stats.zscore(, axis=0)
z_train = np.expand_dims(z_here[:index_train], axis=1)
z_test = np.expand_dims(z_here[index_train:index_end], axis=1)


from sklearn import linear_model
model = linear_model.LogisticRegression()
#model =  xgboost.XGBRegressor()
#model =  xgboost.XGBClassifier()
model.fit(y_train, z_train)
zPred_test = model.predict(y_test)
zPred_train = model.predict(y_train)

plt.figure()
plt.subplot(121)
plt.plot(z_test, label='test label')
plt.plot(zPred_test, label='test predict')
plt.legend()


plt.subplot(122)
plt.plot(z_train, label='train label')
plt.plot(zPred_train, label='train predict')
plt.legend()
plt.show(block=False)


metrics.balanced_accuracy_score(z_test, zPred_test)





# read here now the pic
arr = feature_reader.feature_arr
label = np.array(arr["volume"])

label = np.nan_to_num(label)
label[label == 0] = label[label != 0].min()
label = label.astype(float)

idx_dat = label != label.min()
l = label[idx_dat]
l_lowhigh = np.array(l<-13)*1


z_here = l  # l
n1 = 1# number of syllables
nx = 2
col = [i for i in arr.columns if i.startswith("ecog")]#[:20]  "high gamma"
#col = [i for i in arr.columns if "gamma" in i]

y = np.array(arr[col])[idx_dat, :]

index_train = 4000 #40000 #
index_end = 4600 #50000 #12600
y_train = stats.zscore(y[:index_train, :], axis=0)  # 
y_test = stats.zscore(y[index_train:index_end, :], axis=0) #stats.zscore(, axis=0)
z_train = np.expand_dims(z_here[:index_train], axis=1)
z_test = np.expand_dims(z_here[index_train:index_end], axis=1)


from sklearn import linear_model
model = linear_model.LinearRegression()
#model =  xgboost.XGBRegressor()
#model =  xgboost.XGBClassifier()
model.fit(y_train, z_train)
zPred_test = model.predict(y_test)
zPred_train = model.predict(y_train)

plt.figure()
plt.subplot(121)
plt.plot(z_test, label='test label')
plt.plot(zPred_test, label='test predict')
plt.legend()


plt.subplot(122)
plt.plot(z_train, label='train label')
plt.plot(zPred_train, label='train predict')
plt.legend()
plt.show(block=False)


metrics.accuracy_score(z_test, zPred_test)
metrics.balanced_accuracy_score(z_test, zPred_test)

metrics.r2_score(z_test, zPred_test)






idSys = PSID.PSID(y_train, z_train, nx=nx, n1=n1, i=2)  # nx = 2, n1 = 1, i = 2 # i needs to be at least nx

zPred_test, yPred, xPred_test = idSys.predict(y_test)

zPred_train, yPred, xPred_train = idSys.predict(y_train)



y = np.array(arr.iloc[:, :-2])  # [col]
z = np.array(arr.iloc[:, -1])
#y = np.array(arr.iloc[:1000, :-2])


z = np.expand_dims(z, axis=1)
z_enc = preprocessing.OrdinalEncoder().fit_transform(z)
z_enc = z_enc + 1
z_here = np.nan_to_num(z_enc)


#z_here = np.array(z_enc == 0)*1

#z_enc = preprocessing.OneHotEncoder().fit_transform(z).toarray()
#z_null = np.nan_to_num(z)

#z_enc = np.expand_dims(z, axis=1)
#
#

#idx_use = np.where(z_enc != 0)[0]
#z_use = z_enc[idx_use]
#y_use = y[idx_use]



# one hot encode z
# extract only gamma





import pandas as pd

df = pd.read_csv("/home/timonmerk/Downloads/thalamic_df_2_11.csv")