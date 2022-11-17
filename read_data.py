import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

import py_neuromodulation as pynm

from py_neuromodulation import nm_IO
import pickle
import mat73

def plot_single_trial(trial_idx: int):

    time_norm_trial = times_all_trials[trial_idx] - times_all_trials[trial_idx][0]
    dat_single_trial = stats.zscore(trials[trial_idx], axis=1)

    ecog_idx = [idx for idx, ch in enumerate(ch_types) if ch == "ecog"]

    plt.imshow(dat_single_trial[ecog_idx, :], aspect="auto")
    plt.colorbar()
    plt.yticks(np.arange(len(ecog_idx)), np.array(ch_names)[ecog_idx])
    plt.xticks(
        np.arange(0, time_norm_trial.shape[0], time_norm_trial.shape[0]/10),
        np.round(np.arange(0, time_norm_trial[-1], time_norm_trial[-1]/10), 2)
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Channels")
    plt.clim(-2, 2)

def write_label_dict(sub_str: str = "3004"):

    PATH_OUT = '/mnt/4TB/timon/OUT_VOL_DECODE'
    PATH_OUT = r"X:\Users\timon\OUT_VOL_DECODE"
    
    PLT_HIST = False
    #PATH_DATA = f'/mnt/Nexus/DBS/DBS{sub_str}/Preprocessed Data/FieldTrip/DBS{sub_str}_ft_raw_filt_trial_denoised.mat'
    PATH_DATA = f"Z:\\DBS\DBS{sub_str}\\Preprocessed Data\\FieldTrip\\DBS{sub_str}_ft_raw_filt_trial_denoised.mat"
    #dat = nm_IO.loadmat(PATH_DATA)
    data_dict = mat73.loadmat(PATH_DATA)

    d = nm_IO._check_keys(data_dict)
    fs = float(d["D"]["fsample"])
    ch_names = [c[0] for c in d["D"]["label"]]
    ch_types = []
    for c in ch_names:
        if "ecog" in c:
            ch_types.append("ecog")
        else:
            ch_types.append("misc")

    times_all_trials = d["D"]["time"]
    trials = d["D"]["trial"]

    #PATH_ANNOT = f'/mnt/Nexus/DBS/DBS{sub_str}/Preprocessed Data/Sync/annot/DBS{sub_str}_produced_phoneme.txt'
    PATH_ANNOT = f"Z:\\DBS\DBS{sub_str}\\Preprocessed Data\\Sync\\annot\\DBS{sub_str}_produced_phoneme.txt"

    annot = pd.read_csv(PATH_ANNOT, sep="\t")


    if PLT_HIST is True:
        # PLOT THE VOLUME
        plt.subplot(121)
        annot["rms_audio_p"].plot.hist(bins=50)
        plt.xlabel("RMS_audio_p")
        plt.title("Speech volume")

        plt.subplot(122)
        annot["rms_audio_p"].apply(lambda x: 10*np.log10(x)).plot.hist(bins=50)
        plt.xlabel("RMS_audio_p [dB]")
        plt.title("Speech volume [dB]")
        plt.tight_layout()

        # idea: predict 'stim' column
        # idea: concatenate the artifact rejected epochs
        # start with the middle of the first PE, extract data till the middle of the second one and so forth

    time_concat = []
    dat_concat = []

    # those are discontinuous events!
    # make a continuous stream! 

    for idx, time in enumerate(d["D"]["time"][:-1]):

        time_middle = time[int(time.shape[0]/2)]
        time_onset = time_middle - 1
        time_end = time_middle + 1

        idx_use = np.where(np.logical_and(time > time_onset, time < time_end))[0]
        time_idx = time[idx_use]
        dat_use = d["D"]["trial"][idx][:, idx_use]
        
        time_concat.append(time_idx)
        dat_concat.append(dat_use)

    t_c = np.concatenate(time_concat, axis=0)
    d_c = np.concatenate(dat_concat, axis=1)
    np.save(os.path.join(PATH_OUT, f't_c_DBS{sub_str}.npy'), t_c) 
    np.save(os.path.join(PATH_OUT, f'd_c_DBS{sub_str}.npy'), d_c)

    t_c = np.load(os.path.join(PATH_OUT, f't_c_DBS{sub_str}.npy'))
    d_c = np.load(os.path.join(PATH_OUT, f'd_c_DBS{sub_str}.npy'))

    # now check in the annot table which stim stim belongs to which time_segment

    label = np.empty(t_c.shape[0], dtype=np.object)
    volume_db = np.empty(t_c.shape[0], dtype=np.object)

    for idx, row in annot.iterrows():
        idx_set = np.where(np.logical_and(t_c > row["starts"], t_c <= row["ends"]))[0]
        if idx_set.shape[0] == 0:
            continue
        else:
            label[idx_set] = row["stim"]
            volume_db[idx_set] = 10*np.log10(row["rms_audio_p"])


    d_out = {
        "data": d_c,
        "time": t_c,
        "stimulus_class": label,
        "volume": volume_db,
        "ch_names" : ch_names,
        "ch_types" : ch_types
    }


    with open(os.path.join(PATH_OUT, f'comb_out_DBS{sub_str}.p'), 'wb') as handle:
        pickle.dump(d_out, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #plt.hist(d_out["volume"][d_out["volume"] != np.array(None)], bins=50)
