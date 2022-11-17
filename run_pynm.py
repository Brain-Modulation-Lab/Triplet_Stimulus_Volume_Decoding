import pickle
import read_data

import os

import numpy as np
from multiprocessing import Pool

from py_neuromodulation import (
    nm_analysis,
    nm_define_nmchannels,
    nm_IO,
    nm_stream_offline,
    nm_plots
)

PATH_OUT = '/mnt/4TB/timon/OUT_VOL_DECODE'
PATH_OUT = r"X:\Users\timon\OUT_VOL_DECODE"

def run_sub(sub_):
    #sub_ = str(sub)
    #read_data.write_label_dict(sub_)
    with open(os.path.join(PATH_OUT, f'comb_out_DBS{sub_}.p'), 'rb') as handle:
        d_out = pickle.load(handle)

    data = d_out['data'] # voltage values for our simulated data
    channels = d_out['ch_names'] # array with all channel names
    ch_types = d_out['ch_types']
    label = d_out["volume"]
    sfreq = 1000  # sampling frequency

    channels.append("volume")
    ch_types.append("misc")

    ch_names = list(channels)

    data = np.append(data, np.expand_dims(label, 0), axis=0)

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference="default",
        bads=None,
        new_names="default",
        used_types=["ecog"],
        target_keywords=["volume"]
    )

    stream = nm_stream_offline.Stream(
        settings=None,
        nm_channels=nm_channels,
        verbose=False,          # Change here if you want to see the outputs of the run
    )

    stream.reset_settings()

    stream.settings['features']['fft'] = True
    # INIT Feature Estimation Time Window Length and Frequency
    stream.settings[
        "sampling_rate_features_hz"
    ] = 100  # features are estimated every 10s
    stream.settings[
        "segment_length_features_ms"
    ] = 1000  # the duration of 10s is used for feature estimation
    stream.settings["fft_settings"]["kalman_filter"] = False

    stream.init_stream(
        sfreq=sfreq,
        line_noise=60,
    )

    try:
        stream.run(
            data=data.astype(complex).real,
            folder_name=sub_,
            out_path_root=PATH_OUT,
        )
    except Exception:
        print(f"could not run {sub_} shape mismatch")

if __name__ == "__main__":

    

    subs = np.arange(3001, 3033, 1) # START WITH 3001
    #read_data.write_label_dict(subs[0])
    #for sub in subs:
    #    sub_ = str(sub)
        #if os.path.exists(os.path.join(PATH_OUT, f'comb_out_DBS{sub_}.p')) is False:
    #    try:
    #        read_data.write_label_dict(sub_)
    #    except:
    #        print(f"could not read {sub_}")
    subs_available = [i[len("comb_out_DBS"):len("comb_out_DBS")+4] for i in os.listdir(PATH_OUT) if i.startswith("comb_out_DBS")]
    pool = Pool(processes=33)
    pool.map(run_sub, subs)
    #for sub in subs:
    #    run_sub(sub)
