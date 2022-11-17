import mne

PATH = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS1534/sEEG/EP1142_66766969-a090-4350-ae2c-049a6d27092c - RNS1534-20161121.edf"

raw = mne.io.read_raw_edf(PATH)

raw.plot(block=True)

