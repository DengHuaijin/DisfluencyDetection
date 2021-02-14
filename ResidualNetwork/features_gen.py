import os
import sys
import math
import pickle
import numpy as np
import librosa
import scipy.io.wavfile as wave

from Config import Config
    
wavdir = Config.WAV_PATH
feature_pickle_dir = Config.SPEC_FEATURE_PATH 

def pickle_dump(obj, filename):
    f = open(filename, "wb")
    pickle.dump(obj, f)
    f.close()

def spectrogram_gen():

    window_size = 0.025
    window_stride = 0.01
    sample_freq = 16000
    window_fn = np.hamming
    num_features = 256

    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)
    num_fft = 2**math.ceil(math.log2(window_size*sample_freq))
    print(num_fft)
    sys.exit(0)
    total = len(os.listdir(wavdir))

    for index, i in enumerate(os.listdir(wavdir)):
        print("{} / {}".format(index+1, total), end = "\r")
        sample_freq, signal = wave.read(os.path.join(wavdir, i))
        # normalize
        signal = signal.astype(np.float32)
        signal = signal * (1.0 / np.max(np.abs(signal) + 1e-5))
    
        powspec = np.square(np.abs(librosa.core.stft(
            signal, n_fft = num_fft,
            hop_length = n_window_stride, 
            win_length = n_window_size,
            center = True,
            window = window_fn)))

        powspec[powspec < 1e-30] = 1e-30
        features = 10 * np.log10(powspec.T)
        
        # remove high frequency
        # features = features[:, :num_features]
        mean = np.mean(features, axis = 0)
        std_dev = np.std(features, axis = 0)
        features = (features - mean) / std_dev

        pickle_dump(features, os.path.join(feature_pickle_dir, i.split(".w")[0] + ".cpickle"))
        # print(sample_freq, num_fft, features.shape)
        # sys.exit(0)
        
if __name__ == "__main__":
    spectrogram_gen()
