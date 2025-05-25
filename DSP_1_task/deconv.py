import scipy
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def deconvolve(x, y):
    y = y[:len(x)]
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    H = Y/X
    h = np.fft.irfft(H)
    return h