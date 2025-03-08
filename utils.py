import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv


def detect_signal_start_cross_correlation(sig, det_preamble):
    sig = sig/np.abs(np.max(sig))
    det_preamble = det_preamble/np.abs(np.max(det_preamble))
    conv = np.convolve(sig, det_preamble[::-1], mode="valid")
    conv = np.abs(conv)
    # plt.plot(sig)
    # plt.plot(conv)
    # plt.show()
    det = np.argmax(conv)
    return det


def estimate_channel(sig, preamble, max_channel_len=10):
    preamble_len = len(preamble)
    X = np.zeros(shape=(preamble_len, max_channel_len))
    X[:, 0] = preamble
    for i in range(1, max_channel_len):
        X[i:, i] = preamble[:-i]
    # y = X . channel; channel = X^-1 . y
    sig = sig.reshape(-1,1)
    sig = sig[:preamble_len]
    channel = np.dot(pinv(X), sig)
    return channel


def pad_signal(sig, l_pad=None, r_pad=None):
    pad_shape = list(sig.shape)
    if l_pad is not None:
        pad_shape[0] = l_pad
        pad_l = np.zeros(pad_shape)
        sig = np.concatenate([pad_l, sig])
    if r_pad is not None:
        pad_shape[0] = r_pad
        pad_r = np.zeros(pad_shape)
        sig = np.concatenate([sig, pad_r])
    return sig


def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr = 10**(snr_db / 10)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


def get_detection_preamble(x, reps, tot_len):
    unit_len = tot_len//reps
    p = np.argmax(x[unit_len:])
    p += unit_len
    unit = x[p-unit_len//2:1+p+unit_len//2]
    full = []
    for _ in range(reps):
        if np.random.rand()>0.5:
            full.append(unit*-1)
        else:
            full.append(unit)
    full = np.concatenate(full)[:tot_len]
    return full


def get_channel_est_preamble(f_lo, f_hi, skip, duration_ms, sampling_rate):
    preamble_duration_sec = duration_ms/1000
    preamble_num_samples = int(sampling_rate * preamble_duration_sec)
    preamble_freq = np.zeros(sampling_rate)
    preamble_freq[f_lo:f_hi:skip] = 1
    preamble_freq[0] = 1
    preamble_time = np.fft.irfft(preamble_freq, n=sampling_rate)
    preamble_time = preamble_time[:preamble_num_samples]
    return preamble_time
