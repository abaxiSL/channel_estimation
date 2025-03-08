import numpy as np
import matplotlib.pyplot as plt
from utils import (get_detection_preamble, get_channel_est_preamble
, add_noise, pad_signal, detect_signal_start_cross_correlation, estimate_channel)

f_low = 20
f_high = 20000
skip_freqs = 1000
sr = 44000
preamble_duration_ms = 100
det_preamble_sample_len = 100
det_preamble_reps = 10
chan_est_preamble = get_channel_est_preamble(f_low, f_high, skip_freqs, preamble_duration_ms, sr)
detection_preamble = get_detection_preamble(chan_est_preamble, det_preamble_reps, det_preamble_sample_len)

full_tx = np.concatenate([detection_preamble, chan_est_preamble])
full_tx = pad_signal(full_tx, 100, 100)
noisy_tx = add_noise(full_tx, snr_db=3)
channel = [0.6, 0.11]

rx = np.convolve(noisy_tx, channel)
det = detect_signal_start_cross_correlation(rx, detection_preamble)
det_signal = rx[det+det_preamble_sample_len:]
est_channel = estimate_channel(det_signal, chan_est_preamble)
print(est_channel)

