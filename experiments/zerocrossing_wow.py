import os

import numpy as np
from matplotlib import pyplot as plt

from util import io_ops
from scipy import signal as dsp

from util.wow_detection import zero_crossings


# https://gist.github.com/sixtenbe/1178136/d52dfaaf987c56bec20bb64d35f3fb35d39e1f80
def detect_crossings(file_path):
	signal, sr, num_channels = io_ops.read_file(file_path)
	# todo bandpass to interesting range
	crossings = zero_crossings(signal[:, 0])
	deltas = np.diff(crossings)
	# distances between samples
	deltas = deltas.astype(np.float32)
	smoothing = 0.001  # s
	size2 = int(round(smoothing * sr))
	size = int(sr / 100 / np.mean(deltas))
	plt.plot(crossings[:len(deltas)] / sr, sr/2/deltas, label="raw", color=(0, 0, 0, 0.3))

	padded = np.pad(deltas, size, mode='reflect')
	win_sq = dsp.get_window("hann", size)
	print(len(deltas), len(win_sq), size2, np.mean(deltas))
	deltas_conv = np.convolve(padded, win_sq/size*2, mode="same")[size:-size]
	# todo transform to a regularly sampled pitch curve
	plt.plot(crossings[:len(deltas_conv)] / sr, sr/2/deltas_conv, label="deltas_conv")

	plt.legend(frameon=True, framealpha=0.75)

# pilot tone with flutter at 4000 Hz
fp = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'samples', "flutter_192.flac"))
detect_crossings(fp)
fp = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'samples', "flutter.flac"))
detect_crossings(fp)

plt.show()