import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from util import io_ops


def zero_crossings(a):
	positive = a > 0
	return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]


def test(file_path):
	signal, sr, channels = io_ops.read_file(file_path)
	crossings = zero_crossings(signal[:, 0])
	deltas = np.diff(crossings)
	smoothing = 1001
	deltas = savgol_filter(deltas, smoothing, 2)
	# deltas = uniform_filter1d(deltas, size=smoothing)
	# print(deltas)
	# with sf.SoundFile("test_raw.wav", 'w+', sr, 1, subtype='FLOAT') as outfile:
	# 	outfile.write( signal )
	plt.figure()
	plt.plot(crossings[:len(deltas)] / sr, deltas, label="crossings")
	# plt.plot(rms_ref, label="rms_ref")
	plt.legend(frameon=True, framealpha=0.75)
	plt.show()

# test("C:/Users/arnfi/Music/1-02 Real Love [Piano-Demo, Take 1].flac")
test("C:/Users/arnfi/Music/1-02 Real Love [Piano-Demo, Take 1] bp2.wav")
