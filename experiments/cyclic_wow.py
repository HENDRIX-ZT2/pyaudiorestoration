import matplotlib.pyplot as plt
import numpy as np

from util import io_ops
from util.spectrum_flat import spectrum_from_audio_stereo
from util.wow_detection import PeakTracker


def get_avg(logfreq, frames_per_rotation):
	num_views = len(logfreq) // frames_per_rotation
	print(f"Testing cycle length: {frames_per_rotation} frames as {num_views} slices")
	l = num_views * frames_per_rotation
	padding = len(track.freqs) - l
	# print(l, len(track.freqs), padding, logfreq.shape)
	# f = np.pad(track.freqs, (0, padding), 'constant')
	# trim
	f = logfreq[:num_views * frames_per_rotation]
	# f = np.reshape(f, (frames_per_rotation, num_views))
	f = np.split(f, num_views)
	# print(f)
	# padding = len(track.freqs) % window
	# print(len(track.freqs), padding)
	# f = np.pad(track.freqs, (0, padding), 'constant')
	# print(len(f), window, len(f)%window)
	# v = np.split(f, window, axis=0)
	# plt.plot(track.freqs)
	return np.mean(f, axis=0)



file_src = "../samples/cyclic_pilot.wav"
file_src = "../samples/cyclic_pilot+n.wav"

fft_size = 8192 * 2
fft_hop = fft_size // 128
signal, sr, num_channels = io_ops.read_file(file_src)
spectra, sr = spectrum_from_audio_stereo(file_src, fft_size, fft_hop, "L", temporal_mean=False)
spectrum = spectra[0]

rpm = 45.0
tolerance = 0.1  # 10% of expected cycle length
rps = rpm / 60.0
spr= 60.0 / rpm
samples_per_rotation = spr * sr
frames_per_rotation_init = int(samples_per_rotation / fft_hop)

track = PeakTracker(spectrum, 0, [(0.0, 700.0), (len(signal)/sr, 700.0)], fft_size, fft_hop, sr, tolerance_st=10, adaptation_mode="Linear",
				 dB_cutoff=75)
logfreq = np.log2(track.freqs)
d = int(frames_per_rotation_init*tolerance)
results = np.empty((d*2, 2))

for i in range(-d, d):
	# i = 0
	frames_per_rotation = frames_per_rotation_init +i
	avg = get_avg(logfreq, frames_per_rotation)
	# plt.plot(avg, label=f"{frames_per_rotation}")
	delta = np.max(avg) - np.min(avg)
	# print(i, d+i, len(results))
	results[d+i] = (frames_per_rotation, delta)

# plt.plot(results[:, 0], results[:, 1], label="speed delta per cycle length")

best_i = np.argmax(results[:, 1])
frames_per_rotation, delta = results[best_i]
print(f"Best cycle length: {frames_per_rotation} with speed delta: {delta/12} semitones")
cycle_duration = frames_per_rotation * fft_hop /  sr
print(f"Cycle duration: {cycle_duration} s compared to ideal target {spr} s")
print(f"Record was playing at {60.0 / cycle_duration} rpm")

avg = get_avg(logfreq, int(frames_per_rotation))
plt.plot(avg, label=f"best {frames_per_rotation}")
plt.legend()
plt.show()