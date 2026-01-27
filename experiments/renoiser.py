import os

import numpy as np
import resampy
from matplotlib import pyplot as plt, mlab

from util import io_ops, fourier
from util.fourier import to_mag
from util.units import to_dB, to_fac

samples_dir = os.path.join(os.path.dirname( __file__ ), '..', 'samples')
# todo - automatically sniff out the FFT settings and alignment that were used to poorly denoise a given input track
signal_path = os.path.join(samples_dir, "nr_signal.wav")
# noise_path = os.path.join(samples_dir, "nr_noise.wav")
noise_path = os.path.join(samples_dir, "nr_noise_eq4.wav")


signal, sr, num_channels = io_ops.read_file(signal_path)
noise, noise_sr, noise_num_channels = io_ops.read_file(noise_path)
srs = (44100,)
# srs = (44100, 48000, 96000)
overlap = 16
gains = (
	# (18.0, -6.0),
	# (18.0, -4.0),
	# (18.0, -2.0),
	# (16.0, -6.0),
	# (16.0, -4.0),
	# (16.0, -2.0),
	# (16.0, 0.0),
	# (12.0, 2.0),
	# (11.0, 5.0),
	(12.0, 3.0),  # pretty good
	# (12.0, 4.0),
	# (10.0, 6.0),
 )  # dB
# fft_sizes = (64, 128, 256, 512, 1024, 2048, 4096)
# fft_sizes = (1024, 2048, )
fft_sizes = (2048, )  # best
# fft_sizes = (1024, 2048, 4096)
# fft_sizes = (2048, 4096)
for fft_size in fft_sizes:
	hop = fft_size // overlap
	# hop = 128
	for sr_trg in srs:
		if sr == sr_trg:
			signal_res = signal
			noise_res = noise
		else:
			signal_res = resampy.resample(signal, sr, sr_trg, axis=0, filter='sinc_window', num_zeros=8)
			noise_res = resampy.resample(noise, sr, sr_trg, axis=0, filter='sinc_window', num_zeros=8)
		n = len(signal_res)
		for gain, overhead in gains:
			y_out = np.empty(signal_res.shape, dtype=signal_res.dtype)
			# take FFT for each channel
			for channel_i in range(num_channels):
				signal_res_pad = fourier.fix_length(signal_res, n + fft_size // 2, axis=0)
				# take FFT for each channel
				fft_signal = np.array(fourier.stft(signal_res_pad[:, channel_i], n_fft=fft_size, step=hop))
				# todo - always take noise from left channel?
				fft_noise = fourier.get_mag(noise_res[:, 0], fft_size, hop, "blackmanharris", zeropad=1)

				db_signal = np.array(to_dB(to_mag(fft_signal)))
				db_noise = to_dB(fft_noise)
				noise_profile = np.average(db_noise, axis=1, keepdims=True)
				# spectrum = np.where(db_signal > (noise_profile + gain), db_signal, db_signal + gain)
				gain_mask = np.where(db_signal > (noise_profile + gain + overhead), 0, gain)
				# plt.imshow(db_signal, aspect='auto', origin='lower', cmap='magma')
				# # plt.plot(noise_profile)
				# plt.title(f"fft: {fft_size} hop: {hop} sr: {sr_trg}")
				fac = to_fac(gain_mask)

				y_out[:, channel_i] = fourier.istft(fft_signal * fac, length=n, hop_length=hop)

			# plt.show()
			io_ops.write_file(signal_path, y_out, sr_trg, num_channels, suffix=f" fft={fft_size}, gain={gain}, overhead={overhead}")
