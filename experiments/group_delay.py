# select time range to correlate
# select number of bands
# plot x: freq, y: phase delay, correlation
import logging
import os

from scipy import signal

from util.correlation import xcorr, parabolic
from util.units import to_dB, to_fac

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
import numpy as np
import soundfile as sf
from dropouts_gui import pairwise

import matplotlib.pyplot as plt

from util import filters, io_ops


def rad_to_Hz(w):
	return w/(2*np.pi)

def Hz_to_w(hz):
	return hz*(2*np.pi)


def get_group_delay(ref_sig, src_sig):
	# first get the time range for both
	# apply bandpass
	# split into pieces and look up the delay for each
	# correlate all the pieces
	# sr = self.sr
	sr = 44100
	f_upper = 2000
	f_lower = 10
	# num = 10
	bandwidth = 45
	num_bands = int((f_upper-f_lower) / bandwidth)

	t0 = 153.2
	t1 = 156.0
	min_corr = 0.6
	s_start = int(t0 * sr)
	s_end = int(t1 * sr)
	s_start = 0
	s_end = len(src_sig)
	s_dur = s_end-s_start
	s_window = np.hanning(s_dur)
	# or logspace?
	# band_limits = np.linspace(f_lower, f_upper, num_bands)
	band_limits = np.logspace(np.log2(f_lower), np.log2(f_upper), num=num_bands, endpoint=True, base=2, dtype=np.uint16)
	lags = []
	correlations = []
	band_centers = []
	magnitudes = []
	for f_lower_band, f_upper_band in pairwise(band_limits):
		logging.info(f"lower {f_lower_band:.1f}, upper {f_upper_band:.1f}, width {f_upper_band-f_lower_band:.1f}")
		ref_s = filters.butter_bandpass_filter(ref_sig[s_start:s_end], f_lower_band, f_upper_band, sr, order=1)
		src_s = filters.butter_bandpass_filter(src_sig[s_start:s_end], f_lower_band, f_upper_band, sr, order=1)
		# plt.plot(ref_s, label="ref_s")
		# plt.plot(src_s, label="src_s")
		# res = np.correlate(ref_s*s_window, src_s*s_window, mode="same")
		# res = scipy.signal.correlate(ref_s*s_window, src_s*s_window, mode='same', method='auto')
		# res = scipy.signal.correlate(ref_s, src_s, mode='same', method='auto')
		# res = xcorr(ref_s*s_window, src_s*s_window, mode='same')
		res = xcorr(ref_s, src_s, mode='same')
		# this should maybe hanned before argmax to kill obvious outliers
		# i_peak = np.argmax(res*s_window)
		i_peak = np.argmax(res)
		# interpolate the most accurate fit
		i_interp, corr = parabolic(res, i_peak)
		v = (s_dur // 2) - i_interp
		# todo - do fft based EQ estimation
		ref_vol = np.sqrt(np.mean(np.square(ref_s)))
		src_vol = np.sqrt(np.mean(np.square(src_s)))
		if corr > min_corr:
			lags.append(v)
			correlations.append(corr)
			band_center = (f_lower_band+f_upper_band)/2
			band_centers.append(band_center)
			magnitudes.append(ref_vol-src_vol)
		else:
			logging.warning(f"Band had too little correlation {corr}")

		#
		# plt.title('Digital filter group delay')
		# plt.plot(ref_s)
		# plt.plot(src_s)
		# plt.ylabel('a')
		# plt.xlabel('t')
		# plt.show()

	plot_corr_lag(band_centers, correlations, lags, magnitudes)


def plot_corr_lag(band_centers, correlations, lags, magnitudes):
	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('frequency [Hz]')
	ax1.set_ylabel('lag [sample]', color=color)
	ax1.semilogx(band_centers, lags, color=color, label=f'measured')
	ax1.tick_params(axis='y', labelcolor=color)
	# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# color = 'tab:blue'
	# ax2.set_ylabel('correlation', color=color)  # we already handled the x-label with ax1
	# ax2.semilogx(band_centers, correlations, color=color)
	# ax2.tick_params(axis='y', labelcolor=color)

	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:green'
	# ax3.set_ylabel('magnitudes', color=color)  # we already handled the x-label with ax1
	# ax3.plot(band_centers, magnitudes, color=color)
	ax3.tick_params(axis='y', labelcolor=color)

	sr = 44100
	for order in range(1, 2):
		freq = 100
		for freq in range(10, 150, 10):
			# analog butter uses rad/s
			# b, a = signal.butter(order, freq / nyq, 'low', analog=False)
			sos = signal.bessel(order, freq, 'low', analog=False, output='sos', fs=sr)
			w, h = signal.freqz_sos(sos)
			# w, h = signal.freqs(b, a, worN=np.logspace(-1, np.log10(Hz_to_w(sr / 2)), 1000))

			# angular frequency = rad / sec
			freqs_hz = rad_to_Hz(w)
			# fig, ax1 = plt.subplots(tight_layout=True)
			# ax1.semilogx(freqs_hz, to_dB(abs(h)), 'C0')
			# ax1.set_ylabel("Amplitude in dB", color='C0')
			# ax1.set(xlabel="Frequency [Hz]")

			phase = np.unwrap(np.angle(h))
			# ax1.semilogx(freqs_hz, -rad_to_Hz(phase) * sr / freqs_hz, label=f'order={order}, f={freq} Hz')
			freqs_hz = w/np.pi*sr/2
			ax1.semilogx(freqs_hz, -rad_to_Hz(phase) * sr / freqs_hz, label=f'order={order}, f={freq} Hz')


	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	ax1.legend()
	plt.show()


#
# # fp = "D:/Users/arnfi/Music/The Beatles/Revolver Companion/Rain/rain rhythm sinc az.wav"
# fp = "C:/Users/arnfi/Desktop/01 Back in the U.S.S.R. wide snip.wav"
# sig, sr, num_channels = io_ops.read_file(fp)
# get_group_delay(sig[:, 0], sig[:, 1])
# for order in range(1, 3):
# 	for freq in range(10, 150, 25):
# 		sos = signal.bessel(order, freq, 'low', analog=False, output='sos', fs=sr)
# 		# sos = signal.butter(2, low, 'low', analog=True, output='sos')
# 		# w, h = signal.freqs(b, a, worN=np.logspace(-1, np.log10(Hz_to_w(44100 / 2)), 1000))
# 		# sig[:, 0] = signal.sosfilt(sos, sig[:, 0])
# 		io_ops.write_file(fp, signal.sosfilt(sos, sig[:, 0]), sr, 1, f"_filt{freq}_o{order}")
#
# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html#scipy.signal.bessel
#


folder = "K:/Musik/The Beatles/Original/The Beatles/The Beatles - CD 3 Esher Demos"
data = [
	# ("01 Back in the U.S.S.R..flac", -9.1, -7.9, ),
	# ("02 Dear Prudence.flac", -8.6, -8.6),
	# ("03 Glass Onion.flac", -6.4, -7.1),  # edits at 0:09, maybe 0:35
	# ("04 Ob-La-Di, Ob-La-Da.flac", -7.5, -9.1),
	# ("05 The Continuing Story of Bungalow Bill.flac", -3.9, -7.4),
	# ("06 While My Guitar Gently Weeps.flac", -10.0, -9.3),
	# ("07 Happiness is a Warm Gun.flac", -4.1,-7.2),
	# ("08 I'm so tired.flac", -4.0, -6.8),
	# ("09 Blackbird.flac", -8.1, -9.0),
	# ("11 Rocky Raccoon.flac", -9.2, -7.4),
	# ("10 Piggies.flac", -5.9, -9.9),
	# ("12 Julia.flac", -7.1, -5.6),
	# ("13 Yer Blues.flac", -11.0, -5.1),
	# ("15 Everybody's Got Something to Hide Except Me and My Monkey.flac", -8.1, -8.2),
	# ("16 Sexy Sadie eq.wav", -6.0, -6.0),
	# ("17 Revolution.flac", -7.2, -7.9),
	# ("18 Honey Pie.flac", -6.6, -6.7),
	# ("21 Junk.flac", -6.4, -8.2),
	("51. Junk (Esher Demo).flac", -6.4, -8.2),
	# ("14 Mother Nature's Son.flac", -7.9, -7.9),
	# ("49. Cry Baby Cry (Esher Demo) res.wav", -6.0, -7.7),
	# ("22 Child of Nature.flac", -6.9, -6.9),
	# ("24 Mean Mr. Mustard.flac", -6.3,-7.2),
	# ("25 Polythene Pam.flac", -6.3,-7.2),
	# ("26 Not Guilty.flac", -8.5,-7.6),
	# ("26 Not Guilty wide.wav", 0, 0),
	# ("20 Sour Milk Sea.flac", -2.9, -7.5),
	# ("27 What's the New Mary Jane.flac", -5.7, -5.5),
]
# negative dB gains from izotope azimuth are the first number, positive the second (always give negative)
for fn, gain_l, gain_r in data:
	fp = os.path.join(folder, fn)
	sig, sr, num_channels = io_ops.read_file(fp)

	L = sig[:, 0] - (sig[:, 1] * to_fac(gain_l))
	R = sig[:, 1] - (sig[:, 0] * to_fac(gain_r))
	io_ops.write_file(fp, L, sr, 1, f"_L_oops")
	io_ops.write_file(fp, R, sr, 1, f"_R_oops")
	# for name, filter_func in (("bessel", signal.bessel), ("butter", signal.butter), ("cheby1", signal.cheby1)):
	for name, filter_func in (("bessel", signal.bessel), ):
		args = {"Wn":50, "btype":'low', "analog":False, "output":'sos', "fs":sr}
		if name == "cheby1":
			args["rp"] = 5.0
		sos1 = filter_func(N=1, **args)
		sos2 = filter_func(N=2, **args)
		L_lp = signal.sosfilt(sos1, L)
		R_lp = signal.sosfilt(sos1, R)
		L_1 = L - (R_lp * to_fac(2.4))
		R_1 = R - (L_lp * to_fac(-9.1))

		L_lp2 = signal.sosfilt(sos2, L)
		R_lp2 = signal.sosfilt(sos2, R)
		L_lp2 = np.roll(L_lp2, -20) # more lag
		R_lp2 = np.roll(R_lp2, 31)
		L_2 = L_1 - (R_lp2 * to_fac(1.8))
		R_2 = R_1 - (L_lp2 * to_fac(5.7))
		io_ops.write_file(fp, L_lp, sr, 1, f"_{name}_L_lp")
		io_ops.write_file(fp, R_lp, sr, 1, f"_{name}_R_lp")
		io_ops.write_file(fp, L_lp2, sr, 1, f"_{name}_L_lp2")
		io_ops.write_file(fp, R_lp2, sr, 1, f"_{name}_R_lp2")
		# io_ops.write_file(fp, L_1, sr, 1, f"_{name}_L1")
		# io_ops.write_file(fp, L_2, sr, 1, f"_{name}_L2")
		# io_ops.write_file(fp, R_1, sr, 1, f"_{name}_R1")
		# io_ops.write_file(fp, R_2, sr, 1, f"_{name}_R2")
