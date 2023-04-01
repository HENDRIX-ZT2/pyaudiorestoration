# select time range to correlate
# select number of bands
# plot x: freq, y: phase delay, correlation
import logging

import scipy

from pyaudiorestoration.util.wow_detection import parabolic

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
import numpy as np
import soundfile as sf
from pyaudiorestoration.dropouts_gui import pairwise

from scipy import signal
import matplotlib.pyplot as plt

from pyaudiorestoration.util import filters


def xcorr(a, b, mode='full'):
	norm_a = np.linalg.norm(a)
	a = a / norm_a
	norm_b = np.linalg.norm(b)
	b = b / norm_b
	# return np.correlate(a, b, mode=mode)
	# return scipy.signal.correlate(ref_s*s_window, src_s*s_window, mode='same', method='auto')
	return scipy.signal.correlate(a, b, mode=mode, method='auto')


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
	s_dur = s_end-s_start
	s_window = np.hanning(s_dur)
	# or logspace?
	# band_limits = np.linspace(f_lower, f_upper, num_bands)
	band_limits = np.logspace(np.log2(f_lower), np.log2(f_upper), num=num_bands, endpoint=True, base=2, dtype=np.uint16)
	lags = []
	correlations = []
	band_centers = []
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
		i_peak = np.argmax(res*s_window)
		# interpolate the most accurate fit
		i_interp, corr = parabolic(res, i_peak)
		v = (s_dur // 2) - i_interp
		if corr > min_corr:
			lags.append(v)
			correlations.append(corr)
			band_center = (f_lower_band+f_upper_band)/2
			band_centers.append(band_center)
		else:
			logging.warning(f"Band had too little correlation {corr}")

		#
		# plt.title('Digital filter group delay')
		# plt.plot(ref_s)
		# plt.plot(src_s)
		# plt.ylabel('a')
		# plt.xlabel('t')
		# plt.show()

	plot_corr_lag(band_centers, correlations, lags)


def plot_corr_lag(band_centers, correlations, lags):
	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('frequency [Hz]')
	ax1.set_ylabel('lag [sample]', color=color)
	ax1.plot(band_centers, lags, color=color)
	ax1.tick_params(axis='y', labelcolor=color)
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:blue'
	ax2.set_ylabel('correlation', color=color)  # we already handled the x-label with ax1
	ax2.plot(band_centers, correlations, color=color)
	ax2.tick_params(axis='y', labelcolor=color)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()


fp = "C:/Users/arnfi/Music/The Beatles/Revolver Companion/Rain/rain rhythm sinc az.wav"
ref_ob = sf.SoundFile(fp)
ref_sig = ref_ob.read(always_2d=True, dtype='float32')
get_group_delay(ref_sig[:, 0], ref_sig[:, 1])

