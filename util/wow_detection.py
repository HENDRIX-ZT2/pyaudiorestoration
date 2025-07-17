import logging

import numpy as np
import scipy.interpolate
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.signal import get_window

from util import fourier, filters
from util.correlation import xcorr, parabolic


def nan_helper(y):
	# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
	return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(y):
	# interpolate any remaining NANs
	nans, x = nan_helper(y)
	y[nans] = np.interp(x(nans), x(~nans), y[~nans])

# super slow
# https://forge-2.ircam.fr/colas/fast-partial-tracking

# librosa piptrack - relatively poor
# https://github.com/librosa/librosa/blob/86275a8949fb4aef3fb16aa88b0e24862c24998f/librosa/core/pitch.py#L165

# todo? investigate polyphonic pitch tracking algortithms
# https://www.jordipons.me/estimating-pitch-in-polyphonic-music/

# todo
# M. Lagrange, S. Marchand, and J. B. Rault, “Using linear prediction to enhance the tracking of partials,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP’04) , May 2004, vol. 4, pp. 241–244.

# 2003 enhanced partial tracking - good!

# todo: create a hann window in log space?
# or just lerp 0, 1, 0 for NL, i, NU

class Track:

	def __init__(self, mode, spectrum, signal, trail, fft_size, hop, sr, tolerance_st=1, adaptation_mode="Linear",
				 dB_cutoff=75):

		# parameters of the fourier transform that was used
		self.fft_size = fft_size
		self.hop = hop
		self.sr = sr
		self.spectrum = spectrum
		self.signal = signal
		self.fft_freqs = fourier.fft_freqs(fft_size, sr)

		# start and stop reading the FFT data here, unless...
		self.frame_0 = 0
		self.num_bins, self.frame_1 = self.spectrum.shape

		# prepare the given frequency trail and build the output data
		self.sample_trail(trail)

		# upper and lower limit of the current band
		self.NL = 0
		self.NU = 0
		# we define the tolerance in semitones
		# on log2 scale, one semitone is 1/12 (of an octave = 1)
		self.tolerance = tolerance_st / 12

		# minimal amount of bins to consider
		self.min_bins = 4

		if mode == "Center of Gravity":
			self.trace_cog()
		elif mode == "Zero-Crossing":
			self.trace_zero_crossing()
		elif mode == "Correlation":
			self.trace_correlation()
		elif mode == "Peak":
			self.trace_peak()
		elif mode == "Peak Track":
			self.trace_peak_track()
		elif mode == "Partials":
			self.trace_partials()
		elif mode == "Freehand Draw":
			pass
		else:
			logging.warning("Not implemented")

		# post-pro, remove NANs
		interp_nans(self.freqs)

	def sample_trail(self, trail):
		# TODO: vectorize this: a[a[:,0].argsort()]
		trail.sort(key=lambda tup: tup[0])
		times_raw = [d[0] for d in trail]
		freqs_raw = [d[1] for d in trail]

		self.ensure_frames(times_raw[0], times_raw[-1])
		self.times = np.linspace(self.frame_0 * self.hop / self.sr, self.frame_1 * self.hop / self.sr,
								 self.frame_1 - self.frame_0)
		# this is both the input, ie. drawn frequency curve, and the output - the freqs are altered in place
		self.freqs = np.interp(self.times, times_raw, freqs_raw)

	def bin_2_freq(self, b):
		return b / self.fft_size * self.sr

	def freq_2_bin(self, f):
		return max(1, min(self.num_bins - 1, int(round(f * self.fft_size / self.sr))))

	def time_2_frame(self, t):
		return int(t * self.sr / self.hop)

	def ensure_frames(self, t0, t1):
		# we have specified start and stop times, which is the usual case
		if t0:
			# make sure we force start and stop at the ends!
			self.frame_0 = max(self.frame_0, self.time_2_frame(t0))
		if t1:
			self.frame_1 = min(self.frame_1, self.time_2_frame(t1))
		if self.frame_0 == self.frame_1:
			logging.warning("No point in tracing just one FFT")

	def set_bin_limits(self, fL, fU):
		"""Turn boundary frequencies into valid bin indices, enforcing a minimal bin width"""
		# Just to be sure: clamp to valid frequency range
		fL = max(1.0, fL)
		fU = min(self.sr / 2, fU)
		# make sure that the bins are far enough apart
		self.NL = self.freq_2_bin(fL)
		self.NU = self.freq_2_bin(fU)
		while (self.NU - self.NL) < self.min_bins:
			self.NL -= 1
			self.NU += 1

	def freq_plus_tolerance(self, freq, tolerance=None):
		# so take our input frequency, log2 it, +- tolerance
		# and take power of two of result to go back to Hz
		if tolerance is None:
			tolerance = self.tolerance
		logfreq = np.log2(freq)
		fL = np.power(2, (logfreq - tolerance))
		fU = np.power(2, (logfreq + tolerance))
		return fL, fU

	def get_peak(self, i, allow_window=False):
		fft_frame = self.spectrum[:, self.frame_0 + i]
		fft_clip = fft_frame[self.NL:self.NU]
		window_len = self.NU-self.NL
		if window_len > 4 and allow_window:
			window = np.hanning(window_len)
		else:
			window = np.ones(window_len)
		peak_in_clip = np.argmax(fft_clip * window)
		# print(peak_i)
		peak_in_frame = self.NL + peak_in_clip
		# make sure i_raw is really a peak and not just a border effect
		if self.is_peak(fft_frame, peak_in_frame):
			# do quadratic interpolation to get more accurate peak
			peak_in_frame, amp_db = parabolic(fft_frame, peak_in_frame)
		return self.bin_2_freq(peak_in_frame)

	@staticmethod
	def is_peak(fft_frame, peak_i):
		"""Returns true if the peak is an actual peak, ie. its amplitude surpasses its neighboring bins"""
		return fft_frame[peak_i-1] < fft_frame[peak_i] > fft_frame[peak_i+1]

	def trace_peak(self):
		# possible approaches
		# start with user tolerance, then decrease down to min_bins and proceed with freq_plus_tolerance(self.freqs[i-1]
		# or
		# work as usual from raw_freq, but tag outliers as NaN, then do a second pass to interp them from the surrounding
		# either lerp or with peak interpolation
		# this is mostly driven from the user input and never strays from it
		for i, raw_freq in enumerate(self.freqs):
			fL, fU = self.freq_plus_tolerance(raw_freq)
			self.set_bin_limits(fL, fU)
			# overwrite with new result
			self.freqs[i] = self.get_peak(i)

	def trace_peak_track(self):
		"""start with user tolerance, then decrease down to min_bins and proceed with freq_plus_tolerance(self.freqs[i-1]"""

		freq = self.freqs[0]
		for i, raw_freq in enumerate(self.freqs):
			if i > 2:
				tolerance = self.tolerance / 2
			else:
				tolerance = self.tolerance
			fL, fU = self.freq_plus_tolerance(freq, tolerance)
			self.set_bin_limits(fL, fU)
			# overwrite with new result
			self.freqs[i] = self.get_peak(i, allow_window=False)

	def trace_zero_crossing(self):
		smoothing = 0.001  # s

		# bandpass the signal to the range of the selection + tolerance
		fL, _ = self.freq_plus_tolerance(np.min(self.freqs))
		_, fU = self.freq_plus_tolerance(np.max(self.freqs))
		# print(fL, fU)
		s_0 = int(self.times[0] * self.sr)
		s_1 = int(self.times[-1] * self.sr)
		filtered_sig = filters.butter_bandpass_filter(self.signal[s_0:s_1, 0], fL, fU, self.sr, order=3)

		crossings = zero_crossings(filtered_sig)
		deltas = np.diff(crossings)
		deltas = deltas.astype(np.float32)
		size = int(round(smoothing * self.sr))

		padded = np.pad(deltas, size, mode='reflect')
		win_sq = get_window("hann", size)
		deltas_conv = np.convolve(padded, win_sq / size * 2, mode="same")[size:-size]
		# print(len(deltas))
		# plt.plot(crossings[:len(deltas)] / self.sr, self.sr / 2 / deltas, label="raw", color=(0, 0, 0, 0.3))
		# plt.plot(crossings[:len(deltas_conv)] / self.sr, self.sr / 2 / deltas_conv, label="deltas_conv")
		# plt.show()

		# transform to a regularly sampled pitch curve
		self.freqs[:] = np.interp(self.times, crossings[:len(deltas_conv)] / self.sr + self.times[0], self.sr / 2 / deltas_conv)

	def trace_partials(self):
		import librosa
		# pitches, magnitudes = librosa.piptrack(S=np.array(self.spectrum[self.frame_0:self.frame_1]), sr=self.sr, n_fft=self.fft_size, hop_length=self.hop, fmin=150.0, fmax=4000.0, threshold=0.1)
		pitches, magnitudes = librosa.piptrack(y=self.signal[:, 0], sr=self.sr, n_fft=self.fft_size, hop_length=self.hop, fmin=150.0, fmax=4000.0, threshold=0.1)
		# plt.plot(pitches, magnitudes)
		plt.imshow(pitches[:100, :], aspect="auto", interpolation="nearest")
		plt.show()

	def COG(self, i):
		# adapted from Czyzewski et al. (2007)
		# 18 calculate the COG with hanned frequency importance
		# error in the printed formula: the divisor also has to contain the hann window
		weighted_magnitudes = np.hanning(self.NU - self.NL) * self.spectrum[self.NL:self.NU, i]
		# 19 convert the center of gravity from log2 space back into linear Hz space
		return 2 ** (np.sum(weighted_magnitudes * np.log2(self.fft_freqs[self.NL:self.NU])) / np.sum(
			weighted_magnitudes))

	def trace_cog(self):
		# adapted from Czyzewski et al. (2007)

		# #calculate the first COG
		# SCoct0  = self.COG( self.frame_0 )

		# #TODO: use uniform tolerance for upper and lower limit?
		# #16a,b
		# #the limits for the first time frame
		# #constant for all time frames
		# #SCoct[0]: the the first COG
		# dfL = SCoct0 - np.log2(fL)
		# dfU = np.log2(fU) - SCoct0

		fL, fU = self.freq_plus_tolerance(self.freqs[0])
		self.set_bin_limits(fL, fU)
		for i in range(len(self.freqs)):
			# 18 calculate the COG with hanned frequency importance
			self.freqs[i] = self.COG(self.frame_0 + i)
			# 15a,b
			# set the limits for the consecutive frame [i+1]
			# based on those of the first frame
			fL, fU = self.freq_plus_tolerance(self.freqs[i])
			self.set_bin_limits(fL, fU)

	def trace_correlation(self):
		fL = min(self.freqs)
		fU = max(self.freqs)
		self.set_bin_limits(fL, fU)
		num_freq_samples = (self.NU - self.NL) * 4

		log_fft_freqs = np.log2(self.fft_freqs[self.NL:self.NU])

		linspace_fft_freqs = np.linspace(log_fft_freqs[0], log_fft_freqs[-1], num_freq_samples)

		# create a new array to store FFTs resampled on a log2 scale
		resampled = np.ones((num_freq_samples, len(self.freqs) + 1), )
		for i in range(len(self.freqs)):
			interpolator = scipy.interpolate.interp1d(log_fft_freqs, self.spectrum[self.NL:self.NU, i], kind='quadratic')
			resampled[:, i] = interpolator(linspace_fft_freqs)

		wind = np.hanning(num_freq_samples)
		# compute the change over each frame
		changes = np.ones(len(self.freqs))
		for i in range(len(self.freqs)):
			# correlation against the next sample, output will be of the same length
			res = xcorr(resampled[:, i]*wind, resampled[:, i + 1]*wind, mode="same")
			# this should maybe hanned before argmax to kill obvious outliers
			i_peak = np.argmax(res)
			# interpolate the most accurate fit
			i_interp, corr = parabolic(res, i_peak)
			changes[i] = (num_freq_samples // 2) - i_interp
		# sum up the changes to a speed curve
		speed = np.cumsum(changes)
		# up to this point, we've been dealing with interpolated "indices"
		# now, we are on the log scale
		# on log scale, +1 means double frequency or speed
		speed = speed / num_freq_samples * (log_fft_freqs[-1] - log_fft_freqs[0])

		log_mean_freq = np.log2((fL + fU) / 2)
		# convert to scale and from log2 scale
		np.power(2, (log_mean_freq + speed), self.freqs)


def adapt_band(freqs, num_bins, freq_2_bin, tolerance, adaptation_mode, i):
	"""
	The goal of this function is, given the last frequency peaks, to return suitable boundaries for the next frequency detection
	
	freqs: the input frequency list
	num_bins: the amount of bins, for limiting
	freq_2_bin: the factor a freq has to be multiplied with to get the corresponding bin
	tolerance: tolerance above and below peak, in semitones (1/12th of an octave)
	adaptation_mode: how should the "prediction" happen?
	
	maxrise: tolerance for absolute rise between a frame, in semitones (redundant? - could be a fraction of tolerance), should this be done here or in postpro?
	"""
	logfreq = np.log2(freqs[i])
	if adaptation_mode == "None":
		# keep the initial limits
		# implement this elsewhere?
		pass
	elif adaptation_mode == "Constant":
		# do nothing to the freq
		pass
	elif adaptation_mode == "Linear":
		# repeat the trend and predict the freq of the next bin to get the boundaries accordingly
		# this should be done in logspace
		if len(freqs) > 1:
			delta = logfreq - np.log2(freqs[i - 2])
			logfreq += delta
	elif adaptation_mode == "Average":
		# take the average of n deltas,
		# and use the preceding freqs as the starting point to avoid outliers?
		logfreqs = np.log2(freqs[max(0, i - 3):i + 1])
		deltas = np.diff(logfreqs)
		logfreq = logfreqs[0]
		if len(deltas): logfreq += np.nanmean(deltas) * len(logfreqs)
	# print(deltas,logfreq)

	fL = np.power(2, (logfreq - tolerance / 12))
	fU = np.power(2, (logfreq + tolerance / 12))
	NL = max(1, min(num_bins - 3, int(round(fL * freq_2_bin))))
	NU = min(num_bins - 2, max(1, int(round(fU * freq_2_bin))))

	if NU - NL > 5:
		window = np.interp(np.arange(NL, NU), (NL, np.power(2, logfreq) * freq_2_bin, NU - 1), (0, 1, 0))
	# print(len(window),window)
	else:
		window = np.ones(NU - NL)
	return NL, NU, window, logfreq


def fit_sin(tt, yy, assumed_freq=None):
	'''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
	# by unsym from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-np
	tt = np.array(tt)
	yy = np.array(yy)
	# new: only use real input, so rfft
	ff = np.fft.rfftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
	fft_data = np.fft.rfft(yy)[1:]
	# new: add an assumed frequency as an optional pre-processing step
	if assumed_freq:
		period = tt[1] - tt[0]
		N = len(yy) + 1
		# find which bin this corresponds to
		peak_est = int(round(assumed_freq * N * period))
		# print("peak_est", peak_est)
		# window the bins accordingly to maximize the expected peak
		win = np.interp(np.arange(0, len(fft_data)), (0, peak_est, len(fft_data)), (0, 1, 0))
		# print("win", win)
		# does this affect the phase?
		fft_data *= win
	peak_bin = np.argmax(np.abs(fft_data)) + 1
	print("peak_bin", peak_bin)
	guess_freq = ff[peak_bin]  # excluding the zero frequency "peak", which is related to offset
	guess_amp = np.std(yy) * 2. ** 0.5
	guess_offset = np.mean(yy)
	# new: get the phase at the peak
	guess_phase = np.angle(fft_data[peak_bin])
	# print("guess_phase",guess_phase)
	guess = np.array([guess_amp, 2. * np.pi * guess_freq, guess_phase, guess_offset])

	# using cosines does not seem to make it better?
	def sinfunc(t, A, w, p, c):     return A * np.sin(w * t + p) + c

	popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
	A, w, p, c = popt
	f = w / (2. * np.pi)
	fitfunc = lambda t: A * np.sin(w * t + p) + c
	return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
			"maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}


def trace_sine_reg(speed_curve, t0, t1, rpm=None):
	"""Perform a regression on an area of the master speed curve to yield a sine fit"""
	# the master speed curve
	times = speed_curve[:, 0]
	speeds = speed_curve[:, 1]

	period = times[1] - times[0]
	# which part to sample
	ind_start = int(t0 / period)
	ind_stop = int(t1 / period)

	try:
		# 33,3 RPM means the period of wow is = 1,8
		# hence its frequency 1/1,8 = 0,55
		assumed_freq = float(rpm) / 60
		print("Source RPM:", rpm)
		print("Assumed Wow Frequency:", assumed_freq)
	except:
		assumed_freq = None

	res = fit_sin(times[ind_start:ind_stop], speeds[ind_start:ind_stop], assumed_freq=assumed_freq)

	# return res["amp"], res["omega"], res["phase"], res["offset"]
	return res["amp"], res["omega"], res["phase"], 0


def zero_crossings(a):
	positive = a > 0
	return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
