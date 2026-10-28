import logging
from inspect import isclass

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


class BoundsBrush:

	@classmethod
	def update_selection_widget(cls, trail, spec_canvas):
		if trail is not None:
			a = spec_canvas.px_to_spectrum(trail[0])
			b = spec_canvas.px_to_spectrum(trail[-1])
			l, r = sorted((a[0], b[0]))
			b, t = sorted((a[1], b[1]))
			verts = []
			verts.append((l, b, -2.0))
			verts.append((l, t, -2.0))
			verts.append((r, t, -2.0))
			verts.append((r, b, -2.0))
			verts.append((l, b, -2.0))
			spec_canvas.selection_box.update_coords(np.array(verts))

class LivePeakBrush:
	# todo - make this parametric from GUI
	MAX_LEN = 0.5  # sec for the trace

	@classmethod
	def update_selection_widget(cls, trail, spec_canvas):
		if trail is not None:
			current_pos = trail[-1]
			pos = spec_canvas.px_to_spectrum(current_pos)
			t = pos[0]
			f = pos[1]
			frame_0 = spec_canvas.time_2_frame(t)
			tolerance = spec_canvas.props.tracing_widget.tolerance / 12
			# grow forward from mouse position
			old_freq = f
			frame = frame_0
			spec = spec_canvas.spectra[0]
			spec_data = spec.fft_storage[spec.key]
			deltas = []
			verts = []
			cls.follow_track(deltas, frame, frame_0, +1, old_freq, spec_canvas, spec_data, tolerance, verts)
			cls.follow_track(deltas, frame, frame_0, -1, old_freq, spec_canvas, spec_data, tolerance, verts)

			# print(deltas, max(deltas))
			spec_canvas.selection_box.update_coords(np.array(verts))

	@classmethod
	def follow_track(cls, deltas, frame, frame_0, inc, old_freq, spec_canvas, spec_data, tolerance_oct, verts):
		max_len = cls.MAX_LEN / 2.0 / (spec_canvas.hop / spec_canvas.sr)
		# the tolerance is around the center so take half of it
		half_tolerance_oct = tolerance_oct / 2
		while True:
			# prepare borders for sampling next fft
			f_low, f_high = cls.freq_plus_tolerance(old_freq, tolerance_oct)
			b_low, b_high = cls.get_bin_limits(f_low, f_high, spec_canvas)

			freq = cls.get_peak(spec_data, spec_canvas, frame, b_low, b_high, allow_window=False)
			# log2 frequency delta
			# this is just a frequency step, irrespective of time, to prevent jumping between
			d_f = abs((np.log2(freq) - np.log2(old_freq)))
			deltas.append(d_f)
			if frame != frame_0 and d_f > half_tolerance_oct:
				break
			if abs(frame - frame_0) > max_len:
				break
			if inc>0:
				verts.append((spec_canvas.frame_2_time(frame), freq, -2.0))
			else:
				verts.insert(0, (spec_canvas.frame_2_time(frame), freq, -2.0))
			frame += inc
			old_freq = float(freq)


class EdgeBrush:

	@classmethod
	def update_selection_widget(cls, trail, spec_canvas):
		if trail is not None:
			verts = []
			tolerance = spec_canvas.props.tracing_widget.tolerance / 12
			for v in trail:
				p = spec_canvas.px_to_spectrum(v)
				if p is not None:
					logfreq = np.log2(p[1])
					f_low = np.power(2, (logfreq - tolerance))
					verts.append((p[0], f_low, -2.0))
			for v in reversed(trail):
				p = spec_canvas.px_to_spectrum(v)
				if p is not None:
					logfreq = np.log2(p[1])
					f_low = np.power(2, (logfreq + tolerance))
					verts.append((p[0], f_low, -2.0))
			if verts:
				verts.append(verts[0])
			spec_canvas.selection_box.update_coords(np.array(verts))


# todo: create a hann window in log space?
# or just lerp 0, 1, 0 for b_low, i, b_high

class Track:

	tooltip = ""
	# minimal amount of bins to consider
	min_bins = 4

	def __init__(self, spec_canvas, spectrum, signal, trail, hop, sr, tolerance_st=1, adaptation_mode="Linear",
				 dB_cutoff=75):

		# parameters of the fourier transform that was used
		self.hop = hop
		self.sr = sr
		self.spec_canvas = spec_canvas
		self.spectrum = spectrum
		self.signal = signal
		self.fft_freqs = fourier.fft_freqs(spec_canvas.effective_fft_size, sr)

		# start and stop reading the FFT data here, unless...
		self.frame_0 = 0
		self.num_bins, self.frame_1 = self.spectrum.shape

		trail = self.pick_trail(trail)
		# prepare the given frequency trail and build the output data
		self.sample_trail(trail)

		# we define the tolerance in semitones
		# on log2 scale, one semitone is 1/12 (of an octave = 1)
		self.tolerance = tolerance_st / 12

		self.trace()
		# post-pro, remove NANs
		interp_nans(self.freqs)

	def trace(self):
		pass

	def sample_trail(self, trail):
		trail = np.array(trail)
		trail = trail[trail[:, 0].argsort()]
		times_raw = [d[0] for d in trail]
		freqs_raw = [d[1] for d in trail]

		self.ensure_frames(times_raw[0], times_raw[-1])
		self.times = np.linspace(self.spec_canvas.frame_2_time(self.frame_0), self.spec_canvas.frame_2_time(self.frame_1), self.frame_1 - self.frame_0)
		# this is both the input, i.e. drawn frequency curve, and the output - the freqs are altered in place
		self.freqs = np.interp(self.times, times_raw, freqs_raw)

	def ensure_frames(self, t0, t1):
		# we have specified start and stop times, which is the usual case
		if t0:
			# make sure we force start and stop at the ends!
			self.frame_0 = max(self.frame_0, self.spec_canvas.time_2_frame(t0))
		if t1:
			self.frame_1 = min(self.frame_1, self.spec_canvas.time_2_frame(t1))
		if self.frame_0 == self.frame_1:
			logging.warning("No point in tracing just one FFT")

	@classmethod
	def get_bin_limits(cls, f_low, f_high, spec_canvas):
		"""Turn boundary frequencies into valid bin indices, enforcing a minimal bin width"""
		# Just to be sure: clamp to valid frequency range
		f_low = max(1.0, f_low)
		f_high = min(spec_canvas.sr / 2, f_high)
		# make sure that the bins are far enough apart
		b_low = spec_canvas.freq_2_bin(f_low)
		b_high = spec_canvas.freq_2_bin(f_high)
		while (b_high - b_low) < cls.min_bins:
			b_low -= 1
			b_high += 1
		return b_low, b_high

	@classmethod
	def freq_plus_tolerance(cls, freq, tolerance_oct):
		# so take our input frequency, log2 it, +- tolerance (in octaves)
		# and take power of two of result to go back to Hz
		logfreq = np.log2(freq)
		f_low = np.power(2, (logfreq - tolerance_oct))
		f_high = np.power(2, (logfreq + tolerance_oct))
		return f_low, f_high

	@classmethod
	def get_peak(cls, spectrum, spec_canvas, frame_i, b_low, b_high, allow_window=False):
		fft_frame = spectrum[:, frame_i]
		fft_clip = fft_frame[b_low:b_high]
		window_len = b_high - b_low
		if window_len > 4 and allow_window:
			window = np.hanning(window_len)
		else:
			window = np.ones(window_len)
		peak_in_clip = np.argmax(fft_clip * window)
		peak_in_frame = b_low + peak_in_clip
		# make sure i_raw is really a peak and not just a border effect
		if cls.is_peak(fft_frame, peak_in_frame):
			# do quadratic interpolation to get more accurate peak
			peak_in_frame, amp_db = parabolic(fft_frame, peak_in_frame)
		return spec_canvas.bin_2_freq(peak_in_frame)

	@staticmethod
	def is_peak(fft_frame, peak_i):
		"""Returns true if the peak is an actual peak, i.e. its amplitude surpasses its neighboring bins"""
		return fft_frame[peak_i-1] < fft_frame[peak_i] > fft_frame[peak_i+1]

	def track_towards(self, local_frame_i, raw_freq):
		f_low, f_high = self.freq_plus_tolerance(raw_freq, self.tolerance)
		b_low, b_high = self.get_bin_limits(f_low, f_high, self.spec_canvas)
		return self.get_peak(self.spectrum, self.spec_canvas, self.frame_0+local_frame_i, b_low, b_high, allow_window=False)

	def pick_trail(self, trail):
		return trail


def fit_sin(tt, yy, assumed_freq=None):
	"""Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc\""""
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

	return res["amp"], res["omega"], res["phase"], 0


class PeakTracker(Track, EdgeBrush):
	name = 'Peak'
	tooltip = "Follows the mouse input to the loudest peak frequency"

	def trace(self):
		# this is mostly driven from the user input and never strays from it
		for local_frame_i, raw_freq in enumerate(self.freqs):
			# overwrite with new result
			self.freqs[local_frame_i] = self.track_towards(local_frame_i, raw_freq)


class PeakTrackTracker(Track, LivePeakBrush):
	name = 'Magnetic'
	tooltip = "Grows the track from the mouse position"

	def pick_trail(self, trail):
		return self.spec_canvas.selection_box.visuals[0].pos


class ZeroCrossingTracker(Track, BoundsBrush):
	name = 'Zero-Crossing'
	tooltip = "Track the distance between zero-crossings of the waveform. Good for flutter detection of clean signals"

	def trace(self):

		# bandpass the signal to the range of the selection + tolerance
		# todo - instead, create a spectral mask and apply it?
		f_low, _ = self.freq_plus_tolerance(np.min(self.freqs), self.tolerance)
		_, f_high = self.freq_plus_tolerance(np.max(self.freqs), self.tolerance)
		s_0 = int(self.times[0] * self.sr)
		s_1 = int(self.times[-1] * self.sr)
		filtered_sig = filters.butter_bandpass_filter(self.signal[s_0:s_1, 0], f_low, f_high, self.sr, order=3)

		# detect zero-crossings
		crossings = zero_crossings(filtered_sig)
		deltas = np.diff(crossings)
		deltas = deltas.astype(np.float32)

		# the zero crossings are pulse density modulated
		# low pass to get to pulse code modulated pitch signal
		# filter the phase based on the frequency
		size = int(self.sr / 100 / np.mean(deltas))
		padded = np.pad(deltas, size, mode='reflect')
		win_sq = get_window("hann", size)
		deltas_conv = np.convolve(padded, win_sq / size * 2, mode="same")[size:-size]

		# transform to a regularly sampled pitch curve
		self.freqs[:] = np.interp(self.times, crossings[:len(deltas_conv)] / self.sr + self.times[0], self.sr / 2 / deltas_conv)


class PartialsTracker(Track, BoundsBrush):
	name = 'Partials'

	# super slow
	# https://forge-2.ircam.fr/colas/fast-partial-tracking

	# librosa piptrack - relatively poor
	# https://github.com/librosa/librosa/blob/86275a8949fb4aef3fb16aa88b0e24862c24998f/librosa/core/pitch.py#L165

	# todo? investigate polyphonic pitch tracking algortithms
	# https://www.jordipons.me/estimating-pitch-in-polyphonic-music/

	# todo
	# M. Lagrange, S. Marchand, and J. B. Rault, “Using linear prediction to enhance the tracking of partials,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP’04) , May 2004, vol. 4, pp. 241–244.

	# 2003 enhanced partial tracking - good!

	def trace(self):
		import librosa
		fl = np.min(self.freqs)
		fu = np.max(self.freqs)
		# pitches, magnitudes = librosa.piptrack(S=np.array(self.spectrum[self.frame_0:self.frame_1]), sr=self.sr, n_fft=self.spec_canvas.effective_fft_size, hop_length=self.hop, fmin=150.0, fmax=4000.0, threshold=0.1)
		pitches, magnitudes = librosa.piptrack(y=self.signal[:, 0], sr=self.sr, n_fft=self.spec_canvas.effective_fft_size, hop_length=self.hop, fmin=fl, fmax=fu, threshold=0.15)
		# plt.imshow(pitches[:, :], aspect="auto", interpolation="nearest", origin='lower')
		plt.yscale('symlog')
		plt.pcolormesh(pitches)
		plt.show()


class FreehandTracker(Track, EdgeBrush):
	name = 'Freehand Draw'

	def trace(self):
		pass

class PointDrawTracker(Track):
	name = 'Point Draw'
	tooltip = "Place control points with Ctrl+Click, drag them to shape the curve, then Apply"

	def trace(self):
		pass

	@classmethod
	def update_selection_widget(cls, trail, spec_canvas):
		# control points are placed on press, so there is no brush preview
		pass
class CorrelationTracker(Track, BoundsBrush):
	name = 'Correlation'
	tooltip = "Compare the spectra for each segment and track the offsets between"

	def trace(self):
		f_low = min(self.freqs)
		f_high = max(self.freqs)
		b_low, b_high = self.get_bin_limits(f_low, f_high, self.spec_canvas)
		num_freq_samples = (b_high - b_low) * 4

		log_fft_freqs = np.log2(self.fft_freqs[b_low:b_high])

		linspace_fft_freqs = np.linspace(log_fft_freqs[0], log_fft_freqs[-1], num_freq_samples)

		# create a new array to store FFTs resampled on a log2 scale
		resampled = np.ones((num_freq_samples, len(self.freqs) + 1), )
		for i in range(len(self.freqs)):
			interpolator = scipy.interpolate.interp1d(log_fft_freqs, self.spectrum[b_low:b_high, i], kind='quadratic')
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

		log_mean_freq = np.log2((f_low + f_high) / 2)
		# convert to scale and from log2 scale
		np.power(2, (log_mean_freq + speed), self.freqs)


class SineRegression(Track, BoundsBrush):
	name = 'Sine Regression'
	tooltip = "Perform a regression on an area of the master speed curve to yield a sine fit"

	def trace(self):
		pass



def zero_crossings(a):
	positive = a > 0
	return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]


wow_detectors = {}
for symbol, value in dict(locals()).items():
	if isclass(value) and value != Track and issubclass(value, Track):
		wow_detectors[value.name] = value
