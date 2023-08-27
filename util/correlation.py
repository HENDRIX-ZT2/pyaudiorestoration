import numpy as np
import scipy
import logging


def xcorr(a, b, mode='full'):
	"""Normalized cross correlation returning correlation in range [-1.0, 1.0]"""
	norm_a = np.linalg.norm(a)
	a = a / norm_a
	norm_b = np.linalg.norm(b)
	b = b / norm_b
	# mostly equivalent to np.correlate(a, b, mode=mode), but can use FFT to speed up
	return scipy.signal.correlate(a, b, mode=mode, method='auto')


def find_delay(a, b, ignore_phase=False, window_name=None):
	"""Calculate the delay between 1D signals a and b"""
	if window_name:
		a *= scipy.signal.get_window(window_name, len(a))
		b *= scipy.signal.get_window(window_name, len(b))
	# correlate both sources
	# scipy differs from numpy: The output is the same size as in1, centered with respect to the ‘full’ output.
	res = xcorr(a, b, mode="same")
	# logging.debug(f"len(a) {len(a)}, len(b) {len(b)}, len(res) {len(res)}")
	# we are not necessarily interested in the largest positive value if the correlation is negative
	if ignore_phase:
		logging.warning("Ignoring phase")
		np.abs(res, out=res)
	# get the index of the strongest correlation
	max_index = np.argmax(res)
	# logging.debug(f"max_index {max_index}")
	# refine the index with interpolation to get the most accurate fit
	i_peak, corr = parabolic(res, max_index)
	logging.debug(f"i_peak {i_peak}")
	# getting delay from the peak index depends on xcorr mode
	# even / odd input lengths magically do not mess with this
	sample_delay = i_peak - len(res) // 2
	return sample_delay, corr


def parabolic(f, x):
	"""Helper function to refine a peak position in an array"""
	xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
	yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
	return xv, yv


def test_delay():
	from matplotlib import pyplot as plt
	delay = 3
	for add in (0, 1):
		len_a = 521
		a = np.arange(0, len_a + add, 1)
		b = np.arange(0, len_a + add, 1) + delay
		is_odd = "odd" if len(a) % 2 else "even"
		sig_a = np.sin(a)
		sig_b = np.sin(b)
		plt.plot(sig_a, label=f"sig_a {is_odd}")
		plt.plot(sig_b, label=f"sig_b {is_odd}")
		print(find_delay(sig_a, sig_b, window_name="hann"))
	plt.legend(frameon=True, framealpha=0.75)
	plt.show()
