import logging

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""Performs a low, high or bandpass filter if low & highcut are in range"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	low_in_range = 0 < low < 1
	high_in_range = 0 < high < 1
	# logging.debug(f"low_in_range {low_in_range}, high_in_range {high_in_range}")
	if low_in_range and high_in_range:
		sos = butter(order, [low, high], btype='band', output='sos')
	elif low_in_range and not high_in_range:
		sos = butter(order, low, btype='high', output='sos')
	elif not low_in_range and high_in_range:
		sos = butter(order, high, btype='low', output='sos')
	else:
		return data
	# return filtfilt(b, a, data)
	return sosfiltfilt(sos, data)


def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def make_odd(n):
	if n % 2:
		return n
	else:
		return n+1