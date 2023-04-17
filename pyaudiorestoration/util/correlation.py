import numpy as np
import scipy


def xcorr(a, b, mode='full'):
	"""Normalized cross correlation returning correlation in range [-1.0, 1.0]"""
	norm_a = np.linalg.norm(a)
	a = a / norm_a
	norm_b = np.linalg.norm(b)
	b = b / norm_b
	# return np.correlate(a, b, mode=mode)
	return scipy.signal.correlate(a, b, mode=mode, method='auto')
