# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
try:
	import pyfftw
except:
	print("Warning: pyfftw is not installed. Run 'pip install pyfftw' to speed up spectrogram generation.")


def stft(x, n_fft=1024, step=512,  window='hann', num_cores=1, windowlen=None,):
	"""Compute the STFT

	Parameters
	----------
	x : array-like
		1D signal to operate on. ``If len(x) < n_fft``, x will be zero-padded
		to length ``n_fft``.
	n_fft : int
		Number of FFT points. Much faster for powers of two.
	step : int | None
		Step size between calculations. If None, ``n_fft // 2``
		will be used.
	window : str | None
		Window function to use. Can be ``'hann'`` for Hann window, or None
		for no windowing.

	Returns
	-------
	stft : ndarray
		Spectrogram of the data, shape (n_freqs, n_steps).

	See also
	--------
	fft_freqs
	"""
	
	n_fft = int(n_fft)
	if windowlen:
		#smaller winlen than n_fft, needs padding
		if windowlen > n_fft:
			raise ValueError('if given, windowlen must be smaller than n_fft')
		def segment():
			return np.pad( w * x[i*step : i*step+windowlen], pad_width, mode="edge")
	else:
		#just continue with business as usual, no extra padding
		windowlen = n_fft
		def segment():
			return w * x[i*step : i*step+windowlen]
	pad_width = (n_fft-windowlen)//2
	
	if x.ndim != 1:
		raise ValueError('x must be 1D')
	if window is not None:
		if window not in ('hann',):
			raise ValueError('window must be "hann" or None')
		w = np.hanning(windowlen)
	else:
		w = np.ones(windowlen)
	step = max(n_fft // 2, 1) if step is None else int(step)

	# Pad both sides with half fft size so that frames are centered
	x = np.pad(x, int(n_fft // 2), mode="reflect")
		
	n_freqs = n_fft // 2 + 1
	n_estimates = (len(x) - windowlen) // step + 1
	result = np.empty((n_freqs, n_estimates), "float32")
	
	#don't force fftw, fallback to numpy fft if pyFFTW import fails
	try:
		#this is the input for the FFT object
		fft_in = pyfftw.empty_aligned(n_fft, dtype='float32')
		#the fft object itself, which must be called for each FFT
		fft_ob = pyfftw.builders.rfft(fft_in, threads=num_cores, planner_effort="FFTW_ESTIMATE", overwrite_input=True)
		for i in range(n_estimates):
			#set the data on the FFT input
			fft_ob.input_array[:] = segment()
			result[:, i] = abs(fft_ob() / n_fft)+.0000001
		# pyfftw.interfaces.cache.enable()
		# for i in range(n_estimates):
			# result[:, i] = abs(pyfftw.interfaces.numpy_fft.rfft(w * x[i * step:i * step + n_fft], threads=num_cores) / n_fft)+.0000001
	except:
		print("Fallback to numpy fftpack!")
		for i in range(n_estimates):
			result[:, i] = abs(np.fft.rfft( segment() ) / n_fft)+.0000001
	return result


def fft_freqs(n_fft, fs):
	"""Return frequencies for DFT

	Parameters
	----------
	n_fft : int
		Number of points in the FFT.
	fs : float
		The sampling rate.
	"""
	return np.arange(0, (n_fft // 2 + 1)) / float(n_fft) * float(fs)
