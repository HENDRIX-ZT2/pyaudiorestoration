import contextlib
import logging
import time
import os

import numpy as np
from numba import jit
import scipy

try:
	import torch
except:
	logging.warning("torch is not installed. If you have a NVIDIA GPU, install pytorch for cuda.")

try:
	import pyfftw
except:
	logging.warning("pyfftw is not installed. Run 'pip install pyfftw' to speed up spectrogram generation.")

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def get_mag(*args, **kwargs):
	"""Get the magnitude spectrum from complex input"""
	return abs(stft(*args, **kwargs)) + .0000001


def stft(x, n_fft=1024, step=512, window_name='blackmanharris'):
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
	window_name : str | None
		Window function to use. Can be ``'hann'`` for Hann window, or None
		for no windowing.

	Returns
	-------
	stft : ndarray
		Spectrogram of the data, shape (n_freqs, n_steps).
	"""

	n_fft = int(n_fft)
	step = max(n_fft // 2, 1) if step is None else int(step)
	if x.ndim != 1:
		raise ValueError('x must be 1D')
	window = scipy.signal.get_window(window_name, n_fft).astype(np.float32)
	for fft_function in (
			torch_rfft2,
			pyfftw_rfft2,
			np_rfft_pick,):
		try:
			return fft_function(n_fft, step, window, x)
		except:
			logging.exception(f"FFT method {fft_function} failed")
			continue


def estimate_and_center(n_fft, step, x):
	# Pad both sides with half fft size so that frames are centered
	x = np.pad(x, int(n_fft // 2), mode="reflect")
	n_estimates = (len(x) - n_fft) // step + 1
	return n_estimates, x


@contextlib.contextmanager
def timed_log(method_name):
	start = time.time()
	yield
	logging.info(f"{method_name} {time.time() - start:0.2f}s")


def torch_rfft2(n_fft, step, window, x):
	with timed_log("pytorch"):
		device = "cuda"
		# device = "cpu"
		if device == "cuda":
			assert torch.cuda.is_available()
			# ensuring that these are 32bit (and thus the complex result 64bit) makes fetching faster
			x = torch.as_tensor(x, dtype=torch.float32, device=device)
			window = torch.as_tensor(window, dtype=torch.float32, device=device)
			result = torch.stft(
				x, n_fft, hop_length=step, window=window, center=True, pad_mode='reflect',
				normalized=True, onesided=True, return_complex=True)
			# fetch from gpu, which is somewhat costly, but still a bit faster than cpu
			result = result.cpu()
		else:
			# manual implementation
			n_estimates, x = estimate_and_center(n_fft, step, x)
			x = torch.as_tensor(x, dtype=None, device=device)
			# create overlapping slices from input
			fft_in = x.unfold(0, n_fft, step)
			# apply the window to all slices (doesn't work like this on when window is a tensor!)
			fft_in *= window
			# flip the dimensions to bring shape to (n_fft, n_estimates)
			fft_in = fft_in.transpose(1, 0)
			# print(x.shape, fft_in.shape)
			# take the fft and normalize it
			result = torch.fft.rfft2(fft_in, dim=0, norm="ortho")
	return result


def pyfftw_rfft2(n_fft, step, window, x):
	# this is the input for the FFT object
	with timed_log("pyfftw"):
		n_estimates, x = estimate_and_center(n_fft, step, x)
		# raise AttributeError
		fft_in = pyfftw.empty_aligned((n_fft, n_estimates), dtype='float32')
		segment_array(fft_in, n_estimates, n_fft, step, window, x)
		fft_ob = pyfftw.builders.rfft(fft_in, axis=0, threads=os.cpu_count(), planner_effort="FFTW_ESTIMATE")
		result = fft_ob(ortho=True, normalise_idft=False)
	return result


def np_rfft_pick(n_fft, step, window, x):
	logging.warning("Fallback to numpy fftpack (slower)!")
	n_estimates, x = estimate_and_center(n_fft, step, x)
	# non-vectorized appears to be faster for small sizes
	if n_fft > 512:
		def segment():
			return window * x[i * step: i * step + n_fft]
		with timed_log("fftpack (non-vectorized)"):
			complex_dtype = dtype_r2c(x.dtype)
			# Pre-allocate the STFT matrix
			n_freqs = n_fft // 2 + 1
			result = np.empty((n_freqs, n_estimates), dtype=complex_dtype, order='F')
			for i in range(n_estimates):
				result[:, i] = np.fft.rfft(segment())
	else:
		with timed_log("fftpack (vectorized)"):
			# Pre-allocate the STFT matrix
			fft_in = np.empty((n_fft, n_estimates), dtype='float32')
			segment_array(fft_in, n_estimates, n_fft, step, window, x)
			result = np.fft.rfft(fft_in, axis=0)
	# normalize for constant volume across different FFT sizes
	return result / np.sqrt(n_fft)


@jit(nopython=True, cache=True, nogil=True)
def segment_array(fft_in, n_estimates, n_fft, step, window, x):
	for i in range(n_estimates):
		# set the data on the FFT input
		fft_in[:, i] = window * x[i * step: i * step + n_fft]


def dtype_r2c(d, default=np.complex64):
	"""Find the complex numpy dtype corresponding to a real dtype.
	This is used to maintain numerical precision and memory footprint
	when constructing complex arrays from real-valued data
	(e.g. in a Fourier transform).
	A `float32` (single-precision) type maps to `complex64`,
	while a `float64` (double-precision) maps to `complex128`.
	Parameters
	----------
	d : np.dtype
		The real-valued dtype to convert to complex.
		If ``d`` is a complex type already, it will be returned.
	default : np.dtype, optional
		The default complex target type, if ``d`` does not match a
		known dtype
	Returns
	-------
	d_c : np.dtype
		The complex dtype
	"""
	mapping = {np.dtype(np.float32): np.complex64,
			   np.dtype(np.float64): np.complex128,
			   np.dtype(np.float): np.complex}

	# If we're given a complex type already, return it
	dt = np.dtype(d)
	if dt.kind == 'c':
		return dt

	# Otherwise, try to map the dtype.
	# If no match is found, return the default.
	return np.dtype(mapping.get(dt, default))


def dtype_c2r(d, default=np.float32):
	"""Find the real numpy dtype corresponding to a complex dtype.
	This is used to maintain numerical precision and memory footprint
	when constructing real arrays from complex-valued data
	(e.g. in an inverse Fourier transform).
	A `complex64` (single-precision) type maps to `float32`,
	while a `complex128` (double-precision) maps to `float64`.
	Parameters
	----------
	d : np.dtype
		The complex-valued dtype to convert to real.
		If ``d`` is a real (float) type already, it will be returned.
	default : np.dtype, optional
		The default real target type, if ``d`` does not match a
		known dtype
	Returns
	-------
	d_r : np.dtype
		The real dtype
	"""
	mapping = {np.dtype(np.complex64): np.float32,
			   np.dtype(np.complex128): np.float64,
			   np.dtype(np.complex): np.float}

	# If we're given a real type already, return it
	dt = np.dtype(d)
	if dt.kind == 'f':
		return dt

	# Otherwise, try to map the dtype.
	# If no match is found, return the default.
	return np.dtype(mapping.get(np.dtype(d), default))


def pad_center(data, size, axis=-1, **kwargs):
	"""Pad an array to a target length along a target axis.
	This differs from `np.pad` by centering the data prior to padding,
	analogous to `str.center`
	Parameters
	----------
	data : np.ndarray
		Vector to be padded and centered
	size : int >= len(data) [scalar]
		Length to pad ``data``
	axis : int
		Axis along which to pad and center the data
	kwargs : additional keyword arguments
	  arguments passed to `np.pad`
	Returns
	-------
	data_padded : np.ndarray
		``data`` centered and padded to length ``size`` along the
		specified axis
	Raises
	------
	ParameterError
		If ``size < data.shape[axis]``
	See Also
	--------
	numpy.pad
	"""

	kwargs.setdefault('mode', 'constant')

	n = data.shape[axis]

	lpad = int((size - n) // 2)

	lengths = [(0, 0)] * data.ndim
	lengths[axis] = (lpad, int(size - n - lpad))

	if lpad < 0:
		raise ParameterError(('Target size ({:d}) must be '
							  'at least input size ({:d})').format(size, n))

	return np.pad(data, lengths, **kwargs)


def tiny(x):
	"""Compute the tiny-value corresponding to an input's data type.
	This is the smallest "usable" number representable in ``x.dtype``
	(e.g., float32).
	This is primarily useful for determining a threshold for
	numerical underflow in division or multiplication operations.
	Parameters
	----------
	x : number or np.ndarray
		The array to compute the tiny-value for.
		All that matters here is ``x.dtype``
	Returns
	-------
	tiny_value : float
		The smallest positive usable number for the type of ``x``.
		If ``x`` is integer-typed, then the tiny value for ``np.float32``
		is returned instead.
	See Also
	--------
	numpy.finfo
	"""

	# Make sure we have an array view
	x = np.asarray(x)

	# Only floating types generate a tiny
	if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
		dtype = x.dtype
	else:
		dtype = np.float32

	return np.finfo(dtype).tiny


def istft(stft_matrix, hop_length=None, win_length=None, window='blackmanharris', center=True, dtype=None, length=None):
	"""
	Inverse short-time Fourier transform (ISTFT).
	Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``
	by minimizing the mean squared error between ``stft_matrix`` and STFT of
	``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).
	In general, window function, hop length and other parameters should be same
	as in stft, which mostly leads to perfect reconstruction of a signal from
	unmodified ``stft_matrix``.
	.. [#] D. W. Griffin and J. S. Lim,
		"Signal estimation from modified short-time Fourier transform,"
		IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.
	Parameters
	----------
	stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
		STFT matrix from ``stft``
	hop_length : int > 0 [scalar]
		Number of frames between STFT columns.
		If unspecified, defaults to ``win_length // 4``.
	win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
		When reconstructing the time series, each frame is windowed
		and each sample is normalized by the sum of squared window
		according to the ``window`` function (see below).
		If unspecified, defaults to ``n_fft``.
	window : string for a window name
	center : boolean
		- If ``True``, ``D`` is assumed to have centered frames.
		- If ``False``, ``D`` is assumed to have left-aligned frames.
	dtype : numeric type
		Real numeric type for ``y``.  Default is to match the numerical
		precision of the input spectrogram.
	length : int > 0, optional
		If provided, the output ``y`` is zero-padded or clipped to exactly
		``length`` samples.
	Returns
	-------
	y : np.ndarray [shape=(n,)]
		time domain signal reconstructed from ``stft_matrix``
	"""

	n_fft = 2 * (stft_matrix.shape[0] - 1)

	# librosa doesn't do this - should we? or do it in get_mag() only?
	stft_matrix *= np.sqrt(n_fft)
	# By default, use the entire frame
	if win_length is None:
		win_length = n_fft

	# Set the default hop, if it's not already specified
	if hop_length is None:
		hop_length = int(win_length // 4)

	window = scipy.signal.get_window(window, win_length, fftbins=True)

	# Pad out to match n_fft, and add a broadcasting axis
	window = pad_center(window, n_fft)[:, np.newaxis]

	# For efficiency, trim STFT frames according to signal length if available
	if length:
		if center:
			padded_length = length + int(n_fft)
		else:
			padded_length = length
		n_frames = min(
			stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
	else:
		n_frames = stft_matrix.shape[1]

	expected_signal_len = n_fft + hop_length * (n_frames - 1)

	if dtype is None:
		dtype = dtype_c2r(stft_matrix.dtype)

	y = np.zeros(expected_signal_len, dtype=dtype)

	n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] *
								  stft_matrix.itemsize)
	n_columns = max(n_columns, 1)

	# fft = get_fftlib()

	frame = 0
	for bl_s in range(0, n_frames, n_columns):
		bl_t = min(bl_s + n_columns, n_frames)

		# invert the block and apply the window function
		ytmp = window * np.fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

		# Overlap-add the istft block starting at the i'th frame
		__overlap_add(y[frame * hop_length:], ytmp, hop_length)

		frame += (bl_t - bl_s)

	# Normalize by sum of squared window
	ifft_window_sum = window_sumsquare(window,
									   n_frames,
									   win_length=win_length,
									   n_fft=n_fft,
									   hop_length=hop_length,
									   dtype=dtype)

	approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
	y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

	if length is None:
		# If we don't need to control length, just do the usual center trimming
		# to eliminate padded data
		if center:
			y = y[int(n_fft // 2):-int(n_fft // 2)]
	else:
		if center:
			# If we're centering, crop off the first n_fft//2 samples
			# and then trim/pad to the target length.
			# We don't trim the end here, so that if the signal is zero-padded
			# to a longer duration, the decay is smooth by windowing
			start = int(n_fft // 2)
		else:
			# If we're not centering, start at 0 and trim/pad as necessary
			start = 0

		y = fix_length(y[start:], length)

	return y


def fix_length(data, size, axis=-1, **kwargs):
	"""Fix the length an array ``data`` to exactly ``size`` along a target axis.
	If ``data.shape[axis] < n``, pad according to the provided kwargs.
	By default, ``data`` is padded with trailing zeros.
	Parameters
	----------
	data : np.ndarray
	  array to be length-adjusted
	size : int >= 0 [scalar]
	  desired length of the array
	axis : int, <= data.ndim
	  axis along which to fix length
	kwargs : additional keyword arguments
		Parameters to ``np.pad``
	Returns
	-------
	data_fixed : np.ndarray [shape=data.shape]
		``data`` either trimmed or padded to length ``size``
		along the specified axis.
	See Also
	--------
	numpy.pad
	"""

	kwargs.setdefault('mode', 'constant')

	n = data.shape[axis]

	if n > size:
		slices = [slice(None)] * data.ndim
		slices[axis] = slice(0, size)
		return data[tuple(slices)]

	elif n < size:
		lengths = [(0, 0)] * data.ndim
		lengths[axis] = (0, size - n)
		return np.pad(data, lengths, **kwargs)

	return data


@jit(nopython=True, cache=True)
def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
	"""Helper function for window sum-square calculation."""

	n = len(x)
	n_fft = len(win_sq)
	for i in range(n_frames):
		sample = i * hop_length
		x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]


def window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
					 dtype=np.float32, norm=None):
	"""Compute the sum-square envelope of a window function at a given hop length.
	This is used to estimate modulation effects induced by windowing observations
	in short-time Fourier transforms.
	Parameters
	----------
	window : string
	n_frames : int > 0
		The number of analysis frames
	hop_length : int > 0
		The number of samples to advance between frames
	win_length : [optional]
		The length of the window function.  By default, this matches ``n_fft``.
	n_fft : int > 0
		The length of each analysis frame.
	dtype : np.dtype
		The data type of the output
	Returns
	-------
	wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
		The sum-squared envelope of the window function
	Examples
	--------
	For a fixed frame length (2048), compare modulation effects for a Hann window
	at different hop lengths:
	>>> n_frames = 50
	>>> wss_256 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=256)
	>>> wss_512 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=512)
	>>> wss_1024 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=1024)
	>>> import matplotlib.pyplot as plt
	>>> fig, ax = plt.subplots(nrows=3, sharey=True)
	>>> ax[0].plot(wss_256)
	>>> ax[0].set(title='hop_length=256')
	>>> ax[1].plot(wss_512)
	>>> ax[1].set(title='hop_length=512')
	>>> ax[2].plot(wss_1024)
	>>> ax[2].set(title='hop_length=1024')
	"""

	if win_length is None:
		win_length = n_fft

	n = n_fft + hop_length * (n_frames - 1)
	x = np.zeros(n, dtype=dtype)

	# Compute the squared window at the desired length
	win_sq = scipy.signal.get_window(window, win_length)
	win_sq = normalize(win_sq, norm=norm) ** 2
	win_sq = pad_center(win_sq, n_fft)

	# Fill the envelope
	__window_ss_fill(x, win_sq, n_frames, hop_length)

	return x


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
	"""Normalize an array along a chosen axis.
	Given a norm (described below) and a target axis, the input
	array is scaled so that::
		norm(S, axis=axis) == 1
	For example, ``axis=0`` normalizes each column of a 2-d array
	by aggregating over the rows (0-axis).
	Similarly, ``axis=1`` normalizes each row of a 2-d array.
	This function also supports thresholding small-norm slices:
	any slice (i.e., row or column) with norm below a specified
	``threshold`` can be left un-normalized, set to all-zeros, or
	filled with uniform non-zero values that normalize to 1.
	Note: the semantics of this function differ from
	`scipy.linalg.norm` in two ways: multi-dimensional arrays
	are supported, but matrix-norms are not.
	Parameters
	----------
	S : np.ndarray
		The matrix to normalize
	norm : {np.inf, -np.inf, 0, float > 0, None}
		- `np.inf`  : maximum absolute value
		- `-np.inf` : mininum absolute value
		- `0`    : number of non-zeros (the support)
		- float  : corresponding l_p norm
			See `scipy.linalg.norm` for details.
		- None : no normalization is performed
	axis : int [scalar]
		Axis along which to compute the norm.
	threshold : number > 0 [optional]
		Only the columns (or rows) with norm at least ``threshold`` are
		normalized.
		By default, the threshold is determined from
		the numerical precision of ``S.dtype``.
	fill : None or bool
		If None, then columns (or rows) with norm below ``threshold``
		are left as is.
		If False, then columns (rows) with norm below ``threshold``
		are set to 0.
		If True, then columns (rows) with norm below ``threshold``
		are filled uniformly such that the corresponding norm is 1.
		.. note:: ``fill=True`` is incompatible with ``norm=0`` because
			no uniform vector exists with l0 "norm" equal to 1.
	Returns
	-------
	S_norm : np.ndarray [shape=S.shape]
		Normalized array
	Raises
	------
	ParameterError
		If ``norm`` is not among the valid types defined above
		If ``S`` is not finite
		If ``fill=True`` and ``norm=0``
	See Also
	--------
	scipy.linalg.norm
	"""

	# Avoid div-by-zero
	if threshold is None:
		threshold = tiny(S)

	elif threshold <= 0:
		raise ParameterError('threshold={} must be strictly '
							 'positive'.format(threshold))

	if fill not in [None, False, True]:
		raise ParameterError('fill={} must be None or boolean'.format(fill))

	if not np.all(np.isfinite(S)):
		raise ParameterError('Input must be finite')

	# All norms only depend on magnitude, let's do that first
	mag = np.abs(S).astype(np.float)

	# For max/min norms, filling with 1 works
	fill_norm = 1

	if norm == np.inf:
		length = np.max(mag, axis=axis, keepdims=True)

	elif norm == -np.inf:
		length = np.min(mag, axis=axis, keepdims=True)

	elif norm == 0:
		if fill is True:
			raise ParameterError('Cannot normalize with norm=0 and fill=True')

		length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

	elif np.issubdtype(type(norm), np.number) and norm > 0:
		length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1. / norm)

		if axis is None:
			fill_norm = mag.size ** (-1. / norm)
		else:
			fill_norm = mag.shape[axis] ** (-1. / norm)

	elif norm is None:
		return S

	else:
		raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

	# indices where norm is below the threshold
	small_idx = length < threshold

	Snorm = np.empty_like(S)
	if fill is None:
		# Leave small indices un-normalized
		length[small_idx] = 1.0
		Snorm[:] = S / length

	elif fill:
		# If we have a non-zero fill value, we locate those entries by
		# doing a nan-divide.
		# If S was finite, then length is finite (except for small positions)
		length[small_idx] = np.nan
		Snorm[:] = S / length
		Snorm[np.isnan(Snorm)] = fill_norm
	else:
		# Set small values to zero by doing an inf-divide.
		# This is safe (by IEEE-754) as long as S is finite.
		length[small_idx] = np.inf
		Snorm[:] = S / length

	return Snorm


@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
	# numba-accelerated overlap add for inverse stft
	# y is the pre-allocated output buffer
	# ytmp is the windowed inverse-stft frames
	# hop_length is the hop-length of the STFT analysis

	n_fft = ytmp.shape[0]
	for frame in range(ytmp.shape[1]):
		sample = frame * hop_length
		y[sample:(sample + n_fft)] += ytmp[:, frame]


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
