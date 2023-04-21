# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
from numba import jit
import scipy
try:
	import pyfftw
except:
	print("Warning: pyfftw is not installed. Run 'pip install pyfftw' to speed up spectrogram generation.")

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

def get_mag(*args, **kwargs):
	"""Get the magnitude spectrum from complex input"""
	return abs(stft(*args, **kwargs))+.0000001

def stft(x, n_fft=1024, step=512,  window='hann', num_cores=1, windowlen=None, prog_sig=None):
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
	"""

	def emit_progress():
		if prog_sig:
			prog_sig.emit((i + 1) / n_estimates * 100)

	n_fft = int(n_fft)
	if windowlen:
		# smaller winlen than n_fft, needs padding
		if windowlen > n_fft:
			raise ValueError('if given, windowlen must be smaller than n_fft')
		def segment():
			return np.pad( w * x[i*step : i*step+windowlen], pad_width, mode="edge")
	else:
		# just continue with business as usual, no extra padding
		windowlen = n_fft
		def segment():
			return w * x[i*step : i*step+windowlen]
	pad_width = (n_fft-windowlen)//2
	
	if x.ndim != 1:
		raise ValueError('x must be 1D')
	w = get_window(window, windowlen)
	step = max(n_fft // 2, 1) if step is None else int(step)

	# Pad both sides with half fft size so that frames are centered
	x = np.pad(x, int(n_fft // 2), mode="reflect")
		
	n_freqs = n_fft // 2 + 1
	n_estimates = (len(x) - windowlen) // step + 1
	dtype = dtype_r2c(x.dtype)

	# Pre-allocate the STFT matrix
	result = np.empty((n_freqs, n_estimates), dtype=dtype, order='F')
	
	# don't force fftw, fallback to numpy fft if pyFFTW import fails
	try:
		# this is the input for the FFT object
		fft_in = pyfftw.empty_aligned(n_fft, dtype='float32')
		# the fft object itself, which must be called for each FFT
		fft_ob = pyfftw.builders.rfft(fft_in, threads=num_cores, planner_effort="FFTW_ESTIMATE", overwrite_input=True)
		for i in range(n_estimates):
			# set the data on the FFT input
			fft_ob.input_array[:] = segment()
			result[:, i] = fft_ob()
			emit_progress()
	except:
		print("Fallback to numpy fftpack!")
		for i in range(n_estimates):
			result[:, i] = np.fft.rfft( segment() )
			emit_progress()
	# normalize
	result /= n_fft
	return result

def dtype_r2c(d, default=np.complex64):
	'''Find the complex numpy dtype corresponding to a real dtype.
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
	'''
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
	'''Find the real numpy dtype corresponding to a complex dtype.
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
	'''
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

def get_window(window, Nx, fftbins=True):
	'''Compute a window function.
	This is a wrapper for `scipy.signal.get_window` that additionally
	supports callable or pre-computed windows.
	Parameters
	----------
	window : string, tuple, number, callable, or list-like
		The window specification:
		- If string, it's the name of the window function (e.g., `'hann'`)
		- If tuple, it's the name of the window function and any parameters
		  (e.g., `('kaiser', 4.0)`)
		- If numeric, it is treated as the beta parameter of the `'kaiser'`
		  window, as in `scipy.signal.get_window`.
		- If callable, it's a function that accepts one integer argument
		  (the window length)
		- If list-like, it's a pre-computed window of the correct length `Nx`
	Nx : int > 0
		The length of the window
	fftbins : bool, optional
		If True (default), create a periodic window for use with FFT
		If False, create a symmetric window for filter design applications.
	Returns
	-------
	get_window : np.ndarray
		A window of length `Nx` and type `window`
	See Also
	--------
	scipy.signal.get_window
	Notes
	-----
	This function caches at level 10.
	Raises
	------
	ParameterError
		If `window` is supplied as a vector of length != `n_fft`,
		or is otherwise mis-specified.
	'''
	if callable(window):
		return window(Nx)

	elif (isinstance(window, (str, tuple)) or np.isscalar(window)):
		# TODO: if we add custom window functions in librosa, call them here

		return scipy.signal.get_window(window, Nx, fftbins=fftbins)

	elif isinstance(window, (np.ndarray, list)):
		if len(window) == Nx:
			return np.asarray(window)

		raise ParameterError('Window size mismatch: '
							 '{:d} != {:d}'.format(len(window), Nx))
	else:
		raise ParameterError('Invalid window specification: {}'.format(window))

def librosa_stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=None, pad_mode='reflect'):
	"""Short-time Fourier transform (STFT).
	The STFT represents a signal in the time-frequency domain by
	computing discrete Fourier transforms (DFT) over short overlapping
	windows.
	This function returns a complex-valued matrix D such that
	- ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
	  at frame ``t``, and
	- ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
	  at frame ``t``.
	The integers ``t`` and ``f`` can be converted to physical units by means
	of the utility functions `frames_to_sample` and `fft_frequencies`.
	Parameters
	----------
	y : np.ndarray [shape=(n,)], real-valued
		input signal
	n_fft : int > 0 [scalar]
		length of the windowed signal after padding with zeros.
		The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
		The default value, ``n_fft=2048`` samples, corresponds to a physical
		duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
		default sample rate in librosa. This value is well adapted for music
		signals. However, in speech processing, the recommended value is 512,
		corresponding to 23 milliseconds at a sample rate of 22050 Hz.
		In any case, we recommend setting ``n_fft`` to a power of two for
		optimizing the speed of the fast Fourier transform (FFT) algorithm.
	hop_length : int > 0 [scalar]
		number of audio samples between adjacent STFT columns.
		Smaller values increase the number of columns in ``D`` without
		affecting the frequency resolution of the STFT.
		If unspecified, defaults to ``win_length // 4`` (see below).
	win_length : int <= n_fft [scalar]
		Each frame of audio is windowed by ``window`` of length ``win_length``
		and then padded with zeros to match ``n_fft``.
		Smaller values improve the temporal resolution of the STFT (i.e. the
		ability to discriminate impulses that are closely spaced in time)
		at the expense of frequency resolution (i.e. the ability to discriminate
		pure tones that are closely spaced in frequency). This effect is known
		as the time-frequency localization trade-off and needs to be adjusted
		according to the properties of the input signal ``y``.
		If unspecified, defaults to ``win_length = n_fft``.
	window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
		Either:
		- a window specification (string, tuple, or number);
		  see `scipy.signal.get_window`
		- a window function, such as `scipy.signal.windows.hann`
		- a vector or array of length ``n_fft``
		Defaults to a raised cosine window (`'hann'`), which is adequate for
		most applications in audio signal processing.
		.. see also:: `filters.get_window`
	center : boolean
		If ``True``, the signal ``y`` is padded so that frame
		``D[:, t]`` is centered at ``y[t * hop_length]``.
		If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
		Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
		time grid by means of `librosa.frames_to_samples`.
		Note, however, that ``center`` must be set to `False` when analyzing
		signals with `librosa.stream`.
		.. see also:: `librosa.stream`
	dtype : np.dtype, optional
		Complex numeric type for ``D``.  Default is inferred to match the
		precision of the input signal.
	pad_mode : string or function
		If ``center=True``, this argument is passed to `np.pad` for padding
		the edges of the signal ``y``. By default (``pad_mode="reflect"``),
		``y`` is padded on both sides with its own reflection, mirrored around
		its first and last sample respectively.
		If ``center=False``,  this argument is ignored.
		.. see also:: `numpy.pad`
	Returns
	-------
	D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
		Complex-valued matrix of short-term Fourier transform
		coefficients.
	"""

	# By default, use the entire frame
	if win_length is None:
		win_length = n_fft

	# Set the default hop, if it's not already specified
	if hop_length is None:
		hop_length = int(win_length // 4)

	fft_window = get_window(window, win_length, fftbins=True)

	# Pad the window out to n_fft size
	fft_window = pad_center(fft_window, n_fft)

	# Reshape so that the window can be broadcast
	fft_window = fft_window.reshape((-1, 1))

	# Check audio is valid
	valid_audio(y)

	# Pad the time series so that frames are centered
	if center:
		if n_fft > y.shape[-1]:
			warnings.warn('n_fft={} is too small for input signal of length={}'.format(n_fft, y.shape[-1]))

		y = np.pad(y, int(n_fft // 2), mode=pad_mode)

	elif n_fft > y.shape[-1]:
		raise ParameterError('n_fft={} is too small for input signal of length={}'.format(n_fft, y.shape[-1]))

	# Window the time series.
	y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

	if dtype is None:
		dtype = dtype_r2c(y.dtype)

	# Pre-allocate the STFT matrix
	stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
						   dtype=dtype,
						   order='F')

	fft = get_fftlib()

	# how many columns can we fit within MAX_MEM_BLOCK?
	n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] *
									   stft_matrix.itemsize)
	n_columns = max(n_columns, 1)

	for bl_s in range(0, stft_matrix.shape[1], n_columns):
		bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

		stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
											 y_frames[:, bl_s:bl_t],
											 axis=0)
	return stft_matrix

def pad_center(data, size, axis=-1, **kwargs):
    '''Pad an array to a target length along a target axis.
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
    '''

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
    '''Compute the tiny-value corresponding to an input's data type.
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
    '''

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny

def istft(stft_matrix, hop_length=None, win_length=None, window='hann', center=True, dtype=None, length=None):
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
	window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
		- a window specification (string, tuple, or number);
		  see `scipy.signal.get_window`
		- a window function, such as `scipy.signal.windows.hann`
		- a user-specified window vector of length ``n_fft``
		.. see also:: `filters.get_window`
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
	stft_matrix *= n_fft
	# By default, use the entire frame
	if win_length is None:
		win_length = n_fft

	# Set the default hop, if it's not already specified
	if hop_length is None:
		hop_length = int(win_length // 4)

	ifft_window = get_window(window, win_length, fftbins=True)

	# Pad out to match n_fft, and add a broadcasting axis
	ifft_window = pad_center(ifft_window, n_fft)[:, np.newaxis]

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
		ytmp = ifft_window * np.fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

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
    '''Fix the length an array ``data`` to exactly ``size`` along a target axis.
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
    '''

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
    '''Helper function for window sum-square calculation.'''

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]


def window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
                     dtype=np.float32, norm=None):
    '''Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing observations
    in short-time Fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
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
    '''
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    '''Normalize an array along a chosen axis.
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
    '''

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
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

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
