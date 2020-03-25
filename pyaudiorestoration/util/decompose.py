# taken from librosa

import numpy as np
from scipy.ndimage import median_filter

def softmask(X, X_ref, power=1, split_zeros=False):
	'''Robustly compute a softmask operation.
		`M = X**power / (X**power + X_ref**power)`
	Parameters
	----------
	X : np.ndarray
		The (non-negative) input array corresponding to the positive mask elements
	X_ref : np.ndarray
		The (non-negative) array of reference or background elements.
		Must have the same shape as `X`.
	power : number > 0 or np.inf
		If finite, returns the soft mask computed in a numerically stable way
		If infinite, returns a hard (binary) mask equivalent to `X > X_ref`.
		Note: for hard masks, ties are always broken in favor of `X_ref` (`mask=0`).
	split_zeros : bool
		If `True`, entries where `X` and X`_ref` are both small (close to 0)
		will receive mask values of 0.5.
		Otherwise, the mask is set to 0 for these entries.
	Returns
	-------
	mask : np.ndarray, shape=`X.shape`
		The output mask array
	Raises
	------
	ParameterError
		If `X` and `X_ref` have different shapes.
		If `X` or `X_ref` are negative anywhere
		If `power <= 0`
	'''
	if X.shape != X_ref.shape:
		print('Shape mismatch: {}!={}'.format(X.shape, X_ref.shape))
		return

	if np.any(X < 0) or np.any(X_ref < 0):
		print('X and X_ref must be non-negative')
		return

	if power <= 0:
		print('power must be strictly positive')
		return

	# We're working with ints, cast to float.
	dtype = X.dtype
	if not np.issubdtype(dtype, np.floating):
		dtype = np.float32

	# Re-scale the input arrays relative to the larger value
	Z = np.maximum(X, X_ref).astype(dtype)
	bad_idx = (Z < np.finfo(dtype).tiny)
	Z[bad_idx] = 1

	# For finite power, compute the softmask
	if np.isfinite(power):
		mask = (X / Z)**power
		ref_mask = (X_ref / Z)**power
		good_idx = ~bad_idx
		mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
		# Wherever energy is below energy in both inputs, split the mask
		if split_zeros:
			mask[bad_idx] = 0.5
		else:
			mask[bad_idx] = 0.0
	else:
		# Otherwise, compute the hard mask
		mask = X > X_ref

	return mask

def harmonic(S, kernel_size=31, power=2.0, mask=False, margin=1.0):
	"""Median-filtering harmonic percussive source separation (HPSS).
	If `margin = 1.0`, decomposes an input spectrogram `S = H + P`
	where `H` contains the harmonic components,
	and `P` contains the percussive components.
	If `margin > 1.0`, decomposes an input spectrogram `S = H + P + R`
	where `R` contains residual components not included in `H` or `P`.

	Parameters
	----------
	S : np.ndarray [shape=(d, n)]
		input spectrogram. May be real (magnitude) or complex.
	kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
		kernel size(s) for the median filters.
		- If scalar, the same size is used for both harmonic and percussive.
		- If tuple, the first value specifies the width of the
		  harmonic filter, and the second value specifies the width
		  of the percussive filter.
	power : float > 0 [scalar]
		Exponent for the Wiener filter when constructing soft mask matrices.
	mask : bool
		Return the masking matrices instead of components.
		Masking matrices contain non-negative real values that
		can be used to measure the assignment of energy from `S`
		into harmonic or percussive components.
		Components can be recovered by multiplying `S * mask_H`
		or `S * mask_P`.
	margin : float or tuple (margin_harmonic, margin_percussive)
		margin size(s) for the masks (as described in [2]_)
		- If scalar, the same size is used for both harmonic and percussive.
		- If tuple, the first value specifies the margin of the
		  harmonic mask, and the second value specifies the margin
		  of the percussive mask.
	Returns
	-------
	harmonic : np.ndarray [shape=(d, n)]
		harmonic component (or mask)
	percussive : np.ndarray [shape=(d, n)]
		percussive component (or mask)

	"""

	if np.isscalar(kernel_size):
		win_harm = kernel_size
		win_perc = kernel_size
	else:
		win_harm = kernel_size[0]
		win_perc = kernel_size[1]

	if np.isscalar(margin):
		margin_harm = margin
		margin_perc = margin
	else:
		margin_harm = margin[0]
		margin_perc = margin[1]

	# margin minimum is 1.0
	if margin_harm < 1 or margin_perc < 1:
		print("Margins must be >= 1.0. A typical range is between 1 and 10.")

	# Compute median filters. Pre-allocation here preserves memory layout.
	harm = np.empty_like(S)
	harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

	perc = np.empty_like(S)
	perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

	split_zeros = (margin_harm == 1 and margin_perc == 1)

	mask_harm = softmask(harm, perc * margin_harm,
							  power=power,
							  split_zeros=split_zeros)

	return S * mask_harm