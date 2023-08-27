import logging

import numpy as np

from util import io_ops, units, fourier

channel_map = {"L": (0,), "R": (1,), "L+R": (0, 1), "Mean": (0, 1)}


def spectra_from_audio(filename, fft_size=4096, hop=256, channel_mode="L"):
	signal, sr, channels = io_ops.read_file(filename)
	spectra = []
	for channel in channel_map[channel_mode]:
		logging.debug(f"channel {channel}")
		if channel == channels:
			logging.warning("not enough channels for L/R comparison  - fallback to mono")
			break
		# get the magnitude spectrum
		imdata = units.to_dB(fourier.get_mag(signal[:, channel], fft_size, hop, "hann"))
		# fetch from pytorch to take mean
		spec = np.mean(np.array(imdata), axis=1)
		spectra.append(spec)
	return spectra, sr


def spectrum_from_audio(filename, fft_size=4096, hop=256, channel_mode="L"):
	spectra, sr = spectra_from_audio(filename, fft_size, hop, channel_mode)
	if len(spectra) > 1:
		return np.mean(spectra, axis=0), sr
	else:
		return spectra[0], sr


def spectrum_from_audio_stereo(filename, fft_size=4096, hop=256, channel_mode="L"):
	spectra, sr = spectra_from_audio(filename, fft_size, hop, channel_mode)
	if len(spectra) < 2:
		spectra.append(spectra[0])
	return spectra, sr
