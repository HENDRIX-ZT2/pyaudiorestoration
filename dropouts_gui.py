import logging
from statistics import correlation

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal
# import librosa

from PyQt5 import QtWidgets, QtCore
from scipy.signal import savgol_filter

from util import fourier, widgets, config, filters, io_ops, units
from util.correlation import xcorr
from util.units import to_fac


def pairwise(iterable):
	it = iter(iterable)
	a = next(it, None)
	for b in it:
		yield (a, b)
		a = b


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setAcceptDrops(True)

		self.cfg = config.load_config()
		self.file_names = []
		self.names_to_full_paths = {}

		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)

		self.setWindowTitle('Dropouts')

		self.dropout_widget = widgets.DropoutWidget()
		self.display_widget = widgets.SpectrumSettingsWidget(with_canvas=False)
		self.display_widget.fft_c.setCurrentIndex(3)  # 512 smp

		self.b_add = QtWidgets.QPushButton('Load Audio')
		self.b_add.setToolTip("Load audio files you want to process.")
		self.b_add.clicked.connect(self.open_audio)

		self.b_remove = QtWidgets.QPushButton('Remove Audio')
		self.b_remove.setToolTip("Remove audio files you do not want to process.")
		self.b_remove.clicked.connect(self.remove_files)

		self.b_process = QtWidgets.QPushButton('Process')
		self.b_process.setToolTip("Process these files according to the current settings.")
		self.b_process.clicked.connect(self.process)

		self.files_widget = QtWidgets.QListWidget()
		self.files_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.b_add, 0, 0)
		self.qgrid.addWidget(self.b_remove, 0, 1)
		self.qgrid.addWidget(self.b_process, 0, 2)
		self.qgrid.addWidget(self.files_widget, 1, 0, 1, 3)
		self.qgrid.addWidget(self.dropout_widget, 2, 0, 1, 3)
		self.qgrid.addWidget(self.display_widget, 3, 0, 1, 3)

		self.central_widget.setLayout(self.qgrid)

	def dragEnterEvent(self, event):
		if event.mimeData().hasUrls:
			event.accept()
		else:
			event.ignore()

	def dragMoveEvent(self, event):
		if event.mimeData().hasUrls:
			event.setDropAction(QtCore.Qt.CopyAction)
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		if event.mimeData().hasUrls:
			event.setDropAction(QtCore.Qt.CopyAction)
			event.accept()
			src_files = [str(url.toLocalFile()) for url in event.mimeData().urls()]
			for audio_path in sorted(src_files):
				self.load_audio(audio_path)
		else:
			event.ignore()

	def open_audio(self):
		# just a wrapper around load_audio so we can access that via drag & drop and button
		# pyqt5 returns a tuple
		src_files = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Sources', self.cfg.get("dir_in", "C:/"),
														   "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		for audio_path in src_files:
			self.load_audio(audio_path)

	def load_audio(self, audio_path):
		# called whenever a potential audio file is added - via drag& drop or open_audio
		if audio_path:
			self.cfg["dir_in"], file_name = os.path.split(audio_path)
			if file_name not in self.file_names:
				self.file_names.append(file_name)
				self.files_widget.addItem(file_name)
				self.names_to_full_paths[file_name] = audio_path

	def remove_files(self):
		for item in self.files_widget.selectedItems():
			file_name = item.text()
			for i in reversed(range(0, len(self.file_names))):
				if self.file_names[i] == file_name:
					self.file_names.pop(i)
					break
			self.files_widget.takeItem(self.files_widget.row(item))

	def process(self):
		# get params from gui
		fft_size = self.display_widget.fft_size
		fft_overlap = self.display_widget.fft_overlap
		hop = fft_size // fft_overlap
		try:
			if self.dropout_widget.mode == "Heuristic":
				self.process_heuristic(fft_size, hop)
			elif self.dropout_widget.mode == "Heuristic New":
				self.process_heuristic_new(fft_size, hop)
			else:
				self.process_max_mono(fft_size, hop)
		except:
			logging.exception("Failed")
		logging.exception("Done")

	def process_max_mono(self, fft_size, hop):
		for file_name in self.file_names:
			try:
				file_path = self.names_to_full_paths[file_name]
				signal, sr, num_channels = io_ops.read_file(file_path)
				if num_channels != 2:
					print("expects stereo input")
					continue

				n = len(signal)
				# pad input stereo signal
				y_pad = fourier.fix_length(signal, n + fft_size // 2, axis=0)
				# take FFT for each channel
				D_L = np.array(fourier.stft(y_pad[:, 0], n_fft=fft_size, step=hop))
				D_R = np.array(fourier.stft(y_pad[:, 1], n_fft=fft_size, step=hop))

				for op_type, mask in (
						("max", np.abs(D_L) > np.abs(D_R)),
						("min", np.abs(D_L) < np.abs(D_R))
						):
					D_out = np.where(mask, D_L, D_R)
					# take iFFT
					y_out = fourier.istft(D_out, length=n, hop_length=hop)

					io_ops.write_file(file_path, y_out, sr, 1, suffix=op_type)
			except:
				logging.exception(f"Failed for {file_name}")


	def process_heuristic_new(self, fft_size, hop):
		# todo
		# detect dropout events
		# try: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html#scipy.signal.find_peaks_cwt
		# each dropout event candidate samples before and after
		# pre-filter hull with blurred version
		# events can be filtered by certain criteria
		# - correlation before / after
		# - volume slope
		# - width
		# - freq. depth
		# - intensity
		# - relative distance between transients

		width = 0.01  # s
		# get params from gui
		max_width = self.dropout_widget.max_width
		max_slope = self.dropout_widget.max_slope
		num_bands = self.dropout_widget.num_bands
		bottom_freedom = self.dropout_widget.bottom_freedom
		f_upper = self.dropout_widget.f_upper
		f_lower = self.dropout_widget.f_lower
		for file_name in self.file_names:
			file_path = self.names_to_full_paths[file_name]
			signal, sr, num_channels = io_ops.read_file(file_path)

			# distance to look around current fft
			# divide by two because we are looking around the center
			d = int(max_width / 2 * sr / hop)
			print(f"lookaround ffts: {d}")
			for channel in range(num_channels):
				print("Processing channel", channel)
				# which range should dropouts be detected in?
				imdata = fourier.get_mag(signal[:, channel], fft_size, hop, "hann")
				imdata = units.to_dB(imdata)
				# cast to np incase torch was used
				imdata = np.array(imdata)

				onsets = librosa.onset.onset_detect(y=signal[:, channel], sr=sr, units='time', sparse=True)
				plt.vlines(onsets * sr / hop, -100, 1, colors=(1, 0, 0, 0.3), linestyles='--', label='onsets',)
				bin_lower = int(f_lower * fft_size / sr)
				bin_upper = int(f_upper * fft_size / sr)
				# take the mean volume across this band
				vol = np.mean(imdata[bin_lower:bin_upper], axis=0)

				# detect valleys in the volume curve
				peaks, properties = scipy.signal.find_peaks(-vol, height=None, threshold=None, distance=None,
															prominence=5, wlen=None, rel_height=0.5,
															plateau_size=None)
				plt.vlines(peaks, -100, 1, colors=(0, 1.0, 0, 0.2), linestyles='--', label='peaks',)
				plt.plot(vol, label='raw vol',)
				filtered_vol = savgol_filter(vol, d*12, 3)
				plt.plot(savgol_filter(vol, d*2, 3), label='filtered 1',)
				plt.plot(filtered_vol, label='filtered vol',)

				# cwt_peaks = scipy.signal.find_peaks_cwt(-vol, widths=np.arange(max(1, int(1/4*d)), 2*d))
				# plt.vlines(cwt_peaks, -100, 1, colors=(1.0, 0, 0, 0.2), linestyles='-.', label='peaks',)
				correlations = np.zeros(len(peaks))
				for i, peak_i in enumerate(peaks):
					try:
						fft_before = imdata[bin_lower:bin_upper, peak_i-d]
						fft_after = imdata[bin_lower:bin_upper, peak_i+d]
						correlations[i] = xcorr(fft_before, fft_after, mode='valid')[0]

						# plt.plot(fft_before, label=f'fft_before {i}',)
						# plt.plot(fft_after, label=f'fft_after {i}',)
						# if i > 2:
						# 	break
					except:
						logging.exception(f"Failed for {file_name}")
				correlations -= 1.0
				correlations *= 1000
				plt.plot(peaks, correlations, )
		plt.show()

	def process_heuristic(self, fft_size, hop):
		# get params from gui
		max_width = self.dropout_widget.max_width
		max_slope = self.dropout_widget.max_slope
		num_bands = self.dropout_widget.num_bands
		bottom_freedom = self.dropout_widget.bottom_freedom
		f_upper = self.dropout_widget.f_upper
		f_lower = self.dropout_widget.f_lower

		# split the range up into n bands
		bands = np.logspace(np.log2(f_lower), np.log2(f_upper), num=num_bands, endpoint=True, base=2, dtype=np.uint16)

		for file_name in self.file_names:
			file_path = self.names_to_full_paths[file_name]
			signal, sr, num_channels = io_ops.read_file(file_path)

			# distance to look around current fft
			# divide by two because we are looking around the center
			d = int(max_width / 1.5 * sr / hop)

			for channel in range(num_channels):
				print("Processing channel", channel)
				# which range should dropouts be detected in?
				imdata = fourier.get_mag(signal[:, channel], fft_size, hop, "hann")
				imdata = units.to_dB(imdata)
				# cast to np incase torch was used
				imdata = np.array(imdata)
				# now what we generally don't want to do is "fix" dropouts of the lower bands only
				# basically, the gain of a band should be always controlled by that of the band above
				# only the top band acts freely

				# initialize correction
				correction_fac = np.ones(imdata.shape[1]) * 1000

				# go over all bands
				for f_lower_band, f_upper_band in reversed(list(pairwise(bands))):

					# get the bin indices for this band
					bin_lower = int(f_lower_band * fft_size / sr)
					bin_upper = int(f_upper_band * fft_size / sr)

					# take the mean volume across this band
					vol = np.mean(imdata[bin_lower:bin_upper], axis=0)

					# detect valleys in the volume curve
					peaks, properties = scipy.signal.find_peaks(-vol, height=None, threshold=None, distance=None,
																prominence=5, wlen=None, rel_height=0.5,
																plateau_size=None)

					# initialize the gain curve for this band
					gain_curve = np.zeros(imdata.shape[1])

					# go over all peak candidates and use good ones
					for peak_i in peaks:
						# avoid errors at the very ends
						if 2 * d < peak_i < imdata.shape[1] - 2 * d - 1:
							# make sure we are not blurring the left side of a transient
							# sample mean volume around +-d samples on either side of the potential dropout
							# patch_region = np.asarray( (peak_i-d, peak_i+d) )
							# patch_coords = vol[patch_region]
							left = np.mean(vol[peak_i - 2 * d:peak_i - d])
							right = np.mean(vol[peak_i + d:peak_i + 2 * d])
							m = (left - right) / (2 * d)
							# only use it if slant is desirable
							# use abs() to avoid adding reverb
							if abs(m) < max_slope:
								# now interpolate a new patch and get gain from difference to original volume curve
								gain_curve[peak_i - d:peak_i + d + 1] = np.interp(range(2 * d + 1), (0, 2 * d),
																				  (left, right)) - vol[
																								   peak_i - d:peak_i + d + 1]
					# we don't want to make pops more quiet, so clip at 1
					# clip the upper boundary according to band above (was processed before)
					# -> clip the factor to be between 1 and the factor of the band above (with some tolerance)
					correction_fac = np.clip(to_fac(gain_curve), 1, correction_fac * bottom_freedom)
					# resample to match the signal
					vol_corr = signal[:, channel] * np.interp(np.linspace(0, 1, len(signal[:, channel])),
															  np.linspace(0, 1, len(correction_fac)),
															  correction_fac - 1)
					# add the extra bits to the signal
					signal[:, channel] += filters.butter_bandpass_filter(vol_corr, f_lower_band, f_upper_band, sr,
																		 order=3)

			io_ops.write_file(file_path, signal, sr, num_channels)


if __name__ == '__main__':
	widgets.startup(MainWindow)
