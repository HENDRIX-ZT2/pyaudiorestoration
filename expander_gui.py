import logging

import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from PyQt5 import QtWidgets
from util import io_ops, units, widgets, config
from util.filters import make_odd
from util.spectrum_flat import spectrum_from_audio_stereo
from util.widgets import PlotMainWindow


class MainWindow(PlotMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)

		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)

		self.setWindowTitle('Spectral Expander')
		self.file_src = ""
		self.spectra = []
		self.fft_size = 512
		self.sr = 44100
		self.fft_hop = self.fft_size // 8
		self.vol_curves = []
		self.cfg = config.load_config()

		self.files_widget = widgets.FilesWidget(self, 1, self.cfg)
		self.files_widget.on_load_file = self.open_file

		self.b_expand = QtWidgets.QPushButton('Expand')
		self.b_expand.setToolTip("Write expanded audio to a new file.")
		self.b_expand.clicked.connect(self.expand)

		self.s_band_lower = QtWidgets.QSpinBox()
		self.s_band_lower.valueChanged.connect(self.plot)
		self.s_band_lower.setRange(0, 22000)
		self.s_band_lower.setSingleStep(1000)
		self.s_band_lower.setValue(13000)
		self.s_band_lower.setToolTip("Lower frequency boundary of noise floor")

		self.s_band_upper = QtWidgets.QSpinBox()
		self.s_band_upper.valueChanged.connect(self.plot)
		self.s_band_upper.setRange(1000, 22000)
		self.s_band_upper.setSingleStep(1000)
		self.s_band_upper.setValue(17000)
		self.s_band_upper.setToolTip("Upper frequency boundary of noise floor")

		self.s_clip_lower = QtWidgets.QSpinBox()
		self.s_clip_lower.valueChanged.connect(self.plot)
		self.s_clip_lower.setRange(-200, 0)
		self.s_clip_lower.setSingleStep(1)
		self.s_clip_lower.setValue(-120)
		self.s_clip_lower.setToolTip("Lower gain boundary of noise floor")

		self.s_clip_upper = QtWidgets.QSpinBox()
		self.s_clip_upper.valueChanged.connect(self.plot)
		self.s_clip_upper.setRange(-200, 0)
		self.s_clip_upper.setSingleStep(1)
		self.s_clip_upper.setValue(-85)
		self.s_clip_upper.setToolTip("Upper gain boundary of noise floor")

		self.c_channels = QtWidgets.QComboBox(self)
		self.c_channels.addItems(list(("L+R", "L", "R", "Mean")))
		self.c_channels.setToolTip("Which channels should be analyzed?")

		self.s_smoothing = QtWidgets.QDoubleSpinBox()
		self.s_smoothing.setRange(.001, 5)
		self.s_smoothing.setSingleStep(.01)
		self.s_smoothing.setValue(.11)
		self.s_smoothing.setToolTip("Smoothing in s.")

		self.l_result = QtWidgets.QLabel("Result: ")

		self.qgrid = QtWidgets.QGridLayout()
		# self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.toolbar, 0, 0, 1, 8)
		self.qgrid.addWidget(self.canvas, 1, 0, 1, 8)
		self.qgrid.addWidget(self.files_widget, 2, 0)
		self.qgrid.addWidget(self.b_expand, 2, 1)
		self.qgrid.addWidget(self.c_channels, 2, 2)
		self.qgrid.addWidget(self.s_band_lower, 2, 3)
		self.qgrid.addWidget(self.s_band_upper, 2, 4)
		self.qgrid.addWidget(self.s_clip_lower, 2, 5)
		self.qgrid.addWidget(self.s_clip_upper, 2, 6)
		self.qgrid.addWidget(self.s_smoothing, 2, 7)

		self.central_widget.setLayout(self.qgrid)

		for btn in (self.s_band_lower, self.s_band_upper, self.s_clip_lower, self.s_clip_upper, self.s_smoothing):
			btn.valueChanged.connect(self.on_param_changed)
		self.c_channels.currentIndexChanged.connect(self.update_spectrum)

	def on_param_changed(self, ):

		self.vol_curves = []
		band_lower = self.s_band_lower.value()
		band_upper = self.s_band_upper.value()
		# clip_lower = self.s_clip_lower.value()
		# clip_upper = self.s_clip_upper.value()

		# sample over an uneven number of points in volume curve
		smoothing = make_odd(int(self.s_smoothing.value() * self.sr / self.fft_hop))

		# update volume curve
		if self.spectra:
			num_bins, last_fft_i = self.spectra[0].shape

			def freq2bin(f):
				return max(1, min(num_bins - 3, int(round(f * self.fft_size / self.sr))))

			bL = freq2bin(band_lower)
			bU = freq2bin(band_upper)

			for i, spectrum in enumerate(self.spectra):
				dBs = np.nanmean(spectrum[bL:bU, :], axis=0)
				# dBs = savgol_filter(dBs, smoothing, 2)
				dBs = uniform_filter1d(dBs, size=smoothing, mode="nearest")
				self.vol_curves.append(dBs)
		self.plot()

	def open_file(self, filepaths):
		for filepath in filepaths:
			self.file_src = filepath
			self.update_spectrum()
			break

	def update_spectrum(self, ):
		if self.file_src:
			try:
				self.spectra, self.sr = spectrum_from_audio_stereo(self.file_src, self.fft_size, self.fft_hop,
																   self.c_channels.currentText(), temporal_mean=False)
				# get the time stamp at which each fft is taken
				self.t = np.arange(0, self.fft_hop * len(self.spectra[0][0]), self.fft_hop) / self.sr
				self.on_param_changed()
			except:
				logging.exception("Failed")

	def onclick(self, event):
		""" Update dB bounds on right click"""
		if event.xdata and event.ydata:
			# right click
			if event.button == 3:
				clip_new = round(event.ydata)
				clip_lower = self.s_clip_lower.value()
				clip_upper = self.s_clip_upper.value()
				middle = (clip_lower + clip_upper) / 2
				if clip_new > middle:
					self.s_clip_upper.setValue(clip_new)
				else:
					self.s_clip_lower.setValue(clip_new)

	def expand(self, ):
		if self.file_src:
			logging.info("Expanding")
			# get input
			clip_lower = self.s_clip_lower.value()
			clip_upper = self.s_clip_upper.value()
			signal, sr, num_channels = io_ops.read_file(self.file_src)
			for channel_i in range(num_channels):
				# map curve to channel output
				if channel_i < len(self.vol_curves):
					dBs = self.vol_curves[channel_i]
				else:
					dBs = self.vol_curves[-1]

				# clip dB curve
				clipped = np.clip(dBs, clip_lower, clip_upper)
				dB_diff = clip_upper - clipped
				fac = units.to_fac(dB_diff)

				# create factor for each sample
				final_fac = np.interp(np.arange(len(signal)), self.t * sr, fac)
				signal[:, channel_i] *= final_fac

			signal = units.normalize(signal)
			io_ops.write_file(self.file_src, signal, sr, num_channels, "_decompressed")

	def plot(self):
		with self.update_plot('Time [s]', 'Input [dB]'):
			if self.spectra:
				# draw clipped curves
				for clipped in self.vol_curves:
					self.ax.plot(self.t, clipped, linewidth=0.5, alpha=0.85)
				# draw bounds
				for bt in (self.s_clip_lower, self.s_clip_upper):
					v = bt.value()
					self.ax.plot((self.t[0], self.t[-1]), (v, v), linestyle="--", color="red", linewidth=0.5, alpha=0.85)
				self.ax.legend(self.c_channels.currentText().split(","), loc='upper left')


if __name__ == '__main__':
	widgets.startup(MainWindow)
