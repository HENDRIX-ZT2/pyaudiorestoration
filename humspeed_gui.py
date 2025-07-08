import logging

import numpy as np

import util.correlation
from util.widgets import PlotMainWindow

try:
	import resampy
except:
	logging.warning("Resampy is not installed. In the commandline, run: pip install resampy")
	resampy = None
from PyQt5 import QtWidgets
from util import fourier, io_ops, widgets, config
from util.spectrum_flat import spectrum_from_audio


def get_spectrum(file_src, channel_mode, fft_size):
	logging.info(f"Analyzing channels: {channel_mode}")
	# get the averaged spectrum for this audio file
	hop = fft_size * 2
	spectrum, sr = spectrum_from_audio(file_src, fft_size, hop, channel_mode)
	freqs = fourier.fft_freqs(fft_size, sr)
	return freqs, spectrum, sr


class MainWindow(PlotMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)

		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)

		self.setWindowTitle('Hum Speed Analysis')
		self.file_src = ""
		self.freqs = None
		self.spectrum = None
		# self.fft_size = 131072
		# self.fft_size = 262144  # 2**18
		self.fft_size = 524288  # 2**19
		self.marker_freqs = []
		self.marker_dBs = []
		self.ratios = []
		self.hum_freqs = []
		self.cfg = config.load_config()

		self.cb = QtWidgets.QApplication.clipboard()

		self.files_widget = widgets.FilesWidget(self, 1, self.cfg)
		self.files_widget.on_load_file = self.open_file

		self.b_resample = QtWidgets.QPushButton('Resample')
		self.b_resample.setToolTip("Write speed-corrected audio to a new file.")
		self.b_resample.clicked.connect(self.resample)

		self.c_channels = QtWidgets.QComboBox(self)
		self.c_channels.addItems(list(("L+R", "L", "R")))
		self.c_channels.setToolTip("Which channels should be analyzed?")
		self.s_base_hum = QtWidgets.QSpinBox()
		self.s_base_hum.setRange(10, 500)
		self.s_base_hum.setSingleStep(10)
		self.s_base_hum.setValue(50)
		self.s_base_hum.setSuffix(" Hz")
		self.s_base_hum.setToolTip("Base frequency of the hum.")

		self.s_num_harmonies = QtWidgets.QSpinBox()
		self.s_num_harmonies.setRange(0, 5)
		self.s_num_harmonies.setSingleStep(1)
		self.s_num_harmonies.setValue(2)
		self.s_num_harmonies.setToolTip("Number of hum harmonies to consider.")

		self.s_tolerance = QtWidgets.QSpinBox()
		self.s_tolerance.setRange(1, 20)
		self.s_tolerance.setSingleStep(1)
		self.s_tolerance.setValue(8)
		self.s_tolerance.setSuffix(" %")
		self.s_tolerance.setToolTip("Maximum derivation from target hum frequency to look for peaks in spectrum.")

		self.l_result = QtWidgets.QLabel("Result: ")

		self.qgrid = QtWidgets.QGridLayout()
		# self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.toolbar, 0, 0, 1, 7)
		self.qgrid.addWidget(self.canvas, 1, 0, 1, 7)
		self.qgrid.addWidget(self.files_widget, 2, 0)
		self.qgrid.addWidget(self.b_resample, 2, 1)
		self.qgrid.addWidget(self.c_channels, 2, 2)
		self.qgrid.addWidget(self.s_base_hum, 2, 3)
		self.qgrid.addWidget(self.s_num_harmonies, 2, 4)
		self.qgrid.addWidget(self.s_tolerance, 2, 5)
		self.qgrid.addWidget(self.l_result, 2, 6)

		self.central_widget.setLayout(self.qgrid)

		for btn in (self.s_base_hum, self.s_num_harmonies, self.s_tolerance):
			btn.valueChanged.connect(self.on_hum_param_changed)
		self.c_channels.currentIndexChanged.connect(self.update_spectrum)

	def on_hum_param_changed(self, ):

		base_hum = self.s_base_hum.value()
		num_harmonies = self.s_num_harmonies.value()
		self.hum_freqs = np.arange(base_hum, base_hum + base_hum * num_harmonies + 1, base_hum)
		self.marker_freqs = []
		self.marker_dBs = []
		self.ratios = []
		for hum_freq in self.hum_freqs:
			self.track_to(hum_freq)
			self.plot()

	def open_file(self, filepaths):
		for filepath in filepaths:
			self.file_src = filepath
			try:
				self.update_spectrum()
			except:
				logging.exception("Failed")
			break

	def update_spectrum(self, ):
		if self.file_src:
			self.freqs, self.spectrum, self.sr = get_spectrum(self.file_src, self.c_channels.currentText(),
															  self.fft_size)
			self.on_hum_param_changed()
			self.plot()

	def onclick(self, event):
		if event.xdata and event.ydata:
			# right click
			if event.button == 3:
				self.marker_freqs = []
				self.marker_dBs = []
				self.ratios = []
				self.track_to(event.xdata)
				self.plot()

	def track_to(self, xpos):
		if self.freqs is not None:
			# a constant around the click; maybe make this into a log2-based value?
			tolerance = self.s_tolerance.value()
			l_ratio = 1 - tolerance / 100
			r_ratio = 1 + tolerance / 100

			# get the closest index to the click's x position in the frequency array

			border_L = max(np.argmin(np.abs(self.freqs - xpos * l_ratio)), 0)
			border_R = min(np.argmin(np.abs(self.freqs - xpos * r_ratio)), len(self.freqs))

			# get peak from the selected region 
			raw_index = np.argmax(self.spectrum[border_L:border_R]) + border_L
			interp_index, dB = util.correlation.parabolic(self.spectrum, raw_index)

			# convert to frequency
			freq = interp_index * self.sr / self.fft_size

			# find closest index and hum freq
			closest_index = np.argmin(np.abs(self.hum_freqs - freq))
			closest_hum = self.hum_freqs[closest_index]

			# todo: tolerance should be a lot smaller
			# get percentage
			ratio = closest_hum / freq
			percent = (ratio - 1) * 100
			# is it close enough?
			if abs(percent) > tolerance:
				self.l_result.setText("hum was not close enough")
				return

			# store data
			self.marker_freqs.append(freq)
			self.marker_dBs.append(dB)
			self.ratios.append(ratio)

			# format as string
			percent_str = "%.3f" % percent

			# set to debug label
			self.l_result.setText("Percent Change: " + percent_str)

			# copy to clipboard
			self.cb.clear(mode=self.cb.Clipboard)
			self.cb.setText(percent_str, mode=self.cb.Clipboard)

	def resample(self, ):
		if self.file_src and self.ratios:
			if resampy is None:
				logging.warning("Can't resample without resampy!")
				return
			logging.info("Resampling")
			# get input
			ratio = self.ratios[-1]
			percentage = (ratio - 1) * 100

			signal, sr, num_channels = io_ops.read_file(self.file_src)
			# resample, first axis is time!
			res = resampy.resample(signal, sr * ratio, sr, axis=0, filter='sinc_window', num_zeros=8)
			io_ops.write_file(self.file_src, res, sr, num_channels, "_resampled_%.3f" % percentage)

	def plot(self):
		with self.update_plot('Frequency (Hz)', 'Volume (dB)'):
			if self.freqs is not None:
				# cutoff view at 500Hz
				end = np.argmin(np.abs(self.freqs - 500))
				self.ax.plot(self.freqs[:end], self.spectrum[:end], linewidth=0.5, alpha=0.85)
				self.ax.plot(self.marker_freqs, self.marker_dBs, 'b+', ms=14)


if __name__ == '__main__':
	widgets.startup(MainWindow)
