import logging

import matplotlib
import numpy as np
import os
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from util import fourier, filters, widgets, config
from util.spectrum_flat import spectrum_from_audio_stereo


# todo: make global sr set by the first file that is loaded, make all others fit
from util.widgets import PlotMainWindow


def write_eq_txt(file_path, freqs, dB):
	with open(file_path, "w") as out:
		out.write('FilterCurve: FilterLength="8191" InterpolateLin="0" InterpolationMethod="B-spline" ')
		for i, (f, d) in enumerate(zip(freqs, dB)):
			out.write(f'f{i}="{f}" ')
			out.write(f'v{i}="{d}" ')


def get_eq(file_src, file_ref, channel_mode):
	print("Comparing channels:", channel_mode)
	# get the averaged spectrum for this audio file
	fft_size = 16384
	hop = 8192
	# todo: set custom times for both, if given
	spectra_src, sr_src = spectrum_from_audio_stereo(file_src, fft_size, hop, channel_mode)
	spectra_ref, sr_ref = spectrum_from_audio_stereo(file_ref, fft_size, hop, channel_mode)

	freqs = fourier.fft_freqs(fft_size, sr_src)
	# resample the ref spectrum to match the source
	if sr_src != sr_ref:
		for channel_i, spectrum in enumerate(spectra_ref):
			spectra_ref[channel_i] = np.interp(freqs, fourier.fft_freqs(fft_size, sr_ref), spectrum)
	return freqs, np.asarray(spectra_ref) - np.asarray(spectra_src)


class MainWindow(PlotMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)

		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)

		self.setWindowTitle('Differential EQ')
		self.names = []
		self.freqs = None
		self.eqs = []
		self.av = []
		self.freqs_av = []
		self.cfg = config.load_config()

		# Just some button connected to `plot` method
		self.files_widget = widgets.FilesWidget(self, 2, self.cfg, ask_user=False)
		self.files_widget.on_load_file = self.foo
		self.b_add = QtWidgets.QPushButton('Add EQ')
		self.b_add.setToolTip("Add a source - reference pair to the list.")
		self.b_add.clicked.connect(self.open)
		self.b_delete = QtWidgets.QPushButton('Delete EQ')
		self.b_delete.setToolTip("Delete the selected source - reference pair from the list.")
		self.b_delete.clicked.connect(self.delete)
		self.b_save = QtWidgets.QPushButton('Export EQ')
		self.b_save.setToolTip("Write the average EQ curve to an XML file.")
		self.b_save.clicked.connect(self.write)

		self.l_highpass = QtWidgets.QLabel("Highpass")
		self.s_highpass = QtWidgets.QSpinBox()
		self.s_highpass.valueChanged.connect(self.plot)
		self.s_highpass.setRange(0, 6000)
		self.s_highpass.setSingleStep(500)
		self.s_highpass.setValue(0)
		self.s_highpass.setToolTip("Do not influence under this frequency")

		self.l_rolloff_start = QtWidgets.QLabel("Rolloff Start")
		self.s_rolloff_start = QtWidgets.QSpinBox()
		self.s_rolloff_start.valueChanged.connect(self.plot)
		self.s_rolloff_start.setRange(0, 22000)
		self.s_rolloff_start.setSingleStep(1000)
		self.s_rolloff_start.setValue(21000)
		self.s_rolloff_start.setToolTip("At this frequency, the EQ still has full influence.")

		self.l_rolloff_end = QtWidgets.QLabel("Rolloff Ende")
		self.s_rolloff_end = QtWidgets.QSpinBox()
		self.s_rolloff_end.valueChanged.connect(self.plot)
		self.s_rolloff_end.setRange(0, 22000)
		self.s_rolloff_end.setSingleStep(1000)
		self.s_rolloff_end.setValue(22000)
		self.s_rolloff_end.setToolTip("At this frequency, the effect of the EQ becomes zero.")

		self.l_channels = QtWidgets.QLabel("Channels")
		self.c_channels = QtWidgets.QComboBox(self)
		self.c_channels.addItems(list(("L+R", "L", "R")))
		self.c_channels.setToolTip("Which channels should be analyzed?")

		self.l_output_res = QtWidgets.QLabel("Resolution")
		self.s_output_res = QtWidgets.QSpinBox()
		self.s_output_res.valueChanged.connect(self.plot)
		self.s_output_res.setRange(20, 2000)
		self.s_output_res.setSingleStep(100)
		self.s_output_res.setValue(200)
		self.s_output_res.setToolTip("Resolution of the output curve.")

		self.l_smoothing = QtWidgets.QLabel("Smoothing")
		self.s_smoothing = QtWidgets.QSpinBox()
		self.s_smoothing.valueChanged.connect(self.plot)
		self.s_smoothing.setRange(1, 200)
		self.s_smoothing.setSingleStep(10)
		self.s_smoothing.setValue(50)
		self.s_smoothing.setToolTip("Increase this if your sample size is small or the speed match between source and reference is not perfect.")

		self.l_strength = QtWidgets.QLabel("Strength")
		self.s_strength = QtWidgets.QSpinBox()
		self.s_strength.valueChanged.connect(self.plot)
		self.s_strength.setRange(10, 150)
		self.s_strength.setSingleStep(10)
		self.s_strength.setValue(100)
		self.s_strength.setSuffix(" %")
		self.s_strength.setToolTip("Adjust the strength of the output EQ curve.")

		self.c_gain = QtWidgets.QCheckBox("Keep Gain")
		self.c_gain.setToolTip("If checked, the original gain remains untouched.")

		self.listWidget = QtWidgets.QListWidget()

		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.toolbar, 0, 0, 1, 3)
		self.qgrid.addWidget(self.canvas, 1, 0, 1, 3)
		self.qgrid.addWidget(self.listWidget, 2, 0, 9, 1)
		self.qgrid.addWidget(self.files_widget, 2, 1, 1, 2)
		self.qgrid.addWidget(self.b_add, 3, 1, 1, 2)
		self.qgrid.addWidget(self.b_delete, 4, 1, 1, 2)
		self.qgrid.addWidget(self.b_save, 5, 1, 1, 2)

		self.qgrid.addWidget(self.l_highpass, 6, 1)
		self.qgrid.addWidget(self.s_highpass, 6, 2)
		self.qgrid.addWidget(self.l_rolloff_start, 7, 1)
		self.qgrid.addWidget(self.s_rolloff_start, 7, 2)
		self.qgrid.addWidget(self.l_rolloff_end, 8, 1)
		self.qgrid.addWidget(self.s_rolloff_end, 8, 2)
		self.qgrid.addWidget(self.l_channels, 9, 1)
		self.qgrid.addWidget(self.c_channels, 9, 2)
		self.qgrid.addWidget(self.l_output_res, 10, 1)
		self.qgrid.addWidget(self.s_output_res, 10, 2)
		self.qgrid.addWidget(self.l_smoothing, 11, 1)
		self.qgrid.addWidget(self.s_smoothing, 11, 2)
		self.qgrid.addWidget(self.l_strength, 12, 1)
		self.qgrid.addWidget(self.s_strength, 12, 2)
		self.qgrid.addWidget(self.c_gain, 13, 1)

		self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		self.central_widget.setLayout(self.qgrid)

	def foo(self, filepaths):
		pass

	def open(self, ):
		filepaths = self.files_widget.filepaths
		if filepaths and len(filepaths) == 2:
			try:
				file_src = filepaths[1]
				file_ref = filepaths[0]
				src_name = os.path.basename(file_src)
				ref_name = os.path.basename(file_ref)
				channel_mode = self.c_channels.currentText()
				eq_name = f"{src_name} ({channel_mode}) -> {ref_name} ({channel_mode})"
				self.freqs, eq = get_eq(file_src, file_ref, channel_mode)
				self.listWidget.addItem(eq_name)
				self.names.append(eq_name)
				self.eqs.append(eq)
				self.update_color(eq_name)
				self.plot()
			except:
				logging.exception(f"Loading EQ curve failed")

	def update_color(self, eq_name):
		item = self.listWidget.findItems(eq_name, QtCore.Qt.MatchFixedString)[-1]
		# don't include the first (blue) -> reserved for the bold line
		item.setForeground(QtGui.QColor(self.colors[self.names.index(eq_name) + 1]))

	def delete(self):
		for item in self.listWidget.selectedItems():
			for i in reversed(range(0, len(self.names))):
				if self.names[i] == item.text():
					self.names.pop(i)
					self.eqs.pop(i)
			self.listWidget.takeItem(self.listWidget.row(item))

		for eq_name in self.names:
			self.update_color(eq_name)
		self.plot()

	def write(self):
		try:
			file_out = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Average EQ', self.cfg.get("dir_out", "C:/"), "TXT files (*.txt)")[0]
			file_base = ".".join(file_out.split(".")[:-1])
			if file_out:
				try:
					self.cfg["dir_out"], eq_name = os.path.split(file_out)
					write_eq_txt(f"{file_base}.txt", self.freqs_av, np.mean(self.av, axis=0))
					write_eq_txt(f"{file_base}_L.txt", self.freqs_av, self.av[0])
					write_eq_txt(f"{file_base}_R.txt", self.freqs_av, self.av[1])
				except PermissionError:
					widgets.showdialog("Could not write files - do you have writing permissions there?")
		except:
			logging.exception(f"Saving failed")

	def plot(self):
		with self.update_plot('Frequency (Hz)', 'Volume Change (dB)'):
			if self.freqs is not None:
				# todo: just calculate it from SR and bin count
				# again, just show from 20Hz
				from20Hz = (np.abs(self.freqs - 20)).argmin()

			if self.names:
				num_in = 2000
				# average over n samples, then reduce the step according to the desired output
				n = self.s_smoothing.value()
				num_out = self.s_output_res.value()
				reduction_step = num_in // num_out
				# take the average curve of all differential EQs
				av_in = np.mean(np.asarray(self.eqs), axis=0)
				highpass = self.s_highpass.value()
				rolloff_start = self.s_rolloff_start.value()
				rolloff_end = self.s_rolloff_end.value()

				# audacity EQ starts at 20Hz
				freqs_spaced = np.power(2, np.linspace(np.log2(20), np.log2(self.freqs[-1]), num=num_in))

				avs = []
				# smoothen the curves, and reduce the points with step indexing
				self.freqs_av = filters.moving_average(freqs_spaced, n=n)[::reduction_step]
				for channel in (0, 1):
					# interpolate this channel's EQ, then smoothen and reduce keys for this channel
					avs.append(
						filters.moving_average(np.interp(freqs_spaced, self.freqs, av_in[channel]), n=n)[::reduction_step])
				self.av = np.asarray(avs)

				# get the gain of the filtered  EQ
				idx1 = np.abs(self.freqs_av - 70).argmin()
				idx2 = np.abs(self.freqs_av - rolloff_end).argmin()
				gain = np.mean(self.av[:, idx1:idx2])
				strength = self.s_strength.value() / 100
				if self.c_gain.isChecked():
					self.av -= gain
				self.av *= strength

				# fade out
				for channel in (0, 1):
					# todo make rolloff_end a band parameter in octaves?
					self.av[channel] *= np.interp(self.freqs_av, (rolloff_start, rolloff_end), (1, 0))
					self.av[channel] *= np.interp(self.freqs_av, (0, highpass), (0, 1))

				if tuple(int(x) for x in matplotlib.__version__.split(".")) >= (3, 5, 0):
					kwargs = {"base": 2}
				else:
					kwargs = {"basex": 2}
				# plot the contributing raw curves
				for name, eq in zip(self.names, np.mean(np.asarray(self.eqs), axis=1)):
					self.ax.semilogx(self.freqs[from20Hz:], eq[from20Hz:], linestyle="--", linewidth=.5, alpha=.5,
									 color=self.colors[self.names.index(name) + 1], **kwargs)
				# take the average
				self.ax.semilogx(self.freqs_av, np.mean(self.av, axis=0), linewidth=2.5, alpha=1,
								 color=self.colors[0], **kwargs)
				# ticks = np.arange(0, 22000, step=1000)
				# self.ax.set_xticks(ticks, labels=ticks)


if __name__ == '__main__':
	widgets.startup(MainWindow)
