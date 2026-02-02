import contextlib
import os

import numpy as np
import logging

import resampy
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QTimer
from anyio import sleep
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# custom modules
from util import spectrum, wow_detection, qt_threads, widgets, io_ops, markers, fourier, vispy_ext
from util.fourier import timed_log, to_mag

from util.undo import AddAction, MoveAction, MergeAction
from util.config import logging_setup
from util.units import to_dB, to_fac, to_mel, to_Hz
from util.widgets import vbox2
from util.wow_detection import wow_detectors

logging_setup()
HEAD = 3

class MainWindow(widgets.MainWindow):
	EXT = ".noise"
	search_text = pyqtSignal(str)

	def __init__(self):
		widgets.MainWindow.__init__(self, "renoiser", widgets.ParamWidget, Canvas, 1)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		button_data = (
			(file_menu, "Open", self.props.load, "CTRL+O", "dir"),
			(file_menu, "Save", self.props.save, "CTRL+S", "save"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R", "curve"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B", "curve2"),
			(file_menu, "Exit", self.close, "", "exit"),
			(edit_menu, "Play/Pause", self.props.audio_widget.play_pause, "SPACE"),
			(edit_menu, "Load Noise Profile", self.canvas.load_noise_profile, "CTRL+N"),
		)
		self.add_to_menu(button_data)


		self.typing_timer = QTimer()
		self.typing_timer.setSingleShot(True)
		self.typing_timer.timeout.connect(self.timer_up)
		# self.entry.textChanged.connect(self.search_text_changed)
		self.search_text.connect(self.canvas.update_renoised)
		self.setup_plot()

		# setup splitter
		w = QtWidgets.QWidget(self)

		qgrid = QtWidgets.QGridLayout()
		qgrid.setHorizontalSpacing(0)
		qgrid.setVerticalSpacing(0)
		qgrid.addWidget(self.toolbar, 0, 0)
		qgrid.addWidget(self.mpl_canvas, 1, 0)
		w.setLayout(qgrid)

		splitter1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		splitter1.addWidget(self.canvas.native)
		splitter1.addWidget(w)
		splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		splitter.addWidget(splitter1)
		splitter.addWidget(self.props)
		self.setCentralWidget(splitter)


	def search_text_changed(self) -> None:
		# wait for 250 ms before
		if self.typing_timer.isActive():
			self.typing_timer.stop()
		self.typing_timer.start(1000)

	def timer_up(self) -> None:
		# emit the text on the signal
		self.search_text.emit("Nope")

	def setup_splitter(self):
		# comment out and put into init to add mpl_canvas
		pass

	def setup_plot(self):
		# a figure instance to plot on
		self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
		self.ax2 = self.ax.twinx()  # instantiate a second Axes that shares the same x-axis

		# color = 'tab:blue'
		self.ax2.set_ylabel('Correction (dB)')  # we already handled the x-label with ax1
		# ax2.plot(t, data2, color=color)
		# ax2.tick_params(axis='y', labelcolor=color)

		self.ax.set_xlabel('Frequency (Hz)')
		self.ax.set_ylabel('Gain (dB)')
		# the range is not automatically fixed
		self.fig.patch.set_facecolor((53 / 255, 53 / 255, 53 / 255))
		self.ax.set_facecolor((35 / 255, 35 / 255, 35 / 255))
		# this is the Canvas Widget that displays the `figure`
		# it takes the `fig` instance as a parameter to __init__
		self.mpl_canvas = FigureCanvas(self.fig)
		# self.mpl_canvas.mpl_connect('button_press_event', self.onclick)
		# this is the Navigation widget
		# it takes the Canvas widget and a parent
		self.toolbar = NavigationToolbar(self.mpl_canvas, self)

		self.circles = []
		self.mousepress = None
		self.currently_dragging = False
		self.current_index = None
		# tuples of x, y
		self.curve = [[1.0, 0.0], [self.canvas.sr / 2, 0.0]]
		control_x, control_y = zip(*self.curve)
		self.line_control = self.ax2.plot(control_x, control_y, '--o', alpha=0.5, c='r', lw=2, picker=True, label="Control")[0]
		self.line_noise =  self.ax.plot(self.canvas.freqs, np.zeros(len(self.canvas.freqs)), linewidth=2.5, alpha=1, label="Floor")[0]
		self.line_final = self.ax.plot(self.canvas.freqs, np.zeros(len(self.canvas.freqs)), linewidth=2.5, alpha=1, label="Cutoff")[0]
		self.ax.set_xscale("function", functions=(to_mel, to_Hz))
		lns = (self.line_noise, self.line_control, self.line_final)
		self.ax.legend(lns, [l.get_label() for l in lns], loc="upper right")
		self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.fig.canvas.mpl_connect('button_press_event', self.on_press)
		self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.fig.canvas.mpl_connect('pick_event', self.on_pick)
		self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		self.canvas.redraw_plot()

	def on_press(self, event):
		self.currently_dragging = True
		if event.button == 3:
			self.mousepress = "right"
		elif event.button == 1:
			self.mousepress = "left"

	def on_release(self, event):
		self.current_index = None
		self.currently_dragging = False

	def on_pick(self, event):
		click = np.array((event.mouseevent.xdata, event.mouseevent.ydata))

		self.current_index = np.sum((np.array(self.curve) - click) ** 2, axis=1).argmin()
		# delete existing circle
		if event.mouseevent.dblclick:
			if event.mouseevent.button == 3:  # right
				if 0 < self.current_index < len(self.curve) - 1:
					# print(f"deleting {self.current_index}")
					self.curve.pop(self.current_index)
					self.canvas.redraw_plot()

	def on_motion(self, event):
		if not self.currently_dragging:
			return
		if self.current_index is None:
			return
		if event.xdata is None:
			return
		if self.current_index is not None:
			if self.current_index == 0 or self.current_index == len(self.curve) - 1:
				self.curve[self.current_index][1] = event.ydata
			else:
				self.curve[self.current_index] = [event.xdata, event.ydata]
			self.canvas.redraw_plot()

	def on_click(self, event):
		if event and event.dblclick and self.mousepress == "left":
			self.curve.append([event.xdata, event.ydata])
			self.canvas.redraw_plot()


class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, spectra_colors=(None,), y_axis='Octaves')
		self.create_native()
		self.native.setParent(parent)

		self.unfreeze()
		self.parent = parent


		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.canvas = self
		self.parent.props.noise_widget.canvas = self
		self.parent.props.alignment_widget.setVisible(False)
		self.parent.props.dropout_widget.setVisible(False)
		self.parent.props.tracing_widget.setVisible(False)
		self.parent.props.stack_widget.setVisible(False)
		self.parent.props.noise_widget.gain_s.valueChanged.connect(self.redraw_plot)
		self.parent.props.noise_widget.overhead_s.valueChanged.connect(self.redraw_plot)
		self.parent.props.display_widget.fft_zeropad = 1
		self.parent.props.display_widget.update_fft_settings()

		self.noise_profile = np.empty(len(self.freqs), dtype=np.float32)
		self.noise_profile[:] = -100.0
		self.final_profile = self.noise_profile.copy()
		self.freeze()

	def setup_grid(self, label_color):
		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0
		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 10
		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 55
		# grid.add_widget(self.speed_yaxis, row=1, col=0)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		grid.add_widget(self.colorbar_display, row=2, col=2)
		# self.speed_view = grid.add_view(row=1, col=1, border_color=label_color)
		# self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, -5, 10, 10), )
		# self.speed_view.height_min = 150
		# self.speed_view.stretch = (1, 1)
		self.spec_view = grid.add_view(row=2, col=1, border_color=label_color)
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		self.spec_view.height_min = 550
		self.spec_view.stretch = (2, 2)
		# link them, but use custom logic to only link the x view
		# self.spec_view.camera.link(self.speed_view.camera)
		# self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)
		self.views = [self.spec_view, ]

	@property
	def freqs(self):
		return fourier.fft_freqs(self.fft_size, self.sr)

	def load_noise_profile(self):
		file_path = \
		QtWidgets.QFileDialog.getOpenFileName(self.parent, 'Load Noise', self.parent.cfg.get("dir_in", "C:/"),
											  self.parent.props.sel_str)[0]
		if not os.path.isfile(file_path):
			return
		noise, noise_sr, noise_num_channels = io_ops.read_file(file_path)
		noise_res = resampy.resample(noise, noise_sr, self.sr, axis=0, filter='sinc_window', num_zeros=8)

		fft_noise = fourier.get_mag(noise_res[:, 0], self.fft_size, self.hop, "blackmanharris", zeropad=self.zeropad)
		db_noise = to_dB(fft_noise)
		self.noise_profile = np.average(db_noise, axis=1)
		self.redraw_plot()

	def update_renoised(self, test="No"):

		for spectrum in self.spectra:
			spectrum.key = (self.fft_size, spectrum.selected_channel, self.hop, self.zeropad)
			# first try to get FFT from current storage and continue directly
			if spectrum.key in spectrum.fft_storage:
				# this happens when only loading from storage is required
				self.update_spectrum(spectrum)

	def update_spectrum(self, spec):
		try:
			spec_data = spec.fft_storage[spec.key]
			fac = self.get_mask_fac(spec_data)
			fft_masked = np.asarray(spec_data, dtype=np.float32) * fac
			spec.update_data(fft_masked, self.hop)
		except:
			logging.exception(f"Updating spectrum failed")
		spec.set_clims(self.vmin, self.vmax)
		spec.set_cmap(self.cmap)

	def get_mask_fac(self, spec_data):
		# add a second axis so the arrays are compatible
		expanded_axis = np.expand_dims(self.final_profile, axis=1)
		gain_mask = np.where(np.array(to_dB(spec_data)) > expanded_axis, 0.0, self.parent.props.noise_widget.gain)
		fac = to_fac(gain_mask).astype(dtype=np.float32)
		return fac

	def redraw_plot(self, ):
		self.parent.curve.sort()
		self.parent.line_control.set_data(*zip(*self.parent.curve))
		y_values = np.array(self.parent.curve)[:, 1]
		self.parent.ax2.set_ylim(np.min(y_values) - HEAD, np.max(y_values) + HEAD)

		control_x, control_y = zip(*self.parent.curve)

		control_interp = np.interp(self.freqs, control_x, control_y)
		self.parent.line_noise.set_data(self.freqs, self.noise_profile)
		self.final_profile = self.noise_profile + self.parent.props.noise_widget.gain + control_interp + self.parent.props.noise_widget.overhead
		self.parent.line_final.set_data(self.freqs, self.final_profile)
		self.parent.ax.set_ylim(np.min(self.noise_profile) - HEAD, np.max(self.final_profile) + HEAD)
		self.parent.mpl_canvas.draw()
		self.parent.search_text_changed()

	def run_resample(self):
		spec = self.spectra[0]
		if spec.audio_path:
			channels = self.props.files_widget.files[0].channel_widget.channels
			if channels:
				# self.resampling_thread.settings = {
				# 	"filenames": (spec.audio_path,),
				# 	"signal_data": ((spec.signal, spec.sr),),
				# 	"speed_curve": self.get_speed_curve(),
				# 	"use_channels": channels}
				# self.props.output_widget.bump_index()
				# self.props.output_widget.to_cfg(self.resampling_thread.settings)
				# self.resampling_thread.start()
				n = len(spec.signal)
				signal_res_pad = fourier.fix_length(spec.signal, n + self.fft_size // 2, axis=0)
				y_out = np.empty((n, len(channels)), spec.signal.dtype)
				for channel_i in channels:
					# take FFT for each channel
					fft_signal = np.array(fourier.stft(signal_res_pad[:, channel_i], n_fft=self.fft_size, step=self.hop))

					fac = self.get_mask_fac(to_mag(fft_signal))
					y_out[:, channel_i] = fourier.istft(fft_signal * fac, length=n, hop_length=self.hop)

				io_ops.write_file(spec.audio_path, y_out, self.sr, len(channels), suffix=f" fft={self.fft_size}")

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]


	def on_mouse_release(self, event):
		if self.filenames[0] and (event.trail() is not None) and event.button == 1 and "Control" in event.modifiers:
			# event.trail() is integer pixel coordinates on vispy canvas
			trail = [self.px_to_spectrum(click) for click in event.trail()]
			# filter out any clicks outside of spectrum, for which px_to_spectrum returns None
			trail = [x for x in trail if x is not None]
			if trail:
				# settings = self.props.tracing_widget
				for spectrum in self.spectra:
					spec_data = spectrum.fft_storage[spectrum.key]
					num_bins, last_fft_i = spec_data.shape
					t0 = trail[0][0]
					t1 = trail[-1][0]
					f0 = int(t0 * self.sr / self.hop)
					f1 = int(t1 * self.sr / self.hop)
					f0 = max(0, f0)
					f1 = min(f1, last_fft_i-1)
					self.noise_profile = to_dB(np.average(spec_data[:, f0:f1], axis=1))
					self.redraw_plot()



if __name__ == '__main__':
	widgets.startup(MainWindow)
