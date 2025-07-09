import logging

import numpy as np
import resampy
import scipy
from PyQt5 import QtWidgets

# custom modules
from util.correlation import find_delay
from util.filters import make_odd
from util.undo import AddAction, DeltaAction
from util import spectrum, qt_threads, widgets, filters, io_ops, markers

from util.config import logging_setup
from util.wow_detection import interp_nans

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
logging_setup()


class MainWindow(widgets.MainWindow):
	EXT = ".tapesync"
	STORE = {"lags": markers.LagSample, "azimuths": markers.AzimuthLine}

	def __init__(self):
		widgets.MainWindow.__init__(self, "pytapesynch", widgets.ParamWidget, Canvas, 2)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		button_data = (
			(file_menu, "Open", self.props.load, "CTRL+O", "dir"),
			(file_menu, "Save", self.props.save, "CTRL+S", "save"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R", "curve"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B", "curve2"),
			(file_menu, "Exit", self.close, "", "exit"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A", "select_extend"),
			(edit_menu, "Improve", self.canvas.improve_lag, "CTRL+I"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL", "x"),
			(edit_menu, "Undo", self.props.undo_stack.undo, "CTRL+Z", "undo"),
			(edit_menu, "Redo", self.props.undo_stack.redo, "CTRL+Y", "redo"),
		)
		self.add_to_menu(button_data)


class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.create_native()
		self.native.setParent(parent)

		self.unfreeze()
		self.parent = parent
		self.lag_line = markers.LagLine(self)

		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.parent.props.display_widget.canvas = self
		self.parent.props.filters_widget.bands_changed.connect(self.lag_line.update_bands)
		self.parent.props.tracing_widget.setVisible(False)
		self.parent.props.dropout_widget.setVisible(False)
		self.freeze()
		self.parent.props.alignment_widget.smoothing_s.valueChanged.connect(self.update_smoothing)

	@property
	def lags(self):
		return [m for m in self.markers if isinstance(m, markers.LagSample)]

	@property
	def azimuths(self):
		return [m for m in self.markers if isinstance(m, markers.AzimuthLine)]

	def update_lines(self):
		self.lag_line.update()

	def update_smoothing(self, k):
		logging.info(f"setting k={k}")
		self.lag_line.smoothing = k
		self.lag_line.update()

	def load_visuals(self, ):
		"""legacy code path"""
		for a0, a1, b0, b1, d in io_ops.read_lag(self.filenames[0]):
			yield markers.LagSample(self, (a0, a1), (b0, b1), d)

	def improve_lag(self):
		deltas = []
		selected = self.selected_markers
		for lag in selected:
			try:
				# prepare some values
				t0, t1, lower, upper = self.get_times_freqs(lag.a, lag.b, self.sr)
				time_delay, lag.corr = self.correlate_sources(t0, t1, lag.d, lower, upper)
				deltas.append(time_delay)
			except:
				logging.exception(f"Refining failed")
		self.props.undo_stack.push(DeltaAction(selected, deltas))
		for trace in selected:
			self.spectra[-1].set_offset(trace.d)
			self.update_corr_view(trace)

	def correlate_sources(self, t0, t1, delay, lower, upper, window_name=None, match_speed=True):
		# todo - this isn't dealing with different sample rates
		sr = self.sr
		t_center = (t0 + t1) / 2
		t_width = (t1 - t0) / 2
		ref, src = self.spectra
		ref_sig = ref.get_signal_around(t_center, t_width)
		# print(f"speed {speed}")
		if match_speed:
			# get rough speed difference for src
			speed = self.get_speed_at(t_center)
			# get respeeded duration around center
			src_sig = src.get_signal_around(t_center - delay, t_width / speed)
			# resample to match expected speed of ref
			src_sig_res = resampy.resample(src_sig, sr / speed, sr, axis=0, filter='sinc_window', num_zeros=8)
			sample_delay_res, corr_res = find_delay(
				filters.butter_bandpass_filter(ref_sig, lower, upper, sr, order=3),
				filters.butter_bandpass_filter(src_sig_res, lower, upper, sr, order=3),
				ignore_phase=self.parent.props.alignment_widget.ignore_phase, window_name=window_name)

			# from matplotlib import pyplot as plt
			# # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
			# plt.plot(np.arange(0, len(ref_sig), 1), ref_sig, label=f"ref_sig", linestyle='-.')
			# # plt.plot(src_sig, label=f"src_sig", linestyle='--')
			# plt.plot(np.arange(0, len(src_sig_res), 1), src_sig_res, label=f"src_sig_res", linestyle='-.')
			# plt.plot(np.arange(0, len(src_sig_res), 1)+sample_delay_res, src_sig_res, label=f"src_sig_res_al", linestyle='-.')
			# plt.vlines(len(ref_sig)/2, -0.1, 0.1, linestyles='--')
			# plt.legend(frameon=True, framealpha=0.75)
			# plt.show()
			# correct delay for speed change
			return sample_delay_res / sr * speed, corr_res
		else:
			src_sig = src.get_signal_around(t_center - delay, t_width)
			# print(f"len(ref_sig) {len(ref_sig)} len(src_sig) {len(src_sig)} len(src_sig_res) {len(src_sig_res)}")
			sample_delay, corr = find_delay(
				filters.butter_bandpass_filter(ref_sig, lower, upper, sr, order=3),
				filters.butter_bandpass_filter(src_sig, lower, upper, sr, order=3),
				ignore_phase=self.parent.props.alignment_widget.ignore_phase, window_name=window_name)
			return sample_delay / sr, corr
		# logging.info(f"corr: raw {corr} vs res {corr_res}")
		# logging.info(f"delay: raw {sample_delay} vs res {sample_delay_res}")
		# logging.info(f"Moved by {sample_delay} samples")

	def run_resample(self):
		self.resample_files((self.filenames[1],))

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg.get("dir_in", ""), "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files(filenames)

	def resample_files(self, files):
		channels = self.props.files_widget.files[1].channel_widget.channels
		if self.filenames[1] and self.markers and channels:
			lag_curve = self.lag_line.data
			self.resampling_thread.settings = {
				"filenames"			: files,
				"lag_curve"			: lag_curve,
				"use_channels"		: channels}
			self.props.resampling_widget.bump_index()
			self.props.resampling_widget.to_cfg(self.resampling_thread.settings)
			self.resampling_thread.start()
		
	def on_mouse_press(self, event):
		# selection
		if event.button == 2:
			closest_marker = self.get_closest(self.markers, event.pos)
			if closest_marker:
				closest_marker.select_handle("Shift" in event.modifiers)
				self.update_corr_view(closest_marker)
				event.handled = True
			# update the last spectrum with pan
			click = self.px_to_spectrum(event.pos)
			if click is not None:
				# sample the lag curve at the click's time and move the source spectrum
				self.spectra[-1].set_offset(self.lag_line.sample_at((click[0],))[0][0])

	def update_corr_view(self, closest_marker):
		v = "None" if closest_marker.corr is None else f"{closest_marker.corr:.3f}"
		self.parent.props.alignment_widget.corr_l.setText(v)

	def get_speed_at(self, t):
		width = 0.05
		# calc speed across range
		data = self.lag_line.data
		# smooth / lowpass lag curve to get a better derivative
		filtered = data[:, 1]
		filtered = filters.butter_bandpass_filter(filtered, 0, 15, self.lag_line.marker_sr, order=3)
		# for input in t, sample lag at t+-range (0.5 s?)
		before = np.interp(t-width, data[:, 0], filtered)
		after = np.interp(t+width, data[:, 0], filtered)
		speed = (after - before) / (2 * width) + 1.0
		logging.info(f"Source runs {(speed-1)*100:0.2f}% wrong")
		return speed
		# from matplotlib import pyplot as plt
		# plt.plot(filtered, label=f"filtered")
		# plt.plot(data[:, 1], label=f"raw")
		# plt.legend(frameon=True, framealpha=0.75)
		# plt.show()

	def on_mouse_release(self, event):
		# coords of the click on the vispy canvas
		if self.filenames[1] and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.px_to_spectrum(last_click)
				b = self.px_to_spectrum(click)
				# are they in spec_view?
				if a is not None and b is not None:
					if "Control" in event.modifiers:
						d = b[0]-a[0]
						self.spectra[1].translate(d)
					elif "Shift" in event.modifiers:
						marker = markers.LagSample(self, a, b)
						self.props.undo_stack.push(AddAction((marker,)))
					elif "Alt" in event.modifiers:
						logging.info("Azimuth mode")
						sr = self.sr
						dur = self.parent.props.alignment_widget.win_s.value()
						overlap = self.parent.props.alignment_widget.overlap_s.value()
						reject = self.parent.props.alignment_widget.reject_s.value()
						# first get the time range for selection
						ref_t0, ref_t1, lower, upper = self.get_times_freqs(a, b, sr)

						sample_times = np.arange(ref_t0, ref_t1, dur/overlap)
						if not len(sample_times):
							return
						data = self.lag_line.data
						# get the current lag for each time we want to sample
						sample_lags = np.interp(sample_times, data[:, 0], data[:, 1])

						out = np.zeros((len(sample_times), 2), dtype=np.float64)
						corrs = np.zeros(len(sample_times), dtype=np.float64)
						out[:, 0] = sample_times
						# apply bandpass
						# split into pieces and look up the delay for each
						# correlate all the pieces
						for i, (x, d) in enumerate(zip(sample_times, sample_lags)):
							time_delay, corr = self.correlate_sources(x-dur, x+dur, d, lower, upper, "hann")
							corrs[i] = corr
							# reject if correlation is too weak
							if abs(corr) < reject:
								out[i, 1] = np.NaN
							else:
								out[i, 1] = d+time_delay
						# lerp rejected lags
						interp_nans(out[:, 1])
						# filter outliers
						out[:, 1] = scipy.ndimage.median_filter(out[:, 1], size=make_odd(overlap), footprint=None, output=None, mode='nearest', cval=0.0, origin=0,)
						marker = markers.AzimuthLine(self, out[:, 0], out[:, 1], corrs, lower, upper)
						self.props.undo_stack.push(AddAction((marker,)))

	def get_times_freqs(self, a, b, sr):
		ref_t0, ref_t1 = sorted((a[0], b[0]))
		freqs = sorted((a[1], b[1]))
		lower = max(freqs[0], 1)
		upper = min(freqs[1], sr // 2 - 1)
		return ref_t0, ref_t1, lower, upper


if __name__ == '__main__':
	widgets.startup(MainWindow)
