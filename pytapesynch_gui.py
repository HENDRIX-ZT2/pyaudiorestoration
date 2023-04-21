import logging
import os

import numpy as np
from PyQt5 import QtWidgets

# custom modules
from util.correlation import xcorr
from util.undo import AddAction, DeleteAction, DeltaAction
from util import spectrum, wow_detection, qt_threads, widgets, filters, io_ops, \
	markers

from util.config import save_config_json, load_config_json, logging_setup

logging_setup()
EXT = ".tapesync"


class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pytapesynch", widgets.ParamWidget, Canvas, 2)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		button_data = (
			(file_menu, "Open", self.props.files_widget.ask_open, "CTRL+O", "dir"),
			(file_menu, "Save", self.canvas.save_traces, "CTRL+S", "save"),
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
		self.lag_samples = []
		self.lag_line = markers.LagLine(self)

		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_widget.onProgress)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_widget.onProgress)
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.setVisible(False)
		self.freeze()
		self.parent.props.alignment_widget.smoothing_s.valueChanged.connect(self.update_smoothing)

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

	def load_project(self):
		"""Load project with all required settings"""
		cfg_path = QtWidgets.QFileDialog.getOpenFileName(self.parent, 'Open Project', self.parent.cfg["dir_in"],
														 f"Tape Sync project files (*{EXT})")[0]
		if os.path.isfile(cfg_path):
			sync = load_config_json(cfg_path)
			self.parent.props.display_widget.fft_size = sync["fft_size"]
			self.parent.props.display_widget.fft_overlap = sync["fft_overlap"]
			self.parent.props.alignment_widget.smoothing = sync["smoothing"]
			for file_widget, fp in zip(self.parent.props.files_widget.files, (sync["ref"], sync["src"])):
				if not os.path.isfile(fp):
					logging.warning(f"Could not find {fp}")
					return
				file_widget.accept_file(fp)
			_markers = []
			for a0, a1, b0, b1, d, corr in sync["data"]:
				_markers.append(markers.LagSample(self, (a0, a1), (b0, b1), d, corr))
			self.props.undo_stack.push(AddAction(_markers))

	def save_traces(self):
		"""Save project with all required settings"""
		sync = {}
		sync["ref"], sync["src"] = self.filenames
		sync["smoothing"] = self.parent.props.alignment_widget.smoothing
		sync["fft_size"] = self.parent.props.display_widget.fft_size
		sync["fft_overlap"] = self.parent.props.display_widget.fft_overlap
		sync["data"] = list(sorted(set((lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.d, lag.corr) for lag in self.lag_samples)))
		cfg_path = os.path.splitext(self.filenames[0])[0]+EXT
		save_config_json(cfg_path, sync)

	def improve_lag(self):
		deltas = []
		selected = [lag for lag in self.lag_samples if lag.selected]
		for lag in selected:
			try:
				# prepare some values
				sr = self.sr
				raw_lag = int(lag.d * sr)
				ref_t0 = int(sr * lag.a[0])
				ref_t1 = int(sr * lag.b[0])
				src_t0 = ref_t0 - raw_lag
				src_t1 = ref_t1 - raw_lag
				freqs = sorted((lag.a[1], lag.b[1]))
				lower = max(freqs[0], 1)
				upper = min(freqs[1], sr // 2 - 1)
				ref_pad_l = 0
				ref_pad_r = 0
				src_pad_l = 0
				src_pad_r = 0
				# channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]

				# trim and pad both sources
				ref_sig = self.signals[0]
				src_sig = self.signals[1]
				if ref_t0 < 0:
					ref_pad_l = abs(ref_t0)
					ref_t0 = 0
				if ref_t1 > len(ref_sig):
					ref_pad_r = ref_t1 - len(ref_sig)

				if src_t0 < 0:
					src_pad_l = abs(src_t0)
					src_t0 = 0
				if src_t1 > len(src_sig):
					src_pad_r = src_t1 - len(src_sig)

				ref_sig = np.pad(ref_sig[ref_t0:ref_t1, 0], (ref_pad_l, ref_pad_r), "constant", constant_values=0)
				src_sig = np.pad(src_sig[src_t0:src_t1, 0], (src_pad_l, src_pad_r), "constant", constant_values=0)

				# correlate both sources
				res = xcorr(
					filters.butter_bandpass_filter(ref_sig, lower, upper, sr, order=3),
					filters.butter_bandpass_filter(src_sig, lower, upper, sr, order=3), mode="same")

				# interpolate to get the most accurate fit
				# we are not necessarily interested in the largest positive value if the correlation is negative
				if self.parent.props.alignment_widget.ignore_phase:
					print("absolute")
					np.abs(res, out=res)

				# get the index of the strongest correlation
				max_index = np.argmax(res)
				# set it to be able to display it
				lag.corr = res[max_index]
				# refine the index with interpolation
				i_peak = wow_detection.parabolic(res, max_index)[0]
				result = raw_lag + i_peak - len(ref_sig) // 2
				# update the lag marker
				delta = (result / sr) - lag.d
				deltas.append(delta)
				print(f"raw accuracy (smp) {raw_lag}")
				print(f"extra accuracy (smp) {result}")
			except:
				logging.exception(f"Refining failed")
		self.props.undo_stack.push(DeltaAction(selected, deltas))

	def select_all(self):
		for trace in self.lag_samples:
			trace.select()

	def deselect_all(self):
		for trace in self.lag_samples:
			trace.deselect()

	def delete_traces(self, delete_all=False):
		deltraces = []
		for trace in reversed(self.lag_samples):
			if (trace.selected and not delete_all) or delete_all:
				deltraces.append(trace)
		self.props.undo_stack.push(DeleteAction(deltraces))

	def run_resample(self):
		self.resample_files((self.filenames[1],))

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files(filenames)

	def resample_files(self, files):
		channels = self.parent.props.resampling_widget.channels
		if self.filenames[1] and self.lag_samples and channels:
			lag_curve = self.lag_line.data
			self.resampling_thread.settings = {
				"filenames"			: files,
				"lag_curve"			: lag_curve,
				"resampling_mode"	: self.parent.props.resampling_widget.mode,
				"sinc_quality"		: self.parent.props.resampling_widget.sinc_quality,
				"use_channels"		: channels}
			self.resampling_thread.start()
		
	def on_mouse_press(self, event):
		# selection
		if event.button == 2:
			closest_lag_sample = self.get_closest(self.lag_samples, event.pos)
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				v = "None" if closest_lag_sample.corr is None else f"{closest_lag_sample.corr:.3f}"
				self.parent.props.alignment_widget.corr_l.setText(v)
				event.handled = True
			# update the last spectrum with pan
			click = self.click_spec_conversion(event.pos)
			if click is not None:
				# sample the lag curve at the click's time
				lag = self.lag_line.sample_at(click[0])
				old_d = self.spectra[-1].delta
				# move the source spectrum
				self.spectra[-1].translate(lag - old_d)
	
	def on_mouse_release(self, event):
		# coords of the click on the vispy canvas
		if self.filenames[1] and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				# are they in spec_view?
				if a is not None and b is not None:
					if "Control" in event.modifiers:
						d = b[0]-a[0]
						self.spectra[1].translate(d)
					elif "Shift" in event.modifiers:
						marker = markers.LagSample(self, a, b)
						self.props.undo_stack.push(AddAction((marker,)))
					# elif "Alt" in event.modifiers:
						# print()
						# print("Start")
						# #first get the time range for both
						# #apply bandpass
						# #split into pieces and look up the delay for each
						# #correlate all the pieces
						# sr = self.sr
						# dur = int(0.2 *sr)
						# times = sorted((a[0], b[0]))
						# ref_t0 = int(sr*times[0])
						# ref_t1 = int(sr*times[1])
						# # src_t0 = int(sr*lag.a[0]+lag.d)
						# # src_t1 = src_t0+ref_t1-ref_t0
						# freqs = sorted((a[1], b[1]))
						# lower = max(freqs[0], 1)
						# upper = min(freqs[1], sr//2-1)
						# # channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
						# ref_ob = sf.SoundFile(self.props.reffilename)
						# ref_sig = ref_ob.read(always_2d=True, dtype='float32')
						# src_ob = sf.SoundFile(self.props.srcfilename)
						# src_sig = src_ob.read(always_2d=True, dtype='float32')
						# sample_times = np.arange(ref_t0, ref_t1, dur//32)
						# data = self.lag_line.data
						# sample_lags = np.interp(sample_times, data[:, 0]*sr, data[:, 1]*sr)
						
						# #could do a stack
						# out = np.zeros((len(sample_times), 2), dtype=np.float32)
						# out[:, 0] = sample_times/sr
						# for i, (x, d) in enumerate(zip(sample_times, sample_lags)):
						#
						# 	ref_s = filters.butter_bandpass_filter(ref_sig[x:x+dur,0], lower, upper, sr, order=3)
						# 	src_s = filters.butter_bandpass_filter(src_sig[x-int(d):x-int(d)+dur,0], lower, upper, sr, order=3)
						# 	res = np.correlate(ref_s*np.hanning(dur), src_s*np.hanning(dur), mode="same")
						# 	i_peak = np.argmax(res)
						# 	# interpolate the most accurate fit
						# 	result = wow_detection.parabolic(res, i_peak)[0] -(len(ref_s)//2)
						# 	# print("extra accuracy (smp)",int(d)+result)
						# 	out[i, 1] = (int(d)+result)/sr
						# self.lag_line.data =out
						# self.lag_line.line_speed.set_data(pos=out)


if __name__ == '__main__':
	widgets.startup(MainWindow)
