import logging
import os

import numpy as np
from PyQt5 import QtWidgets

# custom modules
from util.correlation import xcorr
from util.undo import AddAction, DeltaAction
from util import spectrum, wow_detection, qt_threads, widgets, filters, io_ops, \
	markers

from util.config import load_json, logging_setup

logging_setup()


class MainWindow(widgets.MainWindow):
	EXT = ".tapesync"
	STORE = {"markers": markers.LagSample}

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

	def improve_lag(self):
		deltas = []
		selected = self.selected_markers
		for lag in selected:
			try:
				# todo - this isn't dealing with different sample rates
				sr = self.sr
				# prepare some values
				t0 = lag.a[0]
				t1 = lag.b[0]
				ref, src = self.spectra
				ref_sig = ref.get_signal(t0, t1)
				src_sig = src.get_signal(t0-lag.d, t1-lag.d)

				freqs = sorted((lag.a[1], lag.b[1]))
				lower = max(freqs[0], 1)
				upper = min(freqs[1], sr // 2 - 1)
				# correlate both sources
				res = xcorr(
					filters.butter_bandpass_filter(ref_sig, lower, upper, sr, order=3),
					filters.butter_bandpass_filter(src_sig, lower, upper, sr, order=3), mode="same")

				# interpolate to get the most accurate fit
				# we are not necessarily interested in the largest positive value if the correlation is negative
				if self.parent.props.alignment_widget.ignore_phase:
					logging.warning("Ignoring phase")
					np.abs(res, out=res)

				# get the index of the strongest correlation
				max_index = np.argmax(res)
				# refine the index with interpolation
				i_peak, lag.corr = wow_detection.parabolic(res, max_index)
				result = i_peak - len(ref_sig) // 2
				# update the lag marker
				delta = result / sr
				deltas.append(delta)
				logging.info(f"Moved by {result} samples")
			except:
				logging.exception(f"Refining failed")
		self.props.undo_stack.push(DeltaAction(selected, deltas))
		for trace in selected:
			self.spectra[-1].set_offset(trace.d)
			self.update_corr_view(trace)

	def run_resample(self):
		self.resample_files((self.filenames[1],))

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
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
				self.spectra[-1].set_offset(self.lag_line.sample_at(click[0]))

	def update_corr_view(self, closest_marker):
		v = "None" if closest_marker.corr is None else f"{closest_marker.corr:.3f}"
		self.parent.props.alignment_widget.corr_l.setText(v)

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
