import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtWidgets

# custom modules
from util import spectrum, resampling, wow_detection, qt_threads, widgets, io_ops, config, markers


class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pyrespeeder", widgets.ParamWidget, Canvas, 1)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		# view_menu = main_menu.addMenu('View')
		# help_menu = main_menu.addMenu('Help')
		button_data = (
			(file_menu, "Open", self.props.file_widget.ask_open, "CTRL+O"),
			(file_menu, "Save", self.canvas.save_traces, "CTRL+S"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B"),
			(file_menu, "Exit", self.close, ""),
			(edit_menu, "Undo", self.canvas.restore_traces, "CTRL+Z"),
			# (edit_menu, "Redo", self.props.foo, "CTRL+Y"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A"),
			(edit_menu, "Invert Selection", self.canvas.invert_selection, "CTRL+I"),
			(edit_menu, "Merge Selected", self.canvas.merge_selected_traces, "CTRL+M"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL"),
			# (edit_menu, "Play/Pause", self.canvas.audio_widget.play_pause, "SPACE"),
			)
		self.add_to_menu(button_data)


class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, spectra_colors=(None,), y_axis='Octaves')
		self.create_native()
		self.native.setParent(parent)

		self.unfreeze()
		self.parent = parent
		self.show_regs = True
		self.show_lines = True
		self.deltraces = []
		self.lines = []
		self.regs = []
		self.master_speed = markers.MasterSpeedLine(self)
		self.master_reg_speed = markers.MasterRegLine(self, (0, 0, 1, .5))

		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_widget.onProgress)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_widget.onProgress)
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.canvas = self
		self.parent.props.alignment_widget.setVisible(False)
		self.freeze()

	def load_visuals(self, ):
		# read any saved traces or regressions
		for offset, times, freqs in io_ops.read_trace(self.filenames[0]):
			markers.TraceLine(self, times, freqs, offset=offset)
		for t0, t1, amplitude, omega, phase, offset in io_ops.read_regs(self.filenames[0]):
			markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)
		self.master_speed.update()
		self.master_reg_speed.update()

	def save_traces(self):
		# get the data from the traces and regressions and save it
		io_ops.write_trace(self.filenames[0], [(line.offset, line.times, line.freqs) for line in self.lines])
		io_ops.write_regs(
			self.filenames[0],
			[(reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) for reg in self.regs])

	def delete_traces(self, not_only_selected=False):
		self.deltraces = []
		for trace in reversed(self.regs + self.lines):
			if (trace.selected and not not_only_selected) or not_only_selected:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.master_speed.update()
		self.master_reg_speed.update()
		# this means a file was loaded, so clear the undo stack
		if not_only_selected:
			self.deltraces = []

	def merge_selected_traces(self):
		self.deltraces = []
		t0 = 999999
		t1 = 0
		means = []
		offsets = []
		for trace in reversed(self.lines):
			if trace.selected:
				self.deltraces.append(trace)
				t0 = min(t0, trace.speed_data[0, 0])
				t1 = max(t1, trace.speed_data[-1, 0])
				means.append(trace.spec_center[1])
				offsets.append(trace.offset)
		if self.deltraces:
			for trace in self.deltraces:
				trace.remove()
			sr = self.sr
			hop = self.hop
			i0 = int(t0 * sr / hop)
			i1 = int(t1 * sr / hop)
			data = self.master_speed.data[i0:i1]
			freqs = np.power(2, data[:, 1] + np.log2(np.mean(means)))
			line = markers.TraceLine(self, data[:, 0], freqs, np.mean(offsets))
			self.master_speed.update()

	def restore_traces(self):
		for trace in self.deltraces:
			trace.initialize()
		self.master_speed.update()
		self.master_reg_speed.update()
		self.deltraces = []

	def run_resample(self):
		if self.filenames[0] and self.lines + self.regs:
			channels = self.props.resampling_widget.channels
			if channels:
				self.resampling_thread.settings = {
					"filenames": (self.filenames[0],),
					"signal_data": ((self.signals[0], self.sr),),
					"speed_curve": self.get_speed_curve(),
					"resampling_mode": self.props.resampling_widget.mode,
					"sinc_quality": self.props.resampling_widget.sinc_quality,
					"use_channels": channels}
				self.resampling_thread.start()

	def get_speed_curve(self):
		if self.regs:
			speed_curve = self.master_reg_speed.get_linspace()
			print("Using regressed speed")
		else:
			speed_curve = self.master_speed.get_linspace()
			print("Using measured speed")
		return speed_curve

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files(filenames)

	def resample_files(self, files):
		channels = self.props.resampling_widget.channels
		if self.filenames[1] and self.lag_samples and channels:
			self.resampling_thread.settings = {
				"filenames"			: files,
				"speed_curve"		: self.get_speed_curve(),
				"resampling_mode"	: self.props.resampling_widget.mode,
				"sinc_quality"		: self.props.resampling_widget.sinc_quality,
				"use_channels"		: channels}
			self.resampling_thread.start()

	def select_all(self):
		for trace in self.lines + self.regs:
			trace.select()

	def invert_selection(self):
		for trace in self.lines + self.regs:
			trace.toggle()

	def on_mouse_press(self, event):
		# #audio cursor
		# b = self.click_spec_conversion(event.pos)
		# #are they in spec_view?
		# if b is not None:
		# self.props.audio_widget.cursor(b[0])
		# selection, single or multi
		if event.button == 2:
			closest_line = self.get_closest_line(event.pos)
			if closest_line:
				closest_line.select_handle("Shift" in event.modifiers)
				event.handled = True

	def on_mouse_release(self, event):
		# coords of the click on the vispy canvas
		if self.filenames[0] and (event.trail() is not None) and event.button == 1 and "Control" in event.modifiers:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				# are they in spec_view?
				if a is not None and b is not None:
					t0, t1 = sorted((a[0], b[0]))
					# f0, f1 = sorted((a[1], b[1]))
					t0 = max(0, t0)
					mode = self.props.tracing_widget.mode
					adapt = self.props.tracing_widget.adapt
					tolerance = self.props.tracing_widget.tolerance
					rpm = self.props.tracing_widget.rpm
					auto_align = self.props.tracing_widget.auto_align
					# maybe query it here from the button instead of the other way
					if mode == "Sine Regression":
						amplitude, omega, phase, offset = wow_detection.trace_sine_reg(
							self.master_speed.get_linspace(), t0, t1, rpm)
						if amplitude == 0:
							print("fallback")
							amplitude, omega, phase, offset = wow_detection.trace_sine_reg(
								self.master_reg_speed.get_linspace(), t0, t1, rpm)
						markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)
						self.master_reg_speed.update()
					else:
						track = wow_detection.Track(
							mode, self.fft_storage[self.keys[0]], t0, t1, self.fft_size,
							self.hop, self.sr, tolerance, adapt,
							trail=[self.click_spec_conversion(click) for click in event.trail()])
						# for times, freqs in res:
						# if len(freqs) and np.nan not in freqs:
						markers.TraceLine(self, track.times, track.freqs, auto_align=auto_align)
						self.master_speed.update()
					return

				# or in speed view?
				# then we are only interested in the Y difference, so we can move the selected speed trace up or down
				a = self.click_speed_conversion(last_click)
				b = self.click_speed_conversion(click)
				if a is not None and b is not None:
					for trace in self.lines + self.regs:
						if trace.selected:
							trace.set_offset(a[1], b[1])
					self.master_speed.update()
					self.master_reg_speed.update()

	def get_closest_line(self, click):
		if click is not None:
			if self.show_regs and self.show_lines:
				return self.get_closest(self.lines + self.regs, click)
			elif self.show_regs and not self.show_lines:
				return self.get_closest(self.regs, click)
			elif not self.show_regs and self.show_lines:
				return self.get_closest(self.lines, click)


if __name__ == '__main__':
	widgets.startup(MainWindow)
