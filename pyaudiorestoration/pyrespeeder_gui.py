import numpy as np
import logging
from PyQt5 import QtWidgets

# custom modules
from util import spectrum, wow_detection, qt_threads, widgets, io_ops, markers


class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pyrespeeder", widgets.ParamWidget, Canvas, 1)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		# view_menu = main_menu.addMenu('View')
		# help_menu = main_menu.addMenu('Help')
		button_data = (
			(file_menu, "Open", self.props.files_widget.ask_open, "CTRL+O"),
			(file_menu, "Save", self.canvas.save_traces, "CTRL+S"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B"),
			(file_menu, "Exit", self.close, ""),
			(edit_menu, "Undo", self.canvas.restore_traces, "CTRL+Z"),
			# (edit_menu, "Redo", self.props.foo, "CTRL+Y"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A"),
			(edit_menu, "Invert Selection", self.canvas.invert_selection, "CTRL+I"),
			(edit_menu, "Merge Selected", self.canvas.merge_selected_traces, "CTRL+M"),
			(edit_menu, "Group", self.canvas.group_traces, "CTRL+G"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL"),
			# (edit_menu, "Play/Pause", self.canvas.audio_widget.play_pause, "SPACE"),
			)
		self.add_to_menu(button_data)


class Action:

	def __init__(self, traces, args):
		pass

	def undo(self):
		pass

	def redo(self):
		pass


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
		self.grouped_traces = []
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

	def delete_traces(self, delete_all=False):
		self.deltraces = []
		for trace in reversed(self.regs + self.lines):
			if (trace.selected and not delete_all) or delete_all:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.master_speed.update()
		self.master_reg_speed.update()
		# this means a file was loaded, so clear the undo stack
		if delete_all:
			self.deltraces = []

	def merge_selected_traces(self):
		self.deltraces = []
		for trace in reversed(self.lines):
			if trace.selected:
				self.deltraces.append(trace)
		self.merge_traces(self.deltraces)
		self.master_speed.update()

	def merge_traces(self, traces_to_merge):
		if traces_to_merge and len(traces_to_merge) > 1:
			logging.info(f"Merging {len(traces_to_merge)} lines")
			t0 = 999999
			t1 = 0
			means = []
			offsets = []
			for trace in traces_to_merge:
				trace.remove()
				t0 = min(t0, trace.speed_data[0, 0])
				t1 = max(t1, trace.speed_data[-1, 0])
				means.append(trace.spec_center[1])
				offsets.append(trace.offset)
			sr = self.sr
			hop = self.hop
			i0 = int(t0 * sr / hop)
			i1 = int(t1 * sr / hop)
			data = self.master_speed.data[i0:i1]
			freqs = np.power(2, data[:, 1] + np.log2(np.mean(means)))
			# todo - taking np.mean(offsets) here is wrong, as they need not be same length, should be weighted mean?
			line = markers.TraceLine(self, data[:, 0], freqs, np.mean(offsets))

	def group_traces(self):
		"""Convert overlapping traces into grouped representations"""
		groups = self.master_speed.get_overlapping_lines()
		logging.info(f"Merging {len(groups)} groups")
		for group in groups:
			self.merge_traces(group)
		self.master_speed.update()

	# def ungroup_traces(self):
	# 	"""group overlapping traces, allow opening the group or collapsing it to display only its evaluated line"""
	# 	pass

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
					"use_channels": channels,
					"suffix": self.props.resampling_widget.suffix}
				self.resampling_thread.start()

	def get_speed_curve(self):
		if self.regs:
			speed_curve = self.master_reg_speed.get_linspace()
			logging.info("Using regressed speed")
		else:
			speed_curve = self.master_speed.get_linspace()
			logging.info("Using measured speed")
		return speed_curve

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files(filenames)

	def resample_files(self, files):
		channels = self.props.resampling_widget.channels
		if self.filenames[0] and self.lines + self.regs and channels:
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
		if self.filenames[0] and (event.trail() is not None) and event.button == 1 and "Control" in event.modifiers:
			# event.trail() is integer pixel coordinates on vispy canvas
			trail = [self.click_spec_conversion(click) for click in event.trail()]
			# filter out any clicks outside of spectrum, for which click_spec_conversion returns None
			trail = [x for x in trail if x is not None]
			if trail:
				t0 = trail[0][0]
				t1 = trail[-1][0]
				settings = self.props.tracing_widget
				# maybe query it here from the button instead of the other way
				if settings.mode == "Sine Regression":
					amplitude, omega, phase, offset = wow_detection.trace_sine_reg(
						self.master_speed.get_linspace(), t0, t1, settings.rpm)
					if amplitude == 0:
						logging.warning("Regressed to no amplitude, trying to sample regression curve")
						amplitude, omega, phase, offset = wow_detection.trace_sine_reg(
							self.master_reg_speed.get_linspace(), t0, t1, settings.rpm)
					markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)
					self.master_reg_speed.update()
				else:
					self.track_wow(settings, trail)
				return

			# or in speed view?
			trail = [self.click_speed_conversion(click) for click in event.trail()]
			trail = [x for x in trail if x is not None]
			if trail:
				# only interested in the Y difference, so we can move the selected speed trace up or down
				a = trail[0]
				b = trail[-1]
				for trace in self.lines + self.regs:
					if trace.selected:
						trace.set_offset(a[1], b[1])
				self.master_speed.update()
				self.master_reg_speed.update()

	def track_wow(self, settings, trail):
		track = wow_detection.Track(
			settings.mode, self.fft_storage[self.keys[0]], trail, self.fft_size,
			self.hop, self.sr, settings.tolerance, settings.adapt)
		markers.TraceLine(self, track.times, track.freqs, auto_align=settings.auto_align)
		self.master_speed.update()

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
