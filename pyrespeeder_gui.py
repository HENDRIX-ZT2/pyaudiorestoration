import numpy as np
import logging
from PyQt5 import QtWidgets

# custom modules
from util import spectrum, wow_detection, qt_threads, widgets, io_ops, markers

from util.undo import AddAction, DeleteAction, MoveAction, MergeAction
from util.config import logging_setup

logging_setup()


class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pyrespeeder", widgets.ParamWidget, Canvas, 1)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		# view_menu = main_menu.addMenu('View')
		# help_menu = main_menu.addMenu('Help')
		button_data = (
			(file_menu, "Open", self.props.files_widget.ask_open, "CTRL+O", "dir"),
			(file_menu, "Save", self.canvas.save_traces, "CTRL+S", "save"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R", "curve"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B", "curve2"),
			(file_menu, "Exit", self.close, "", "exit"),
			(edit_menu, "Undo", self.props.undo_stack.undo, "CTRL+Z", "undo"),
			(edit_menu, "Redo", self.props.undo_stack.redo, "CTRL+Y", "redo"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A", "select_extend"),
			(edit_menu, "Invert Selection", self.canvas.invert_selection, "CTRL+I", "select_intersect"),
			(edit_menu, "Merge Selected", self.canvas.merge_selected_traces, "CTRL+M"),
			(edit_menu, "Merge Overlapping", self.canvas.group_traces, "CTRL+G"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL", "x"),
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
		self.grouped_traces = []
		self.master_speed = markers.MasterSpeedLine(self)
		self.master_reg_speed = markers.MasterRegLine(self, (0, 0, 1, .5))

		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.canvas = self
		self.parent.props.alignment_widget.setVisible(False)
		self.freeze()

	@property
	def lines(self):
		return [m for m in self.markers if isinstance(m, markers.TraceLine)]

	@property
	def regs(self):
		return [m for m in self.markers if isinstance(m, markers.RegLine)]

	def load_visuals(self, ):
		# read any saved traces or regressions
		for offset, times, freqs in io_ops.read_trace(self.filenames[0]):
			yield markers.TraceLine(self, times, freqs, offset=offset)
		# self.master_speed.update()
		for t0, t1, amplitude, omega, phase, offset in io_ops.read_regs(self.filenames[0]):
			yield markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)

	def save_traces(self):
		# get the data from the traces and regressions and save it
		io_ops.write_trace(self.filenames[0], [(line.offset, line.times, line.freqs) for line in self.lines])
		io_ops.write_regs(
			self.filenames[0],
			[(reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) for reg in self.regs])
		self.parent.props.undo_stack.setClean()

	def merge_selected_traces(self):
		self.merge_traces(list(reversed(self.selected_markers)))

	def merge_traces(self, traces_to_merge):
		if traces_to_merge and len(traces_to_merge) > 1:
			logging.info(f"Merging {len(traces_to_merge)} lines")
			t0 = 999999
			t1 = 0
			means = []
			offsets = []
			for trace in traces_to_merge:
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
			self.props.undo_stack.push(MergeAction((line,), traces_to_merge))

	def group_traces(self):
		"""Convert overlapping traces into grouped representations"""
		groups = self.master_speed.get_overlapping_lines()
		logging.info(f"Merging {len(groups)} groups")
		for group in groups:
			self.merge_traces(group)

	def run_resample(self):
		spec = self.spectra[0]
		if spec.filename and self.lines + self.regs:
			channels = self.props.files_widget.files[0].channel_widget.channels
			if channels:
				self.resampling_thread.settings = {
					"filenames": (spec.filename,),
					"signal_data": ((spec.signal, spec.sr),),
					"speed_curve": self.get_speed_curve(),
					"use_channels": channels}
				self.props.resampling_widget.to_cfg(self.resampling_thread.settings)
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
		channels = self.props.files_widget.files[0].channel_widget.channels
		if self.lines + self.regs and channels:
			self.resampling_thread.settings = {
				"filenames"			: files,
				"speed_curve"		: self.get_speed_curve(),
				"use_channels"		: channels}
			self.props.resampling_widget.to_cfg(self.resampling_thread.settings)
			self.resampling_thread.start()

	def update_lines(self):
		self.master_speed.update()
		self.master_reg_speed.update()

	def on_mouse_press(self, event):
		# #audio cursor
		# b = self.px_to_spectrum(event.pos)
		# #are they in spec_view?
		# if b is not None:
		# self.props.audio_widget.cursor(b[0])
		# selection, single or multi
		if event.button == 2:
			closest_marker = self.get_closest(self.markers, event.pos)
			if closest_marker:
				closest_marker.select_handle("Shift" in event.modifiers)
				event.handled = True

	def on_mouse_release(self, event):
		if self.filenames[0] and (event.trail() is not None) and event.button == 1 and "Control" in event.modifiers:
			# event.trail() is integer pixel coordinates on vispy canvas
			trail = [self.px_to_spectrum(click) for click in event.trail()]
			# filter out any clicks outside of spectrum, for which px_to_spectrum returns None
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
					marker = markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)
					self.props.undo_stack.push(AddAction((marker,)))
				else:
					self.track_wow(settings, trail)
				return

			# or in speed view?
			trail = [self.px_to_speed(click) for click in event.trail()]
			trail = [x for x in trail if x is not None]
			if trail:
				# only interested in the Y difference, so we can move the selected speed trace up or down
				a = trail[0]
				b = trail[-1]
				self.props.undo_stack.push(MoveAction(self.selected_markers, a[1], b[1]))

	def track_wow(self, settings, trail):
		spec = self.spectra[0]
		track = wow_detection.Track(
			settings.mode, self.fft_storage[spec.key], trail, self.fft_size,
			self.hop, spec.sr, settings.tolerance, settings.adapt)
		marker = markers.TraceLine(self, track.times, track.freqs, auto_align=settings.auto_align)
		self.props.undo_stack.push(AddAction((marker,)))


if __name__ == '__main__':
	widgets.startup(MainWindow)
