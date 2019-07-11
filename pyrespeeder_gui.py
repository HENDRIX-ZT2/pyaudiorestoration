import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtWidgets

#custom modules
from util import spectrum, resampling, wow_detection, qt_threads, snd, widgets, io_ops, config, markers
		
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.filename = ""
		self.deltraces = []
		
		self.display_widget = widgets.DisplayWidget(self.parent.canvas)
		self.tracing_widget = widgets.TracingWidget(self.parent.canvas)
		self.resampling_widget = widgets.ResamplingWidget()
		self.progress_widget = widgets.ProgressWidget()
		self.audio_widget = snd.AudioWidget()
		self.inspector_widget = widgets.InspectorWidget()
		
		buttons = [self.display_widget, self.tracing_widget, self.resampling_widget, self.progress_widget, self.audio_widget, self.inspector_widget ]
		widgets.vbox2(self, buttons)

		self.resampling_thread = qt_threads.ResamplingThread(self.progress_widget.onProgress)
		self.parent.canvas.fourier_thread.notifyProgress.connect( self.progress_widget.onProgress )
		
	def open_audio(self):
		#just a wrapper around load_audio so we can access that via drag & drop and button
		#pyqt5 returns a tuple
		filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', self.parent.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
		self.load_audio(filename)
			
	def load_audio(self, filename):
		#called whenever a potential audio file is given via drag&drop or open_audio
	
		#ask the user if it should really be opened, if another file was already open
		if widgets.abort_open_new_file(self, filename, self.filename):
			return
		
		try:
			self.parent.canvas.compute_spectra( (filename,), self.display_widget.fft_size, self.display_widget.fft_overlap)
		# file could not be opened
		except RuntimeError as err:
			print(err)
		# no issues, we can continue
		else:
			self.filename = filename
			
			#Cleanup of old data
			self.delete_traces(not_only_selected=True)
			self.resampling_widget.refill(self.parent.canvas.channels)
			
			#read any saved traces or regressions
			for offset, times, freqs in io_ops.read_trace(self.filename):
				markers.TraceLine(self.parent.canvas, times, freqs, offset=offset)
			for t0, t1, amplitude, omega, phase, offset in io_ops.read_regs(self.filename):
				markers.RegLine(self.parent.canvas, t0, t1, amplitude, omega, phase, offset)
			self.parent.canvas.master_speed.update()
			self.parent.canvas.master_reg_speed.update()
			self.parent.update_file(self.filename)
				
	def save_traces(self):
		#get the data from the traces and regressions and save it
		io_ops.write_trace(self.filename, [ (line.offset, line.times, line.freqs) for line in self.parent.canvas.lines ] )
		io_ops.write_regs(self.filename, [ (reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) for reg in self.parent.canvas.regs ] )
			
	def delete_traces(self, not_only_selected=False):
		self.deltraces= []
		for trace in reversed(self.parent.canvas.regs+self.parent.canvas.lines):
			if (trace.selected and not not_only_selected) or not_only_selected:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.parent.canvas.master_speed.update()
		self.parent.canvas.master_reg_speed.update()
		#this means a file was loaded, so clear the undo stack
		if not_only_selected:
			self.deltraces= []
		
	def merge_selected_traces(self):
		self.deltraces= []
		t0 = 999999
		t1 = 0
		means = []
		offsets = []
		for trace in reversed(self.parent.canvas.lines):
			if trace.selected:
				self.deltraces.append(trace)
				t0 = min(t0, trace.speed_data[0, 0])
				t1 = max(t1, trace.speed_data[-1, 0])
				means.append(trace.spec_center[1])
				offsets.append(trace.offset)
		if self.deltraces:
			for trace in self.deltraces:
				trace.remove()
			sr = self.parent.canvas.sr
			hop = self.parent.canvas.hop
			i0 = int(t0*sr/hop)
			i1 = int(t1*sr/hop)
			data = self.parent.canvas.master_speed.data[i0:i1]
			freqs = np.power(2, data[:,1]+np.log2(np.mean(means)))
			line = TraceLine(self.parent.canvas, data[:,0], freqs, np.mean(offsets))
			self.parent.canvas.master_speed.update()
			
	def restore_traces(self):
		for trace in self.deltraces:
			trace.initialize()
		self.parent.canvas.master_speed.update()
		self.parent.canvas.master_reg_speed.update()
		self.deltraces = []
			
	def run_resample(self):
		if self.filename and self.parent.canvas.lines:
			channels = self.resampling_widget.channels
			if channels:
				if self.parent.canvas.regs:
					speed_curve = self.parent.canvas.master_reg_speed.get_linspace()
					print("Using regressed speed")
				else:
					speed_curve = self.parent.canvas.master_speed.get_linspace()
					print("Using measured speed")
				self.resampling_thread.settings = {"filenames"			:(self.filename,),
													"speed_curve"		:speed_curve, 
													"resampling_mode"	:self.resampling_widget.mode,
													"sinc_quality"		:self.resampling_widget.sinc_quality,
													"use_channels"		:channels}
				self.resampling_thread.start()
			
	def select_all(self):
		for trace in self.parent.canvas.lines+self.parent.canvas.regs:
			trace.select()
		
	def invert_selection(self):
		for trace in self.parent.canvas.lines+self.parent.canvas.regs:
			trace.toggle()

class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pyrespeeder", ObjectWidget, Canvas)
		mainMenu = self.menuBar() 
		fileMenu = mainMenu.addMenu('File')
		editMenu = mainMenu.addMenu('Edit')
		#viewMenu = mainMenu.addMenu('View')
		#helpMenu = mainMenu.addMenu('Help')
		button_data = ( (fileMenu, "Open", self.props.open_audio, "CTRL+O"), \
						(fileMenu, "Save", self.props.save_traces, "CTRL+S"), \
						(fileMenu, "Resample", self.props.run_resample, "CTRL+R"), \
						(fileMenu, "Exit", self.close, ""), \
						(editMenu, "Undo", self.props.restore_traces, "CTRL+Z"), \
						# (editMenu, "Redo", self.props.foo, "CTRL+Y"), \
						(editMenu, "Select All", self.props.select_all, "CTRL+A"), \
						(editMenu, "Invert Selection", self.props.invert_selection, "CTRL+I"), \
						(editMenu, "Merge Selected", self.props.merge_selected_traces, "CTRL+M"), \
						(editMenu, "Delete Selected", self.props.delete_traces, "DEL"), \
						(editMenu, "Play/Pause", self.props.audio_widget.play_pause, "SPACE"), \
						)
		self.add_to_menu(button_data)

class Canvas(spectrum.SpectrumCanvas):

	def __init__(self):
		spectrum.SpectrumCanvas.__init__(self, spectra_colors=(None,), y_axis='Octaves' )
		self.unfreeze()
		self.show_regs = True
		self.show_lines = True
		self.lines = []
		self.regs = []
		self.master_speed = markers.MasterSpeedLine(self)
		self.master_reg_speed = markers.MasterRegLine(self, (0, 0, 1, .5))
		self.freeze()
		
	def on_mouse_press(self, event):
		#audio cursor
		b = self.click_spec_conversion(event.pos)
		#are they in spec_view?
		if b is not None:
			self.props.audio_widget.cursor(b[0])
		#selection, single or multi
		if event.button == 2:
			closest_line = self.get_closest_line( event.pos )
			if closest_line:
				if "Shift" in event.modifiers:
					closest_line.select_handle(multi=True)
					event.handled = True
				else:
					closest_line.select_handle()
					event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		if self.filenames and (event.trail() is not None) and event.button == 1 and "Control" in event.modifiers:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				#are they in spec_view?
				if a is not None and b is not None:
					t0, t1 = sorted((a[0], b[0]))
					f0, f1 = sorted((a[1], b[1]))
					t0 = max(0, t0)
					mode = self.props.tracing_widget.mode
					adapt = self.props.tracing_widget.adapt
					tolerance = self.props.tracing_widget.tolerance
					rpm = self.props.tracing_widget.rpm
					auto_align = self.props.tracing_widget.auto_align
					#maybe query it here from the button instead of the other way
					if mode == "Sine Regression":
						amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_speed.get_linspace(), t0, t1, rpm)
						if amplitude == 0:
							print("fallback")
							amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_reg_speed.get_linspace(), t0, t1, rpm)
						markers.RegLine(self, t0, t1, amplitude, omega, phase, offset)
						self.master_reg_speed.update()
					else:
						track = wow_detection.Track(mode, self.fft_storage[self.keys[0]], t0, t1, self.fft_size, self.hop, self.sr, tolerance, adapt, trail = [self.click_spec_conversion(click) for click in event.trail()])
						# for times, freqs in res:
							# if len(freqs) and np.nan not in freqs:
						markers.TraceLine(self, track.times, track.freqs, auto_align=auto_align)
						self.master_speed.update()
					return
				
				#or in speed view?
				#then we are only interested in the Y difference, so we can move the selected speed trace up or down
				a = self.click_speed_conversion(last_click)
				b = self.click_speed_conversion(click)
				if a is not None and b is not None:
					for trace in self.lines+self.regs:
						if trace.selected:
							trace.set_offset(a[1], b[1])
					self.master_speed.update()
					self.master_reg_speed.update()

	def get_closest_line(self, click):
		if click is not None:
			if self.show_regs and self.show_lines:
				return self.get_closest(self.lines+self.regs, click)
			elif self.show_regs and not self.show_lines:
				return self.get_closest(self.regs, click)
			elif not self.show_regs and self.show_lines:
				return self.get_closest(self.lines, click)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
	widgets.startup( MainWindow )
