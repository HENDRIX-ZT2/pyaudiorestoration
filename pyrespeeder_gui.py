import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtCore, QtWidgets
from scipy.signal import butter, filtfilt

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, qt_theme, snd, widgets

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""Performs a low, high or bandpass filter if low & highcut are in range"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	low_in_range = 0 < low < 1
	high_in_range = 0 < high < 1
	if low_in_range and high_in_range:
		b, a = butter(order, [low, high], btype='band')
	elif low_in_range and not high_in_range:
		b, a = butter(order, low, btype='high')
	elif not low_in_range and high_in_range:
		b, a = butter(order, high, btype='low')
	else:
		return data
	return filtfilt(b, a, data)
	
class ResamplingThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	def run(self):
		name, speed_curve, resampling_mode, sinc_quality, use_channels = self.settings
		resampling.run(name, speed_curve= speed_curve, resampling_mode = resampling_mode, sinc_quality=sinc_quality, use_channels=use_channels, prog_sig=self)
			
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
		self.audio_widget = snd.AudioWidget()
		self.inspector_widget = widgets.InspectorWidget()
		
		buttons = [self.display_widget, self.tracing_widget, self.resampling_widget, self.audio_widget, self.inspector_widget ]

		vbox = QtWidgets.QVBoxLayout()
		for w in buttons: vbox.addWidget(w)
		vbox.addStretch(1.0)
		self.setLayout(vbox)

		self.resampling_thread = ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.resampling_widget.onProgress)
		
	def open_audio(self):
		#just a wrapper around load_audio so we can access that via drag & drop and button
		#pyqt5 returns a tuple
		filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Audio files (*.flac *.wav)")[0]
		self.load_audio(filename)
			
	def load_audio(self, filename):
		#called whenever a potential audio file is set as self.filename - via drag& drop or open_audio
		if filename:
			if filename != self.filename:
				#ask the user if it should really be opened, if another file was already open
				if self.filename:
					qm = QtWidgets.QMessageBox
					ret = qm.question(self,'', "Do you really want to load "+os.path.basename(filename)+"? You will lose unsaved work on "+os.path.basename(self.filename)+"!", qm.Yes | qm.No)
					if ret == qm.No:
						return
				
				# is the (dropped) file an audio file, ie. can it be read by pysoundfile?
				try:
					soundob = sf.SoundFile(filename)
					self.filename = filename
				except:
					print(filename+" could not be read, is it a valid audio file?")
					return
				
				#Cleanup of old data
				self.parent.canvas.init_fft_storages()
				self.delete_traces(not_only_selected=True)
				self.resampling_widget.refill(soundob.channels)
				
				#finally - proceed with spectrum stuff elsewhere
				self.parent.setWindowTitle('pyrespeeder '+os.path.basename(self.filename))

				self.parent.canvas.set_file_or_fft_settings((filename,),
													 fft_size = self.display_widget.fft_size,
													 fft_overlap = self.display_widget.fft_overlap)
				# also force a cmap update here
				self.display_widget.update_cmap()
				
				#read any saved traces or regressions
				data = resampling.read_trace(self.filename)
				for offset, times, freqs in data:
					TraceLine(self.parent.canvas, times, freqs, offset=offset)
				self.parent.canvas.master_speed.update()
				data = resampling.read_regs(self.filename)
				for t0, t1, amplitude, omega, phase, offset in data:
					RegLine(self.parent.canvas, t0, t1, amplitude, omega, phase, offset)
				self.parent.canvas.master_reg_speed.update()
				
	def save_traces(self):
		#get the data from the traces and regressions and save it
		resampling.write_trace(self.filename, [ (line.offset, line.times, line.freqs) for line in self.parent.canvas.lines ] )
		resampling.write_regs(self.filename, [ (reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) for reg in self.parent.canvas.regs ] )
			
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
				self.resampling_thread.settings = ((self.filename,), speed_curve, self.resampling_widget.mode, self.resampling_widget.sinc_quality, channels)
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

class BaseMarker:
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, container, color_def, color_sel):
		self.selected = False
		self.vispy_canvas = vispy_canvas
		self.visuals = []
		self.color_def = color_def
		self.color_sel = color_sel
		self.spec_center = (0,0)
		self.speed_center = (0,0)
		self.offset = None
		self.container = container
		self.parents = (self.vispy_canvas.speed_view.scene, self.vispy_canvas.spec_view.scene)
		
	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.show()
		self.container.append(self)
		
	def show(self):
		for v, p in zip(self.visuals, self.parents): v.parent = p
		
	def hide(self):
		for v, p in zip(self.visuals, self.parents): v.parent = None
		self.deselect()
		
	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		for v in self.visuals: v.set_data(color = self.color_def)
		
	def select(self):
		"""Select this line"""
		self.selected = True
		for v in self.visuals: v.set_data(color = self.color_sel)
		
	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()
		
		# TODO: evaluate performance penalty of looping here!
		
		# go over all selected markers
		target_freqs = [ marker.spec_center[1]/(2**marker.offset) for marker in self.container if marker.selected]
		# to set the offset properly, we need to know
		# we have a current mean Hz + offset
		# and a target mean Hz
		
		# offset works in log2 scale
		# offset +1 halfes the final frequency
		# offset -1 doubles the final frequency
		self.vispy_canvas.props.tracing_widget.target_s.setValue(np.mean(target_freqs))
		
	def select_handle(self, multi=False):
		if not multi:
			for trace in self.vispy_canvas.regs+self.vispy_canvas.lines:
				trace.deselect()
		self.toggle()
		
	def remove(self):
		for v in self.visuals: v.parent = None
		#note: this has to search the list
		self.container.remove(self)
	
class MasterSpeedLine:
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		
		self.vispy_canvas = vispy_canvas
		
		#create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.data[:, 1] = (0, 0)
		self.bands = (0, 9999999)
		self.line_speed = scene.Line(pos=self.data, color=(1, 0, 0, .5), method='gl')
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		
	def update(self):
		#set the output data
		num = self.vispy_canvas.num_ffts
		#get the times at which the average should be sampled
		times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
		self.data = np.zeros((len(times), 2), dtype=np.float32)
		self.data[:, 0] = times
		if self.vispy_canvas.lines:
			#create the array for sampling
			out = np.zeros((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
			#lerp and sample all lines, use NAN for missing parts
			for i, line in enumerate(self.vispy_canvas.lines):
				line_sampled = np.interp(times, line.times, line.speed, left = np.nan, right = np.nan)
				out[:, i] = line_sampled
			#take the mean and ignore nans
			mean_with_nans = np.nanmean(out, axis=1)
			#lerp over nan areas
			nans, x = wow_detection.nan_helper(mean_with_nans)
			mean_with_nans[nans]= np.interp(x(nans), x(~nans), mean_with_nans[~nans])
			
			#bandpass filter the output
			fs = self.vispy_canvas.sr / self.vispy_canvas.hop
			lowcut, highcut = sorted(self.bands)
			self.data[:, 1] = butter_bandpass_filter(mean_with_nans, lowcut, highcut, fs, order=3)
		self.line_speed.set_data(pos=self.data)

	def get_linspace(self):
		"""Convert the log2 spaced speed curve back into linspace for direct use in resampling"""
		out = np.array(self.data)
		np.power(2, out[:,1], out[:,1])
		return out

class MasterRegLine:
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		
		self.vispy_canvas = vispy_canvas
		
		#create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.data[:, 1] = (0, 0)
		self.line_speed = scene.Line(pos=self.data, color=(0, 0, 1, .5), method='gl')
		self.line_speed.parent = vispy_canvas.speed_view.scene
		
	def update(self):
		if self.vispy_canvas.regs:
			
			#here we want to interpolate smoothly between the regressed sines around their centers
			#https://stackoverflow.com/questions/11199509/sine-wave-that-slowly-ramps-up-frequency-from-f1-to-f2-for-a-given-time
			#https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time
			
			#get the times at which the average should be sampled
			num = self.vispy_canvas.num_ffts
			times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
			
			#sort the regressions by their time
			self.vispy_canvas.regs.sort(key=lambda tup: tup.t_center)
			
			pi2 = 2*np.pi
			t_centers = []
			amp_centers = []
			phi_centers =[]
			for i, reg in enumerate(self.vispy_canvas.regs):
				if i == 0:
					phi_centers.append(reg.omega * times[0] + reg.phase % pi2 + reg.offset*pi2)
					t_centers.append(times[0])
					amp_centers.append(reg.amplitude)
				phi_centers.append(reg.omega * reg.t_center + reg.phase % pi2 + reg.offset*pi2)
				t_centers.append(reg.t_center)
				amp_centers.append(reg.amplitude)
				if i == len(self.vispy_canvas.regs)-1:
					phi_centers.append(reg.omega * times[-1] + reg.phase % pi2 + reg.offset*pi2)
					t_centers.append(times[-1])
					amp_centers.append(reg.amplitude)
			sine_curve = np.sin( np.interp(times, t_centers, phi_centers))
			amplitudes_sampled = np.interp(times, t_centers, amp_centers)
			
			#create the speed curve visualization, boost it a bit to distinguish from the raw curves
			self.data = np.zeros((len(times), 2), dtype=np.float32)
			self.data[:, 0] = times
			self.data[:, 1] = 1.5  * amplitudes_sampled *  sine_curve
		else:
			self.data = np.zeros((2, 2), dtype=np.float32)
			self.data[:, 0] = (0, 999)
			self.data[:, 1] = (0, 0)
		self.line_speed.set_data(pos=self.data)

	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		
	def get_linspace(self):
		"""Convert the log2 spaced speed curve back into linspace for further processing"""
		out = np.array(self.data)
		np.power(2, out[:,1], out[:,1])
		return out
		
class RegLine(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""
	def __init__(self, vispy_canvas, t0, t1, amplitude, omega, phase, offset):
	
		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.regs, color_def, color_sel)
		#the extents on which this regression operated
		self.t0 = t0
		self.t1 = t1
		#here, the reg values are most accurate
		self.t_center = (t0+ t1)/2
		self.speed_center = np.array( (self.t_center, 0) )
		self.spec_center = np.array( (self.t_center, 2000) )
		
		#the following is more or less duped in the tracer - resolve?
		speed_curve = vispy_canvas.master_speed.get_linspace()
		
		times = speed_curve[:,0]
		speeds = speed_curve[:,1]
		
		#which part to process?
		period = times[1]-times[0]
		ind_start = int(self.t0 / period)
		ind_stop = int(self.t1 / period)
		clipped_times = times[ind_start:ind_stop]
		
		#set the properties
		self.amplitude = amplitude
		self.omega = omega
		self.phase = phase
		self.offset = offset
		
		#some conventions are needed
		#correct the amplitude & phase so we can interpolate properly
		if self.amplitude < 0:
			self.amplitude *= -1
			self.phase += np.pi
		#phase should be in 0 < x< 2pi
		#this is not completely guaranteed by this
		# if self.phase < 0:
			# self.phase += (2*np.pi)
		# if self.phase > 2*np.pi:
			# self.phase -= (2*np.pi)
		#self.phase = self.phase % (2*np.pi)
		
		#create the speed curve visualization
		self.speed_data = np.stack( (clipped_times, self.amplitude * np.sin(self.omega * clipped_times + self.phase)), axis=-1)
		#sine_on_hz = np.power(2, sine + np.log2(2000))
		self.visuals.append( scene.Line(pos=self.speed_data, color=(0, 0, 1, .5), method='gl') )
		self.initialize()
		
	def set_offset(self, a, b):
		#user manipulation: custom amplitude for sample
		self.amplitude *= (b/a)
		self.speed_data[:, 1]*= (b/a)
		self.visuals[0].set_data(pos=self.speed_data)
		
	def select(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = True
		self.visuals[0].set_data(color = self.color_sel)
		#set the offset in the ui
		self.vispy_canvas.props.tracing_widget.phase_s.setValue(self.offset)
		
	def update_phase(self, v):
		"""Adjust this regressions's phase offset according to the UI input."""
		if self.selected: self.offset = v
		
class TraceLine(BaseMarker):
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, times, freqs, offset=None, auto_align=False):
		
		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.lines, color_def, color_sel)
		self.times = np.asarray(times)
		self.freqs = np.asarray(freqs)
		
		#note: the final, output speed curve output should be linscale and centered on 1
		self.speed = np.log2(freqs)
		self.speed-= np.mean(self.speed)
		#we don't want to overwrite existing offsets loaded from files
		if offset is None:
			if not auto_align:
				offset = 0
			else:
				#create the array for sampling
				out = np.ones((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
				#lerp and sample all lines, use NAN for missing parts
				for i, line in enumerate(self.vispy_canvas.lines):
					line_sampled = np.interp(times, line.times, line.speed, left = np.nan, right = np.nan)
					out[:, i] = line_sampled
				#take the mean and ignore nans
				mean_with_nans = np.nanmean(out, axis=1)
				offset = np.nanmean(mean_with_nans-self.speed)
				offset = 0 if np.isnan(offset) else offset
		self.offset = offset
		self.speed += offset
		
		#calculate the centers
		mean_times = np.mean(self.times)
		self.spec_center = np.array( (mean_times, np.mean(self.freqs)) )
		self.speed_center = np.array( (mean_times, np.mean(self.speed)) )
		
		#create the speed curve visualization
		self.speed_data = np.stack( (self.times, self.speed), axis=-1)
		self.visuals.append( scene.Line(pos=self.speed_data, color=color_def, method='gl') )
		
		#create the spectral visualization
		#could also do a stack here; note the z coordinate!
		spec_data = np.stack( (self.times, self.freqs, np.ones(len(self.times), dtype=np.float32)*-2), axis=-1)
		self.visuals.append( scene.Line(pos=spec_data, color=color_def, method='gl') )
		#the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform
		
		self.initialize()

	def lock_to(self, f):
		if self.selected:
			# print("lock_to")
			# print(f, np.log2(f))
			# print(self.spec_center[1], np.log2(self.spec_center[1]))
			# print()
			offset = np.log2(self.spec_center[1]) - np.log2(f)
			
			old_offset = self.offset
			self.offset = offset
			self.speed_center[1] += offset - old_offset
			self.speed += offset - old_offset
			self.speed_data[:, 1] = self.speed
			self.visuals[0].set_data(pos = self.speed_data)
		
	def set_offset(self, a, b):
		offset = b-a
		self.offset += offset
		self.speed_center[1] += offset
		self.speed += offset
		self.speed_data[:, 1] = self.speed
		self.visuals[0].set_data(pos = self.speed_data)
	
class Canvas(spectrum.SpectrumCanvas):

	def __init__(self):
		spectrum.SpectrumCanvas.__init__(self, spectra_colors=(None,), y_axis='Octaves',)
		self.unfreeze()
		self.show_regs = True
		self.show_lines = True
		self.lines = []
		self.regs = []
		self.master_speed = MasterSpeedLine(self)
		self.master_reg_speed = MasterRegLine(self)
		self.freeze()
		
	#called if either  the file or FFT settings have changed
	def set_file_or_fft_settings(self, files, fft_size = 256, fft_overlap = 1):
		if files:
			self.compute_spectra(files, fft_size, fft_overlap)
		
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
						RegLine(self, t0, t1, amplitude, omega, phase, offset)
						self.master_reg_speed.update()
					else:
						times, freqs = wow_detection.trace_handle(mode, self.fft_storages[0][(self.fft_size, self.hop, 0)], self.fft_size, self.hop, self.sr, f0, f1, t0, t1,  tolerance, adapt, trail = [self.click_spec_conversion(click) for click in event.trail()])
						if len(freqs) and np.nan not in freqs:
							TraceLine(self, times, freqs, auto_align=auto_align)
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
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	appQt.setPalette(qt_theme.dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = MainWindow()
	win.show()
	appQt.exec_()
