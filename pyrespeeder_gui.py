# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


import sys
import numpy as np
from vispy import scene, gloo, visuals, color, util
import soundfile as sf
from PyQt5 import QtGui, QtCore, QtWidgets

#custom modules
import vispy_ext
import fourier
import resampling
import wow_detection

#log10(x) = log(x) / log(10) = (1 / log(10)) * log(x)
norm_luminance = """
float norm_luminance(vec2 pos) {
	if( pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1 ) {
		return -1.0f;
	}
	vec2 uv = vec2(pos.x, pos.y);
	return (texture2D($texture, uv).r - $vmin)/($vmax - $vmin);
}
"""

simple_cmap = """
vec4 simple_cmap(float x) {
	return vec4(0, 0, x, 1);

}
"""

class SpectrumPiece(scene.Image):
	"""
	The visualization of one part of the whole spectrogram.
	"""
	def __init__(self, texdata, parent):
		#flip the shape, then it is much more straightforward for the rest
		self._shape = (texdata.shape[1], texdata.shape[0])
		self.get_data = visuals.shaders.Function(norm_luminance)
		self.get_data['vmin'] = -80
		self.get_data['vmax'] = -40
		self.get_data['texture'] = gloo.Texture2D(texdata, format='luminance', internalformat='r32f', interpolation="linear")

		scene.Image.__init__(self, method='subdivide', grid=(1000,1), parent=parent)
		
		#maybe to overlay multiple channels? but only makes sense with single colors
		#self.set_gl_state('additive')#, cull_face=False)
		
		#set in the main program
		self.shared_program.frag['get_data'] = self.get_data
		
		#alternative: needs no external color map
		#self.shared_program.frag['color_transform'] = visuals.shaders.Function(simple_cmap)
		#self.shared_program.frag['color_transform'] = visuals.shaders.Function(color.get_colormap(colormap).glsl_map)
		
	def set_size(self, size):
		#not sure if update is needed
		self._shape = size
		self.update()
	
	def set_cmap(self, colormap):
		#update is needed
		self.shared_program.frag['color_transform'] = visuals.shaders.Function(color.get_colormap(colormap).glsl_map)
		self.update()
		
	def set_clims(self, vmin, vmax):
		self.get_data['vmin'] = vmin
		self.get_data['vmax'] = vmax
		
	@property
	def size(self):
		return self._shape

	def _prepare_draw(self, view):
		if self._need_vertex_update:
			self._build_vertex_data()

		if view._need_method_update:
			self._update_method(view)

def to_mel(val):
	### just to set the image size correctly	
	return np.log(val / 700 + 1) * 1127

class TaskThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	settings = "foo"
	def run(self):
		name, speed_curve, resampling_mode, frequency_prec, use_channels, dither = self.settings
		resampling.run(name, speed_curve= speed_curve, resampling_mode = resampling_mode, frequency_prec=frequency_prec, use_channels=use_channels, dither=dither, prog_sig=self)
			
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""
	settings_hard_changed = QtCore.pyqtSignal(name='objectChanged')
	settings_soft_changed = QtCore.pyqtSignal(name='objectChanged2')

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.filename = ""
		
		self.open_b = QtWidgets.QPushButton('Open Audio')
		self.open_b.clicked.connect(self.open_audio)
		
		self.save_b = QtWidgets.QPushButton('Save Traces')
		self.save_b.clicked.connect(self.save_traces)
		
		fft_l = QtWidgets.QLabel("FFT Size")
		self.fft_c = QtWidgets.QComboBox(self)
		self.fft_c.addItems(list(("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768")))
		self.fft_c.currentIndexChanged.connect(self.update_param_hard)
		
		overlap_l = QtWidgets.QLabel("FFT Overlap")
		self.overlap_c = QtWidgets.QComboBox(self)
		self.overlap_c.addItems(list(("1", "2", "4", "8", "16")))
		self.overlap_c.currentIndexChanged.connect(self.update_param_hard)
		
		cmap_l = QtWidgets.QLabel("Colors")
		self.cmap_c = QtWidgets.QComboBox(self)
		self.cmap_c.addItems(sorted(color.colormap.get_colormaps().keys()))
		self.cmap_c.setCurrentText("viridis")
		self.cmap_c.currentIndexChanged.connect(self.update_param_soft)

		self.resample_b = QtWidgets.QPushButton("Resample")
		self.resample_b.clicked.connect(self.run_resample)
		
		mode_l = QtWidgets.QLabel("Resampling Mode")
		self.mode_c = QtWidgets.QComboBox(self)
		self.mode_c.addItems(("Linear", "Expansion", "Sinc", "Blocks"))
		self.mode_c.currentIndexChanged.connect(self.update_resampling_presets)
		
		trace_l = QtWidgets.QLabel("Tracing Mode")
		self.trace_c = QtWidgets.QComboBox(self)
		self.trace_c.addItems(("Center of Gravity","Correlation","Freehand Draw", "Sine Draw", "Sine Regression"))
		self.trace_c.currentIndexChanged.connect(self.update_other_settings)
		
		self.delete_selected_b = QtWidgets.QPushButton('Delete Selected Trace')
		self.delete_selected_b.clicked.connect(self.delete_selected_tracess)
		self.delete_all_b = QtWidgets.QPushButton('Delete All Traces')
		self.delete_all_b.clicked.connect(self.delete_all_traces)
		
		rpm_l = QtWidgets.QLabel("Source RPM")
		rpm_l.setToolTip("This helps avoid bad values in the Sine regression. If you don't know the source, measure the duration of one wow cycle. RPM = 60/cycle length")
		self.rpm_c = QtWidgets.QComboBox(self)
		self.rpm_c.setEditable(True)
		self.rpm_c.addItems(("Unknown","33.333","45","78"))
		self.rpm_c.currentIndexChanged.connect(self.update_other_settings)
		
		show_l = QtWidgets.QLabel("Show Speed for")
		self.show_c = QtWidgets.QComboBox(self)
		self.show_c.addItems(("Both","Traces only","Regressions only"))
		self.show_c.currentIndexChanged.connect(self.update_show_settings)
		
		self.autoalign_b = QtWidgets.QCheckBox("Auto-Align")
		self.autoalign_b.setChecked(True)
		self.autoalign_b.stateChanged.connect(self.update_other_settings)
		
		
		prec_l = QtWidgets.QLabel("Precision")
		self.prec_s = QtWidgets.QDoubleSpinBox()
		self.prec_s.setRange(0.0001, 100.0)
		self.prec_s.setSingleStep(0.1)
		self.prec_s.setValue(0.01)
		
		self.progressBar = QtWidgets.QProgressBar(self)
		self.progressBar.setRange(0,100)
		self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
		
		#channel_l = QtWidgets.QLabel("No Channels")
		self.mygroupbox = QtWidgets.QGroupBox('Channels')
		self.channel_layout = QtWidgets.QVBoxLayout()
		self.channel_layout.setSpacing(0)
		self.mygroupbox.setLayout(self.channel_layout)
		self.scroll = QtWidgets.QScrollArea()
		self.scroll.setWidget(self.mygroupbox)
		self.scroll.setWidgetResizable(True)
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(3)
		self.qgrid.setVerticalSpacing(0)
		#column 01
		self.qgrid.addWidget(self.open_b, 0, 0)
		self.qgrid.addWidget(self.save_b, 0, 1)
		
		self.qgrid.addWidget(fft_l, 1, 0)
		self.qgrid.addWidget(self.fft_c, 1, 1)
		
		self.qgrid.addWidget(overlap_l, 2, 0)
		self.qgrid.addWidget(self.overlap_c, 2, 1)
		
		self.qgrid.addWidget(cmap_l, 3, 0)
		self.qgrid.addWidget(self.cmap_c, 3, 1)
		
		#column 23
		self.qgrid.addWidget(trace_l, 0, 2)
		self.qgrid.addWidget(self.trace_c, 0, 3)
		self.qgrid.addWidget(self.delete_selected_b, 1, 2)
		self.qgrid.addWidget(self.delete_all_b, 1, 3)
		self.qgrid.addWidget(rpm_l, 2, 2)
		self.qgrid.addWidget(self.rpm_c, 2, 3)
		self.qgrid.addWidget(show_l, 3, 2)
		self.qgrid.addWidget(self.show_c, 3, 3)
		
		#column 45
		self.qgrid.addWidget(self.autoalign_b, 0, 4)
		self.qgrid.addWidget(self.resample_b, 0, 5)
		
		self.qgrid.addWidget(prec_l, 1, 4)
		self.qgrid.addWidget(self.prec_s, 1, 5)
		
		self.qgrid.addWidget(mode_l, 2, 4)
		self.qgrid.addWidget(self.mode_c, 2, 5)
		
		#column 6 - channels
		self.qgrid.addWidget(self.scroll, 0, 6, 4, 1)
		
		
		self.myLongTask = TaskThread()
		self.myLongTask.notifyProgress.connect(self.onProgress)
		
		self.qgrid.addWidget(self.progressBar, 3, 4, 1, 2 )

		vbox = QtWidgets.QVBoxLayout()
		vbox.addLayout(self.qgrid)
		vbox.addStretch(1.0)

		self.channel_bs = [ ]
		
		self.setLayout(vbox)
		
	
	def onProgress(self, i):
		self.progressBar.setValue(i)
		
	def open_audio(self):
		#pyqt5 returns a tuple
		self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Audio files (*.flac *.wav)")[0]
		if self.filename:
			self.settings_hard_changed.emit()
			data = resampling.read_trace(self.filename)
			for offset, times, freqs in data:
				TraceLine(self.parent.canvas, times, freqs, offset=offset)
			self.parent.canvas.master_speed.update()
			
			data = resampling.read_regs(self.filename)
			for t0, t1, amplitude, omega, phase, offset in data:
				RegLine(self.parent.canvas, t0, t1, amplitude, omega, phase, offset)
			self.parent.canvas.master_reg_speed.update()
			num_channels = 4
	
			soundob = sf.SoundFile(self.filename)
			num_channels = soundob.channels
			
			#delete all previous channel widgets
			for channel in self.channel_bs:
				self.channel_layout.removeWidget(channel)
				channel.deleteLater()
			
			self.channel_bs = []
			channel_names = ("Front Left", "Front Right", "Center", "LFE", "Back Left", "Back Right")
			#all active by default?
			active = [0,]
			for i in range(0, num_channels):
				name = channel_names[i] if i < 6 else str(i)
				self.channel_bs.append(QtWidgets.QCheckBox(name))
				if i in active:
					self.channel_bs[-1].setChecked(True)
				else:
					self.channel_bs[-1].setChecked(False)
				self.channel_bs[-1].stateChanged.connect(self.update_other_settings)
				self.channel_layout.addWidget( self.channel_bs[-1] )
			
	def save_traces(self):
		#get the data from the traces and save it
		data = []
		for line in self.parent.canvas.lines:
			data.append( (line.offset, line.times, line.freqs) )
		print("Saved",len(data),"traces")
		if data:
			resampling.write_trace(self.filename, data)
			# speed_data = self.parent.canvas.master_speed.get_linspace()
			# resampling.write_speed(self.filename, speed_data)
			
		#get the data from the regs and save it
		data = []
		for reg in self.parent.canvas.regs:
			#self.amplitude * np.sin(self.omega * clipped_times + self.phase) + self.offset
			data.append( (reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) )
		if data:
			resampling.write_regs(self.filename, data)
			#speed_data = self.parent.canvas.master_reg_speed.get_linspace()
			#resampling.write_speed(self.filename, speed_data)
			
	def delete_selected_tracess(self):
		for trace in reversed(self.parent.canvas.selected_traces):
			trace.remove()
		self.parent.canvas.master_speed.update()
		self.parent.canvas.master_reg_speed.update()
			
	def delete_all_traces(self):
		for line in reversed(self.parent.canvas.lines):
			line.remove()
		for reg in reversed(self.parent.canvas.regs):
			reg.remove()
		self.parent.canvas.master_speed.update()
		self.parent.canvas.master_reg_speed.update()
			
	def update_other_settings(self):
		self.parent.canvas.trace_mode = self.trace_c.currentText()
		self.parent.canvas.auto_align = self.autoalign_b.isChecked()
		self.parent.canvas.rpm = self.rpm_c.currentText()
	
	def update_show_settings(self):
		show = self.show_c.currentText()
		#"Traces only","Regressions only","Both"
		if show == "Traces only":
			self.parent.canvas.show_regs = False
			self.parent.canvas.show_lines = True
			self.parent.canvas.master_speed.show()
			for trace in self.parent.canvas.lines:
				trace.show()
			self.parent.canvas.master_reg_speed.hide()
			for reg in self.parent.canvas.regs:
				reg.hide()
		elif show == "Regressions only":
			self.parent.canvas.show_regs = True
			self.parent.canvas.show_lines = False
			self.parent.canvas.master_speed.hide()
			for trace in self.parent.canvas.lines:
				trace.hide()
			self.parent.canvas.master_reg_speed.show()
			for reg in self.parent.canvas.regs:
				reg.show()
		elif show == "Both":
			self.parent.canvas.show_regs = True
			self.parent.canvas.show_lines = True
			self.parent.canvas.master_speed.show()
			for trace in self.parent.canvas.lines:
				trace.show()
			self.parent.canvas.master_reg_speed.show()
			for reg in self.parent.canvas.regs:
				reg.show()
				
	def run_resample(self):
		if self.filename:
			mode = self.mode_c.currentText()
			prec = self.prec_s.value()
			#make a copy to prevent unexpected side effects
			channels = [i for i in range(len(self.channel_bs)) if self.channel_bs[i].isChecked()]
			print(channels)
			speed_curve = self.parent.canvas.master_speed.get_linspace()
			if self.parent.canvas.regs:
				speed_curve = self.parent.canvas.master_reg_speed.get_linspace()
				print("Using regressed speed")
			else:
				print("Using measured speed")
			print("Resampling",self.filename, mode, prec)
			self.myLongTask.settings = (self.filename, speed_curve, mode, prec, channels, "Diffused")
			self.myLongTask.start()
			
	def update_resampling_presets(self, option):
		mode = self.mode_c.currentText()
		if mode == "Expansion":
			self.prec_s.setValue(5.0)
		elif mode == "Blocks":
			self.prec_s.setValue(0.01)
		elif mode == "Sinc":
			self.prec_s.setValue(0.05)
		elif mode == "Linear":
			self.prec_s.setValue(0.01)
		
	def update_param_hard(self, option):
		self.settings_hard_changed.emit()
		
	def update_param_soft(self, option):
		self.settings_soft_changed.emit()


class MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)

		self.resize(700, 500)
		self.setWindowTitle('pyrespeeder 2.0')

		splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

		self.canvas = Canvas()
		self.canvas.create_native()
		self.canvas.native.setParent(self)

		self.props = ObjectWidget(parent=self)
		splitter.addWidget(self.props)
		splitter.addWidget(self.canvas.native)

		self.setCentralWidget(splitter)
		self.props.settings_hard_changed.connect(self.update_settings_hard)
		self.props.settings_soft_changed.connect(self.update_settings_soft)

	def update_settings_hard(self):
		self.canvas.set_data_hard(self.props.filename,
								 fft_size = int(self.props.fft_c.currentText()),
								 fft_overlap = int(self.props.overlap_c.currentText()))
		#also force a soft update here
		self.update_settings_soft()
		
	def update_settings_soft(self):
		self.canvas.set_data_soft(cmap=self.props.cmap_c.currentText())	

class MasterSpeedLine:
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		
		self.vispy_canvas = vispy_canvas
		
		#create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.data[:, 1] = (0, 0)
		self.line_speed = scene.Line(pos=self.data, color=(1, 0, 0, .5), method='gl')
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		
	def update(self):
		num = self.vispy_canvas.num_ffts
		#get the times at which the average should be sampled
		times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
		#create the array for sampling
		out = np.zeros((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
		#lerp and sample all lines, use NAN for missing parts
		for i, line in enumerate(self.vispy_canvas.lines):
			line_sampled = np.interp(times, line.times, line.speed, left = np.nan, right = np.nan)
			out[:, i] = line_sampled
		#take the mean and ignore nans
		mean_with_nans = np.nanmean(out, axis=1)
		mean_with_nans[np.isnan(mean_with_nans)]=0
		#set the output data
		self.data = np.zeros((len(times), 2), dtype=np.float32)
		self.data[:, 0] = times
		self.data[:, 1] = mean_with_nans
		self.line_speed.set_data(pos=self.data)
		
	def get_linspace(self):
		"""Convert the log2 spaced speed curve back into linspace for further processing"""
	
		out = np.array(self.data)
		lin_scale = np.power(2, out[:,1])
		#print(lin_scale/np.mean(lin_scale))
		out[:,1] = lin_scale#/np.mean(lin_scale)
		#print(out[:,1])
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
			
			num = self.vispy_canvas.num_ffts
			#get the times at which the average should be sampled
			times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
			
			#sample all regressions
			reg_data = []
			offsets_sampled = 0
			#amplitudes_sampled = 0
			for reg in self.vispy_canvas.regs:
				reg_data.append((reg.t_center, reg.amplitude, reg.omega, reg.phase % (2*np.pi), reg.offset))
				#amplitudes_sampled+=reg.amplitude
			#amplitudes_sampled/=len(self.vispy_canvas.regs)
			#interp needs x keys to be sorted
			reg_data.sort(key=lambda tup: tup[0])
			
			#create lists
			phi_centers = []
			t_centers = []
			amp_centers = []
			
			t_center, amplitude, omega, phase, offset = reg_data[0]
			phi_centers.append(omega * times[0] + phase)
			t_centers.append(times[0])
			amp_centers.append(amplitude)
			
			for t_center, amplitude, omega, phase, offset in reg_data:
				phi_centers.append(omega * t_center + phase)
				t_centers.append(t_center)
				amp_centers.append(amplitude)
				
			#do the last one
			t_center, amplitude, omega, phase, offset = reg_data[-1]
			phi_centers.append(omega * times[-1] + phase)
			t_centers.append(times[-1])
			amp_centers.append(amplitude)
			
			
			#phi = omegas_sampled * times + phases_sampled
			phi = np.interp(times, t_centers, phi_centers)
			amplitudes_sampled = np.interp(times, t_centers, amp_centers)
			
			#create the speed curve visualization
			self.data = np.zeros((len(times), 2), dtype=np.float32)
			self.data[:, 0] = times
			#boost it a bit
			self.data[:, 1] = 1.5 * amplitudes_sampled * np.sin(phi) + offsets_sampled
			
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
		lin_scale = np.power(2, out[:,1])
		#print(lin_scale/np.mean(lin_scale))
		out[:,1] = lin_scale#/np.mean(lin_scale)
		#print(out[:,1])
		return out
		
class RegLine:
	"""Stores a single sinc regression's data and displays it"""
	def __init__(self, vispy_canvas, t0, t1, amplitude, omega, phase, offset):
	
		self.vispy_canvas = vispy_canvas
		
		#the extents on which this regression operated
		self.t0 = t0
		self.t1 = t1
		#here, the reg values are most accurate
		self.t_center = (t0+ t1)/2
		self.speed_center = np.array( (self.t_center, 0) )
		
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
		#self.offset = offset
		#for now
		self.offset = 0
		
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
		self.data = np.zeros((len(clipped_times), 2), dtype=np.float32)
		self.data[:, 0] = clipped_times
		self.data[:, 1] = self.amplitude * np.sin(self.omega * clipped_times + self.phase) + self.offset
		#sine_on_hz = np.power(2, sine + np.log2(2000))
		self.line_speed = scene.Line(pos=self.data, color=(0, 0, 1, .5), method='gl')
		self.show()
		#self.line_speed.parent = vispy_canvas.speed_view.scene
		self.vispy_canvas.regs.append(self)
		self.vispy_canvas.master_reg_speed.update()
		
	def set_offset(self, a, b):
		#print(a,b,b/a)
		self.amplitude *= (b/a)
		self.data[:, 1]*= (b/a)
		self.line_speed.set_data(pos=self.data)
		
	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.line_speed.set_data(color = (1, 1, 1, .5))
		#self.line_spec.set_data(color = (1, 1, 1, 1))
		self.vispy_canvas.selected_traces.remove(self)
		
	def select(self, multi=False):
		"""Toggle this line's selection state"""
		if not multi:
			for trace in reversed(self.vispy_canvas.selected_traces):
				trace.deselect()
		if self in self.vispy_canvas.selected_traces:
			self.deselect()
		else:
			self.line_speed.set_data(color = (0, 1, 0, 1))
			#self.line_spec.set_data(color = (0, 1, 0, 1))
			self.vispy_canvas.selected_traces.append(self)
			
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		
	def remove(self):
		self.line_speed.parent = None
		#self.line_spec.parent = None
		#note: this has to search the list
		self.vispy_canvas.regs.remove(self)
		self.vispy_canvas.master_speed.update()
		if self in self.vispy_canvas.selected_traces:
			self.deselect()
	

class TraceLine:
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, times, freqs, offset=None):
		
		self.vispy_canvas = vispy_canvas
		
		self.times = np.asarray(times)
		self.freqs = np.asarray(freqs)
		
		mean_freqs = np.mean(self.freqs)
		mean_times = np.mean(self.times)
		
		
		#note: the final, output speed curve output should be linscale and centered on 1
		self.speed = np.log2(freqs)
		self.speed-= np.mean(self.speed)
		#we don't want to overwrite existing offsets loaded from files
		if offset is None:
			if not vispy_canvas.auto_align:
				#print("no align")
				offset = 0
			else:
				#print("Setting automatic offset")
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
		
		self.spec_center = np.array( (mean_times, mean_freqs) )
		self.speed_center = np.array( (mean_times, np.mean(self.speed)) )
		
		data = np.ones((len(times), 3), dtype=np.float32)*-2
		data[:, 0] = times
		data[:, 1] = freqs
		
		#create the spectral visualization
		self.line_spec = scene.Line(pos=data, color=(1, 1, 1, 1), method='gl')
		#the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.line_spec.transform =	vispy_ext.MelTransform()
		self.line_spec.parent = vispy_canvas.spec_view.scene
		
		#create the speed curve visualization
		self.speed_data = np.ones((len(times), 2), dtype=np.float32)
		self.speed_data[:, 0] = self.times
		self.speed_data[:, 1] = self.speed
		self.line_speed = scene.Line(pos=self.speed_data, color=(1, 1, 1, .5), method='gl')
		self.line_speed.parent = vispy_canvas.speed_view.scene
		#self.vispy_canvas.master_speed.update()
		self.vispy_canvas.lines.append(self)
		
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		
	def set_offset(self, a, b):
		offset = b-a
		#print("offset",offset)
		self.offset += offset
		self.speed_center[1] += offset
		self.speed += offset
		#print("new center",self.center)
		self.speed_data[:, 1] = self.speed
		self.line_speed.set_data(pos = self.speed_data)
		#self.vispy_canvas.master_speed.update()
	
	
	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.line_speed.set_data(color = (1, 1, 1, .5))
		self.line_spec.set_data(color = (1, 1, 1, 1))
		self.vispy_canvas.selected_traces.remove(self)
		
	def select(self, multi=False):
		"""Toggle this line's selection state"""
		if not multi:
			for trace in reversed(self.vispy_canvas.selected_traces):
				trace.deselect()
		if self in self.vispy_canvas.selected_traces:
			self.deselect()
		else:
			self.line_speed.set_data(color = (0, 1, 0, 1))
			self.line_spec.set_data(color = (0, 1, 0, 1))
			self.vispy_canvas.selected_traces.append(self)
		
	def remove(self):
		self.line_speed.parent = None
		self.line_spec.parent = None
		#note: this has to search the list
		self.vispy_canvas.lines.remove(self)
		self.vispy_canvas.master_speed.update()
		if self in self.vispy_canvas.selected_traces:
			self.deselect()
	
class Canvas(scene.SceneCanvas):

	def __init__(self):
		
		#some default dummy values
		self.filename = "None"
		cm = "fire"
		self.vmin = -80
		self.vmax = -40
		self.auto_align = True
		self.trace_mode = "Center of Gravity"
		self.rpm = "Unknown"
		self.selected_traces = []
		self.show_regs = True
		self.show_lines = True
		
		self.last_click = None
		self.fft_size = 1024
		self.hop = 256
		self.sr = 44100
		self.num_ffts = 0
		
		self.MAX_TEXTURE_SIZE = None
		
		scene.SceneCanvas.__init__(self, keys="interactive", size=(1024, 512), bgcolor="grey")
		
		self.unfreeze()
		
		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0
		
		#speed chart
		self.speed_yaxis = scene.AxisWidget(orientation='left',
								 axis_label='Octaves',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.speed_yaxis.width_max = 55
		
		#spectrum
		self.spec_yaxis = scene.AxisWidget(orientation='left',
								 axis_label='Hz',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.spec_yaxis.width_max = 55
		
		self.spec_xaxis = scene.AxisWidget(orientation='bottom',
								 axis_label='sec',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.spec_xaxis.height_max = 55

		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 20
		
		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 70

		#create the color bar display
		self.colorbar_display = scene.ColorBarWidget(label="Gain [dB]", clim=(self.vmin, self.vmax),
										   cmap=cm, orientation="right",
											border_width=1)
		self.colorbar_display.label.font_size = 10
		self.colorbar_display.label.color = "white"

		self.colorbar_display.ticks[0].font_size = 8
		self.colorbar_display.ticks[1].font_size = 8
		self.colorbar_display.ticks[0].color = "white"
		self.colorbar_display.ticks[1].color = "white"
		
		#grid.add_widget(self.header, row=0, col=0, col_span=2)
		grid.add_widget(self.speed_yaxis, row=1, col=0)
		#grid.add_widget(self.speed_xaxis, row=1, col=1)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		self.speed_view = grid.add_view(row=1, col=1, border_color='white')
		self.spec_view = grid.add_view(row=2, col=1, border_color='white')
		grid.add_widget(self.colorbar_display, row=2, col=2)
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, -1.0, 10, 2.0), )
		
		self.speed_view.height_min = 150
		
		#TODO: make sure they are bound when started, not just after scrolling
		#self.speed_xaxis.link_view(self.speed_view)
		self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)
		
		self.images = []
		self.lines = []
		self.regs = []
		self.fft_storage = {}
		
		self.master_speed = MasterSpeedLine(self)
		self.master_reg_speed = MasterRegLine(self)
		
		self.freeze()
		
		
	#fast stuff that does not require rebuilding everything
	def set_data_soft(self, cmap="fire"):
		for image in self.images:
			image.set_cmap(cmap)
		self.colorbar_display.cmap = cmap
	
		
	def set_data_hard(self, filename, fft_size = 256, fft_overlap = 1):
		if filename:
			
			#only set this one once, but can't set it in _init_
			if not self.MAX_TEXTURE_SIZE:
				self.MAX_TEXTURE_SIZE = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)
				print("self.MAX_TEXTURE_SIZE", self.MAX_TEXTURE_SIZE)
			
			#has the file changed?
			if self.filename != filename:
				#clear the fft storage!
				self.fft_storage = {}
				
				print("removing old lines")
				for line in reversed(self.lines):
					line.remove()
					
			soundob = sf.SoundFile(filename)
			sr = soundob.samplerate
			
			hop = int(fft_size/fft_overlap)
			
			#set this for the tracers etc.
			self.fft_size = fft_size
			self.hop = hop
			self.sr = sr
			
			print("fft_size,hop",fft_size,hop)
			
			#store the FFTs for fast shuffling around
			#TODO: analyze the RAM consumption of this
			#TODO: perform FFT at minimal hop and get shorter hops via stride operation ::x
			k = (fft_size, hop)
			if k not in self.fft_storage:
				print("storing new fft")
				signal = soundob.read(always_2d=True)[:,0]
				#this will automatically zero-pad the last fft
				#get the magnitude spectrum
				#avoid divide by 0 error in log10
				imdata = np.abs(fourier.stft(signal, fft_size, hop, "hann")+.0000001)
				#change to dB scale later, for the tracers
				#imdata = 20 * np.log10(imdata)
				#clamping the data to 0,1 range happens in the vertex shader
				
				#now store this for retrieval later
				self.fft_storage[k] = imdata.astype('float32')
			#retrieve the FFT data
			imdata = self.fft_storage[k]
			num_bins, num_ffts = imdata.shape
			self.num_ffts = num_ffts

			#has the file changed?
			if self.filename != filename:
				print("file has changed!")
				self.filename = filename
				#self.header.text = filename
				
				#(re)set the spec_view
				#only the camera dimension is mel'ed, as the image gets it from its transform
				self.speed_view.camera.rect = (0, -1.0, num_ffts * hop / sr, 2.0)
				self.spec_view.camera.rect = (0, 0, num_ffts * hop / sr, to_mel(sr//2))
				#link them, but use custom logic to only link the x view
				self.spec_view.camera.link(self.speed_view.camera)
				
			#TODO: is this as efficient as possible?
			print("removing old images")
			for i in reversed(range(0, len(self.images))):
				self.images[i].parent = None
				self.images.pop(i)
				
			#spectra may only be of a certain size, so split them
			#to support 2**17 FFT sizes, it would also need to split along the Y axis
			for i in range(0, num_ffts, self.MAX_TEXTURE_SIZE):
				imdata_piece = imdata[:,i:i+self.MAX_TEXTURE_SIZE]
				
				#do the dB conversion here because the tracers don't like it
				self.images.append(SpectrumPiece(20 * np.log10(imdata_piece), parent=self.spec_view.scene))
				
				self.images[-1].set_clims(self.vmin, self.vmax)
				self.images[-1].set_size((imdata_piece.shape[1]*hop/sr, sr//2))
				#add this piece's offset with STT
				self.images[-1].transform = visuals.transforms.STTransform( translate=(i * hop / sr, 0)) * vispy_ext.MelTransform()
			self.master_speed.update()
			self.master_reg_speed.update()
		
	def on_mouse_wheel(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		
		#colorbar scroll
		if self.click_on_widget(click, self.colorbar_display):
			grid_space = self.colorbar_display._colorbar.transform.imap(click)
			dim = self.colorbar_display.size
			d = event.delta
			self.vmin, self.vmax = self.colorbar_display.clim
			
			#now split Y in three parts
			a = dim[1]/3
			b = a*2
			if grid_space[1] < a:
				self.vmax += d[1]
			elif a < grid_space[1] < b:
				self.vmin += d[1]
				self.vmax -= d[1]
			elif b < grid_space[1]:
				self.vmin += d[1]

			self.colorbar_display.clim = (int(self.vmin), int(self.vmax))
			for image in self.images:
				image.set_clims(self.vmin, self.vmax)
				
		#spec & speed X axis scroll
		if self.click_on_widget(click, self.spec_xaxis):
			#the center of zoom should be assigned a new x coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (scene_space[0], self.spec_view.camera.center[1])
			self.spec_view.camera.zoom(((1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30), 1), c)

		#spec Y axis scroll
		if self.click_on_widget(click, self.spec_yaxis):
			#the center of zoom should be assigned a new y coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (self.spec_view.camera.center[0], scene_space[1])
			self.spec_view.camera.zoom((1, (1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)
		
		#speed Y axis scroll
		if self.click_on_widget(click, self.speed_yaxis):
			#the center of zoom should be assigned a new y coordinate
			grid_space = self.speed_view.transform.imap(click)
			scene_space = self.speed_view.scene.transform.imap(grid_space)
			c = (self.speed_view.camera.center[0], scene_space[1])
			self.speed_view.camera.zoom((1, (1 + self.speed_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)

	def on_mouse_press(self, event):
		#coords of the click on the vispy canvas
		self.last_click = None
		click = np.array([event.pos[0],event.pos[1],0,1])
		if event.button == 1:
			if "Control" in event.modifiers:
				self.last_click = click
					
		#selection, single or multi
		if event.button == 2:
			closest_line = self.get_closest_line( click )
			if closest_line:
				if "Shift" in event.modifiers:
					closest_line.select(multi=True)
					event.handled = True
				else:
					closest_line.select()
					event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		click = event.pos
		if self.last_click is not None:
			a = self.click_spec_conversion(self.last_click)
			b = self.click_spec_conversion(click)
			#are they in spec_view?
			if a is not None and b is not None:
				t0, t1 = sorted((a[0], b[0]))
				f0, f1 = sorted((a[1], b[1]))
				t0 = max(0, t0)
				fft_key = (self.fft_size, self.hop)
				#maybe query it here from the button instead of the other way
				if self.trace_mode == "Sine Regression":
					amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_speed.get_linspace(), t0, t1, self.rpm)
					RegLine(self, t0, t1, amplitude, omega, phase, offset)
					return
				else:
					if self.trace_mode == "Center of Gravity":
						times, freqs = wow_detection.trace_cog(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
					elif self.trace_mode == "Correlation":
						times, freqs = wow_detection.trace_correlation(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, t0 = t0, t1 = t1)
					elif self.trace_mode == "Sine Draw":
						times, freqs = wow_detection.trace_sine(fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
					elif self.trace_mode == "Freehand Draw":
						#TODO: vectorize this: a[a[:,0].argsort()]
						#TODO: reduce resolution with np.interp at speed curve sample rate
						data = [self.click_spec_conversion(click) for click in event.trail()]
						data.sort(key=lambda tup: tup[0])
						times = [d[0] for d in data]
						freqs = [d[1] for d in data]
					if len(freqs) and np.nan not in freqs:
						TraceLine(self, times, freqs)
						self.master_speed.update()
						return
			
			#or in speed view?
			#then we are only interested in the Y difference, so we can move the selected speed trace up or down
			a = self.click_speed_conversion(self.last_click)
			b = self.click_speed_conversion(click)
			if a is not None and b is not None:
				#diff = b[1]-a[1]
				for trace in self.selected_traces:
					trace.set_offset(a[1], b[1])
				self.master_speed.update()
				self.master_reg_speed.update()
				
				
	def get_closest_line(self, click):
		if click is not None:
			#first, check in speed view
			c = self.click_speed_conversion(click)
			if c is not None:
				if self.show_regs and self.show_lines:
					A = np.array([line.speed_center for line in self.lines] + [reg.speed_center for reg in self.regs])
				elif self.show_regs and not self.show_lines:
					A = np.array([reg.speed_center for reg in self.regs])
				elif not self.show_regs and self.show_lines:
					A = np.array([line.speed_center for line in self.lines])
			else:
				#then, check in spectral view, and only true lines, no regs here
				if self.show_lines:
					c = self.click_spec_conversion(click)
					A = np.array([line.spec_center for line in self.lines])
			#returns the line (if any exists) whose center (including any offsets) is closest to pt
			if c is not None and len(A):
				ind = np.array([np.linalg.norm(x+y) for (x,y) in A-(c[0], c[1])]).argmin()
				if self.show_regs and self.show_lines:
					return (self.lines+self.regs)[ind]
				elif self.show_regs and not self.show_lines:
					return self.regs[ind]
				elif not self.show_regs and self.show_lines:
					return self.lines[ind]
	
	def click_on_widget(self, click, wid):
		grid_space = wid.transform.imap(click)
		dim = wid.size
		return (0 < grid_space[0] < dim[0]) and (0 < grid_space[1] < dim[1])
	
	def click_speed_conversion(self, click):
		#in the grid on the canvas
		grid_space = self.speed_view.transform.imap(click)
		#is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.speed_view):
			return self.speed_view.scene.transform.imap(grid_space)
				
	def click_spec_conversion(self, click):
		#in the grid on the canvas
		grid_space = self.spec_view.transform.imap(click)
		#is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.spec_view):
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			#careful, we need a spectrum already
			if self.images:
				#in fact the simple Y mel transform would be enough in any case
				#but this would also support other transforms
				return self.images[0].transform.imap(scene_space)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	appQt = QtWidgets.QApplication(sys.argv)
	win = MainWindow()
	win.show()
	appQt.exec_()
