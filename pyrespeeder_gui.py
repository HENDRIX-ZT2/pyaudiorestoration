# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


#import sys
import os
import numpy as np
from vispy import scene, gloo, visuals, color, util, io
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

mel_transform = vispy_ext.MelTransform()

class Spectrum():
	"""
	The visualization of the whole spectrogram.
	"""
	def __init__(self, parent):
		self.pieces = []
		self.parent = parent
		self.MAX_TEXTURE_SIZE = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)
	
	def update_data(self, imdata, hop, sr):
		num_bins, num_ffts = imdata.shape
		#TODO: only delete / add visuals when required due to changing data, not always
		print("removing old pieces")
		for i in reversed(range(0, len(self.pieces))):
			self.pieces[i].parent = None
			self.pieces.pop(i)
		#this seems to be correct, so could be used to decimate the old pieces
		num_pieces_new = num_ffts//self.MAX_TEXTURE_SIZE + 1
		#spectra may only be of a certain size, so split them
		for x in range(0, num_ffts, self.MAX_TEXTURE_SIZE):
			for y in range(0, num_bins, self.MAX_TEXTURE_SIZE):
				imdata_piece = imdata[y:y+self.MAX_TEXTURE_SIZE, x:x+self.MAX_TEXTURE_SIZE]
				num_piece_bins, num_piece_ffts = imdata_piece.shape
				if num_piece_bins > 1 and num_piece_ffts > 1:
					#do the dB conversion here because the tracers don't like it
					self.pieces.append(SpectrumPiece(20 * np.log10(imdata_piece), parent=self.parent.scene))
					
					#to get correct heights for subsequent pieces, the start y has to be added in mel space
					height_Hz_in = int(num_piece_bins/num_bins*sr/2)
					ystart_Hz = (y/num_bins*sr/2)
					height_Hz_corrected = to_Hz(to_mel(height_Hz_in + ystart_Hz) - to_mel(ystart_Hz))
					
					self.pieces[-1].set_size((num_piece_ffts*hop/sr, height_Hz_corrected))

					#add this piece's offset with STT
					self.pieces[-1].transform = visuals.transforms.STTransform( translate=(x * hop / sr, to_mel(ystart_Hz))) * mel_transform
		
	def set_clims(self, vmin, vmax):
		for image in self.pieces:
			image.set_clims(vmin, vmax)
			
	def set_cmap(self, colormap):
		for image in self.pieces:
			image.set_cmap(colormap)

class SpectrumPiece(scene.Image):
	"""
	The visualization of one part of the whole spectrogram.
	"""
	def __init__(self, texdata, parent):
		#just set a dummy value
		self._shape = (10.0, 22500)
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

def to_Hz(val):
	### just to set the image size correctly	
	return (np.exp(val / 1127) - 1) * 700
	
class ResamplingThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	def run(self):
		name, speed_curve, resampling_mode, sinc_quality, use_channels = self.settings
		resampling.run(name, speed_curve= speed_curve, resampling_mode = resampling_mode, sinc_quality=sinc_quality, use_channels=use_channels, prog_sig=self)
			
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""
	file_or_fft_settings_changed = QtCore.pyqtSignal(name='objectChanged')
	settings_soft_changed = QtCore.pyqtSignal(name='objectChanged2')

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.filename = ""
		self.deltraces = []
		
		myFont=QtGui.QFont()
		myFont.setBold(True)

		display_l = QtWidgets.QLabel("Display")
		display_l.setFont(myFont)
		
		fft_l = QtWidgets.QLabel("FFT Size")
		self.fft_c = QtWidgets.QComboBox(self)
		self.fft_c.addItems(("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072"))
		self.fft_c.setToolTip("This determines the frequency resolution.")
		self.fft_c.currentIndexChanged.connect(self.update_param_hard)
		self.fft_c.setCurrentIndex(5)
		
		overlap_l = QtWidgets.QLabel("FFT Overlap")
		self.overlap_c = QtWidgets.QComboBox(self)
		self.overlap_c.addItems(("1", "2", "4", "8", "16", "32"))
		self.overlap_c.setToolTip("Increase to improve temporal resolution.")
		self.overlap_c.currentIndexChanged.connect(self.update_param_hard)
		self.overlap_c.setCurrentIndex(2)
		
		cmap_l = QtWidgets.QLabel("Colors")
		self.cmap_c = QtWidgets.QComboBox(self)
		self.cmap_c.addItems(sorted(color.colormap.get_colormaps().keys()))
		self.cmap_c.setCurrentText("viridis")
		self.cmap_c.currentIndexChanged.connect(self.update_param_soft)

		tracing_l = QtWidgets.QLabel("\nTracing")
		tracing_l.setFont(myFont)
		trace_l = QtWidgets.QLabel("Mode")
		self.trace_c = QtWidgets.QComboBox(self)
		self.trace_c.addItems(("Center of Gravity","Peak","Correlation","Freehand Draw", "Sine Regression"))
		self.trace_c.currentIndexChanged.connect(self.update_other_settings)
		
		adapt_l = QtWidgets.QLabel("Adaptation")
		self.adapt_c = QtWidgets.QComboBox(self)
		self.adapt_c.addItems(("Average", "Linear", "Constant", "None"))
		self.adapt_c.setToolTip("Used to predict the next frequencies when tracing.")
		self.adapt_c.currentIndexChanged.connect(self.update_other_settings)
		
		rpm_l = QtWidgets.QLabel("Source RPM")
		self.rpm_c = QtWidgets.QComboBox(self)
		self.rpm_c.setEditable(True)
		self.rpm_c.addItems(("Unknown","33.333","45","78"))
		self.rpm_c.currentIndexChanged.connect(self.update_other_settings)
		self.rpm_c.setToolTip("This helps avoid bad values in the sine regression. \nIf you don't know the source, measure the duration of one wow cycle. \nRPM = 60/cycle length")
		
		show_l = QtWidgets.QLabel("Show")
		self.show_c = QtWidgets.QComboBox(self)
		self.show_c.addItems(("Both","Traces","Regressions"))
		self.show_c.currentIndexChanged.connect(self.update_show_settings)
		
		self.autoalign_b = QtWidgets.QCheckBox("Auto-Align")
		self.autoalign_b.setChecked(True)
		self.autoalign_b.stateChanged.connect(self.update_other_settings)
		self.autoalign_b.setToolTip("Should new traces be aligned with existing ones?")
		
		
		resampling_l = QtWidgets.QLabel("\nResampling")
		resampling_l.setFont(myFont)
		mode_l = QtWidgets.QLabel("Mode")
		self.mode_c = QtWidgets.QComboBox(self)
		self.mode_c.addItems(("Linear", "Sinc"))
		sinc_quality_l = QtWidgets.QLabel("Sinc Quality")
		self.sinc_quality_s = QtWidgets.QSpinBox()
		self.sinc_quality_s.setRange(1, 100)
		self.sinc_quality_s.setSingleStep(1)
		self.sinc_quality_s.setValue(50)
		self.sinc_quality_s.setToolTip("Number of input samples that contribute to each output sample.\nMore samples = more quality, but slower. Only for sinc mode.")
		
		self.progressBar = QtWidgets.QProgressBar(self)
		self.progressBar.setRange(0,100)
		self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
		
		
		phase_l = QtWidgets.QLabel("Phase Offset")
		self.phase_s = QtWidgets.QSpinBox()
		self.phase_s.setRange(-20, 20)
		self.phase_s.setSingleStep(1)
		self.phase_s.setValue(0)
		self.phase_s.valueChanged.connect(self.update_phase_offset)
		self.phase_s.setToolTip("Adjust the phase of the selected sine regression to match the surrounding regions.")
		
		self.mygroupbox = QtWidgets.QGroupBox('Channels')
		self.mygroupbox.setToolTip("Only selected channels will be resampled.")
		self.channel_layout = QtWidgets.QVBoxLayout()
		self.channel_layout.setSpacing(0)
		self.mygroupbox.setLayout(self.channel_layout)
		self.scroll = QtWidgets.QScrollArea()
		self.scroll.setWidget(self.mygroupbox)
		self.scroll.setWidgetResizable(True)
		
		self.inspector_l = QtWidgets.QLabel("\n        -.- Hz\n-:--:--:--- h:m:s:ms")
		myFont2=QtGui.QFont("Monospace")
		myFont2.setStyleHint(QtGui.QFont.TypeWriter)
		self.inspector_l.setFont(myFont2)
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(3)
		self.qgrid.setVerticalSpacing(0)
		
		buttons = [(display_l,), (fft_l, self.fft_c), (overlap_l, self.overlap_c), (show_l, self.show_c), (cmap_l,self.cmap_c), \
					(tracing_l,), (trace_l, self.trace_c), (adapt_l, self.adapt_c), (rpm_l,self.rpm_c), (phase_l, self.phase_s), (self.autoalign_b, ), \
					(resampling_l, ), (mode_l, self.mode_c), (sinc_quality_l, self.sinc_quality_s), (self.scroll,), (self.progressBar,), (self.inspector_l,) ]
		for i, line in enumerate(buttons):
			for j, element in enumerate(line):
				#we want to stretch that one
				if 1 == len(line):
					self.qgrid.addWidget(line[j], i, j, 1, 2)
				else:
					self.qgrid.addWidget(line[j], i, j)
		
		self.resampling_thread = ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.onProgress)
		
		
		for i in range(2):
			self.qgrid.setColumnStretch(i, 1)
		vbox = QtWidgets.QVBoxLayout()
		vbox.addLayout(self.qgrid)
		vbox.addStretch(1.0)

		self.channel_checkboxes = [ ]
		self.setLayout(vbox)
		
	def onProgress(self, i):
		self.progressBar.setValue(i)
		
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
				self.parent.canvas.fft_storage = {}
				self.delete_traces(not_only_selected=True)
				for channel in self.channel_checkboxes:
					self.channel_layout.removeWidget(channel)
					channel.deleteLater()
				self.channel_checkboxes = []
				
				#fill the channel UI
				channel_names = ("Front Left", "Front Right", "Center", "LFE", "Back Left", "Back Right")
				num_channels = soundob.channels
				for i in range(0, num_channels):
					name = channel_names[i] if i < 6 else str(i)
					self.channel_checkboxes.append(QtWidgets.QCheckBox(name))
					# set the startup option to just resample channel 0
					if i == 0:
						self.channel_checkboxes[-1].setChecked(True)
					else:
						self.channel_checkboxes[-1].setChecked(False)
					self.channel_checkboxes[-1].stateChanged.connect(self.update_other_settings)
					self.channel_layout.addWidget( self.channel_checkboxes[-1] )
				
				#finally - proceed with spectrum stuff elsewhere
				self.parent.setWindowTitle('pyrespeeder '+os.path.basename(self.filename))
				self.file_or_fft_settings_changed.emit()
				
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
		#get the data from the traces and save it
		data = [ (line.offset, line.times, line.freqs) for line in self.parent.canvas.lines ]
		if data:
			print("Saved",len(data),"traces")
			resampling.write_trace(self.filename, data)
		#get the data from the regressions and save it
		data = [ (reg.t0, reg.t1, reg.amplitude, reg.omega, reg.phase, reg.offset) for reg in self.parent.canvas.regs ]
		if data:
			print("Saved",len(data),"regressions")
			resampling.write_regs(self.filename, data)
			
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
	
	def restore_traces(self):
		for trace in self.deltraces:
			trace.initialize()
		self.parent.canvas.master_speed.update()
		self.parent.canvas.master_reg_speed.update()
		self.deltraces = []
			
	def update_phase_offset(self):
		v = self.phase_s.value()
		for reg in self.parent.canvas.regs:
			reg.update_phase(v)
		self.parent.canvas.master_reg_speed.update()
	
	def update_other_settings(self):
		self.parent.canvas.trace_mode = self.trace_c.currentText()
		self.parent.canvas.adapt_mode = self.adapt_c.currentText()
		self.parent.canvas.auto_align = self.autoalign_b.isChecked()
		self.parent.canvas.rpm = self.rpm_c.currentText()
		
	def update_show_settings(self):
		show = self.show_c.currentText()
		if show == "Traces":
			self.parent.canvas.show_regs = False
			self.parent.canvas.show_lines = True
			self.parent.canvas.master_speed.show()
			for trace in self.parent.canvas.lines:
				trace.show()
			self.parent.canvas.master_reg_speed.hide()
			for reg in self.parent.canvas.regs:
				reg.hide()
		elif show == "Regressions":
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
		if self.filename and self.parent.canvas.lines:
			channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
			if channels:
				if self.parent.canvas.regs:
					speed_curve = self.parent.canvas.master_reg_speed.get_linspace()
					print("Using regressed speed")
				else:
					speed_curve = self.parent.canvas.master_speed.get_linspace()
					print("Using measured speed")
				self.resampling_thread.settings = (self.filename, speed_curve, self.mode_c.currentText(), self.sinc_quality_s.value(), channels)
				self.resampling_thread.start()
			
	def update_param_hard(self, option):
		self.file_or_fft_settings_changed.emit()
		
	def update_param_soft(self, option):
		self.settings_soft_changed.emit()
		
	def foo(self):
		print("foo")
		
	def select_all(self):
		for trace in self.parent.canvas.lines+self.parent.canvas.regs:
			trace.select()
		
	def invert_selection(self):
		for trace in self.parent.canvas.lines+self.parent.canvas.regs:
			trace.toggle()
		
class MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)		
		
		self.resize(720, 400)
		self.setWindowTitle('pyrespeeder')
		try:
			scriptDir = os.path.dirname(os.path.realpath(__file__))
			self.setWindowIcon(QtGui.QIcon(os.path.join(scriptDir,'icon.png')))
		except: pass
		
		self.setAcceptDrops(True)
		splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

		self.canvas = Canvas()
		self.canvas.create_native()
		self.canvas.native.setParent(self)

		self.props = ObjectWidget(parent=self)
		splitter.addWidget(self.canvas.native)
		splitter.addWidget(self.props)

		self.canvas.props = self.props
		self.setCentralWidget(splitter)
		self.props.file_or_fft_settings_changed.connect(self.update_settings_hard)
		self.props.settings_soft_changed.connect(self.update_settings_soft)
		
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
						(editMenu, "Delete Selected", self.props.delete_traces, "DEL"), \
						)
		
		for submenu, name, func, shortcut in button_data:
			button = QtWidgets.QAction(name, self)
			button.triggered.connect(func)
			if shortcut: button.setShortcut(shortcut)
			submenu.addAction(button)
		
	def update_settings_hard(self):
		self.canvas.set_file_or_fft_settings(self.props.filename,
								 fft_size = int(self.props.fft_c.currentText()),
								 fft_overlap = int(self.props.overlap_c.currentText()))
		#also force a soft update here
		self.update_settings_soft()
		
	def dragEnterEvent(self, event):
		if event.mimeData().hasUrls:
			event.accept()
		else:
			event.ignore()

	def dragMoveEvent(self, event):
		if event.mimeData().hasUrls:
			event.setDropAction(QtCore.Qt.CopyAction)
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		if event.mimeData().hasUrls:
			event.setDropAction(QtCore.Qt.CopyAction)
			event.accept()
			for url in event.mimeData().urls():
				self.props.load_audio( str(url.toLocalFile()) )
				return
		else:
			event.ignore()

	def update_settings_soft(self):
		self.canvas.set_colormap(self.props.cmap_c.currentText())	

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
		"""Convert the log2 spaced speed curve back into linspace for direct use in resampling"""
		out = np.array(self.data)
		lin_scale = np.power(2, out[:,1])
		out[:,1] = lin_scale
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
		lin_scale = np.power(2, out[:,1])
		out[:,1] = lin_scale
		return out
		
class RegLine:
	"""Stores a single sinc regression's data and displays it"""
	def __init__(self, vispy_canvas, t0, t1, amplitude, omega, phase, offset):
	
		self.selected = False
	
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
		self.data = np.zeros((len(clipped_times), 2), dtype=np.float32)
		self.data[:, 0] = clipped_times
		self.data[:, 1] = self.amplitude * np.sin(self.omega * clipped_times + self.phase)# + self.offset
		#sine_on_hz = np.power(2, sine + np.log2(2000))
		self.line_speed = scene.Line(pos=self.data, color=(0, 0, 1, .5), method='gl')
		self.initialize()
		
	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.show()
		self.vispy_canvas.regs.append(self)
		self.vispy_canvas.master_speed.update()
		
	def set_offset(self, a, b):
		#user manipulation: custom amplitude for sample
		self.amplitude *= (b/a)
		self.data[:, 1]*= (b/a)
		self.line_speed.set_data(pos=self.data)
		
	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		self.line_speed.set_data(color = (1, 1, 1, .5))
		
	def select(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = True
		self.line_speed.set_data(color = (0, 1, 0, 1))
		#set the offset in the ui
		self.vispy_canvas.props.phase_s.setValue(self.offset)
		
	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()
			
	def select_handle(self, multi=False):
		"""Toggle this line's selection state, and update the phase offset ui value"""
		if not multi:
			for trace in self.vispy_canvas.regs+self.vispy_canvas.lines:
				trace.deselect()
		self.toggle()
			
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		self.deselect()
		
	def remove(self):
		self.hide()
		#note: this has to search the list
		self.vispy_canvas.regs.remove(self)
		
		
	def update_phase(self, v):
		"""Adjust this regressions's phase offset according to the UI input."""
		if self.selected: self.offset = v
		
class TraceLine:
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, times, freqs, offset=None):
		
		self.selected = False
		
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
		self.line_spec.transform = mel_transform
		
		#create the speed curve visualization
		self.speed_data = np.ones((len(times), 2), dtype=np.float32)
		self.speed_data[:, 0] = self.times
		self.speed_data[:, 1] = self.speed
		self.line_speed = scene.Line(pos=self.speed_data, color=(1, 1, 1, .5), method='gl')
		self.initialize()

		
	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.show()
		self.line_spec.parent = self.vispy_canvas.spec_view.scene
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		self.vispy_canvas.lines.append(self)
		
	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		
	def hide(self):
		self.line_speed.parent = None
		self.deselect()
		
	def set_offset(self, a, b):
		offset = b-a
		#print("offset",offset)
		self.offset += offset
		self.speed_center[1] += offset
		self.speed += offset
		#print("new center",self.center)
		self.speed_data[:, 1] = self.speed
		self.line_speed.set_data(pos = self.speed_data)
	
	
	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		self.line_speed.set_data(color = (1, 1, 1, .5))
		self.line_spec.set_data(color = (1, 1, 1, 1))
		
	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.line_speed.set_data(color = (0, 1, 0, 1))
		self.line_spec.set_data(color = (0, 1, 0, 1))
		
	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()
		
	def select_handle(self, multi=False):
		if not multi:
			for trace in self.vispy_canvas.regs+self.vispy_canvas.lines:
				trace.deselect()
		self.toggle()
		
	def remove(self):
		self.line_speed.parent = None
		self.line_spec.parent = None
		#note: this has to search the list
		self.vispy_canvas.lines.remove(self)

class Canvas(scene.SceneCanvas):
	"""This class wraps the vispy canvas and controls all the visualization, as well as the interaction with it."""

	def __init__(self):
		
		#some default dummy values
		self.props = None
		self.filename = ""
		self.vmin = -80
		self.vmax = -40
		self.auto_align = True
		self.trace_mode = "Center of Gravity"
		self.adapt_mode = "Linear"
		self.rpm = "Unknown"
		self.show_regs = True
		self.show_lines = True
		
		self.last_click = None
		self.fft_size = 1024
		self.hop = 256
		self.sr = 44100
		self.num_ffts = 0
		
		scene.SceneCanvas.__init__(self, keys="interactive", size=(1024, 512), bgcolor="#353535")
		
		self.unfreeze()
		
		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0
		
		#speed chart
		self.speed_yaxis = scene.AxisWidget(orientation='left', axis_label='Octaves', axis_font_size=8, axis_label_margin=35, tick_label_margin=5)
		self.speed_yaxis.width_max = 55
		
		#spectrum
		self.spec_yaxis = vispy_ext.ExtAxisWidget(orientation='left', axis_label='Hz', axis_font_size=8, axis_label_margin=35, tick_label_margin=5, scale_type="logarithmic")
		self.spec_yaxis.width_max = 55
		
		self.spec_xaxis = scene.AxisWidget(orientation='bottom', axis_label='sec', axis_font_size=8, axis_label_margin=35, tick_label_margin=5)
		self.spec_xaxis.height_max = 55

		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 10
		
		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 70
		
		#create the color bar display
		self.colorbar_display = scene.ColorBarWidget(label="Gain [dB]", clim=(self.vmin, self.vmax), cmap="viridis", orientation="right", border_width=1, label_color="white")
		self.colorbar_display.label.font_size = 8
		self.colorbar_display.ticks[0].font_size = 8
		self.colorbar_display.ticks[1].font_size = 8
		
		grid.add_widget(self.speed_yaxis, row=1, col=0)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		colorbar_column = grid.add_widget(self.colorbar_display, row=2, col=2)
		
		self.speed_view = grid.add_view(row=1, col=1, border_color='white')
		self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, -0.1, 10, 0.2), )
		self.speed_view.height_min = 150
		self.spec_view = grid.add_view(row=2, col=1, border_color='white')
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		#link them, but use custom logic to only link the x view
		self.spec_view.camera.link(self.speed_view.camera)
		
		self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)
		
		self.lines = []
		self.regs = []
		self.fft_storage = {}
		
		self.master_speed = MasterSpeedLine(self)
		self.master_reg_speed = MasterRegLine(self)
		self.spectrum = Spectrum(self.spec_view)
		
		self.freeze()
		
	#fast stuff that does not require rebuilding everything
	def set_colormap(self, cmap):
		self.spectrum.set_cmap(cmap)
		self.colorbar_display.cmap = cmap
		
	#fast stuff that does not require rebuilding everything
	def set_clims(self, vmin, vmax):
		self.spectrum.set_clims(vmin, vmax)
		self.colorbar_display.clim = (vmin, vmax)
	
	#called if either  the file or FFT settings have changed
	def set_file_or_fft_settings(self, filename, fft_size = 256, fft_overlap = 1):
		if filename:
			soundob = sf.SoundFile(filename)
				
			#set this for the tracers etc.
			self.fft_size = fft_size
			self.hop = fft_size // fft_overlap
			self.sr = soundob.samplerate
			
			#store the FFTs for fast shuffling around
			#TODO: analyze the RAM consumption of this
			#TODO: perform FFT at minimal hop and get shorter hops via stride operation ::x
			k = (self.fft_size, self.hop)
			if k not in self.fft_storage:
				print("storing new fft",self.fft_size)
				signal = soundob.read(always_2d=True)[:,0]
				#this will automatically zero-pad the last fft
				#get the magnitude spectrum
				#avoid divide by 0 error in log10
				imdata = np.abs(fourier.stft(signal, self.fft_size, self.hop, "hann")+.0000001)
				#change to dB scale later, for the tracers
				#imdata = 20 * np.log10(imdata)
				#clamping the data to 0,1 range happens in the vertex shader
				
				#now store this for retrieval later
				self.fft_storage[k] = imdata.astype('float32')
			
			#retrieve the FFT data
			imdata = self.fft_storage[k]
			self.num_ffts = imdata.shape[1]

			#has the file changed?
			if self.filename != filename:
				print("file has changed!")
				self.filename = filename
				#(re)set the spec_view
				#only the camera dimension is mel'ed, as the image gets it from its transform
				self.speed_view.camera.rect = (0, -0.1, self.num_ffts * self.hop / self.sr, 0.2)
				self.spec_view.camera.rect = (0, 0, self.num_ffts * self.hop / self.sr, to_mel(self.sr//2))
			self.spectrum.update_data(imdata, self.hop, self.sr)
			self.set_clims(self.vmin, self.vmax)
			self.master_speed.update()
			self.master_reg_speed.update()
		
	def on_mouse_wheel(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		
		#colorbar scroll
		if self.click_on_widget(click, self.colorbar_display):
			y_pos = self.colorbar_display._colorbar.transform.imap(click)[1]
			d = int(event.delta[1])
			#now split Y in three parts
			lower = self.colorbar_display.size[1]/3
			upper = lower*2
			if y_pos < lower:
				self.vmax += d
			elif lower < y_pos < upper:
				self.vmin += d
				self.vmax -= d
			elif upper < y_pos:
				self.vmin += d
			self.set_clims(self.vmin, self.vmax)
				
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
		#ok, so left matches up with the axis! this means we need to transform mel's imap function
		#print((np.exp(self.spec_view.scene.transform.imap(self.spec_view.transform.imap(click)) / 1127) - 1) * 700, self.spec_view.scene.transform.imap(self.spec_view.transform.imap(click)), self.click_spec_conversion(click))
		if event.button == 1:
			if "Control" in event.modifiers:
				self.last_click = click
					
		#selection, single or multi
		if event.button == 2:
			closest_line = self.get_closest_line( click )
			if closest_line:
				if "Shift" in event.modifiers:
					closest_line.select_handle(multi=True)
					event.handled = True
				else:
					closest_line.select_handle()
					event.handled = True
	
	def on_mouse_move(self, event):
		#update the inspector label
		click = self.click_spec_conversion(event.pos)
		self.props.inspector_l.setText("\n        -.- Hz\n-:--:--:--- h:m:s:ms")
		if click is not None:
			t, f = click[0:2]
			if t >= 0 and  self.sr/2 > f >= 0:
				m, s = divmod(t, 60)
				s, ms = divmod(s*1000, 1000)
				h, m = divmod(m, 60)
				self.props.inspector_l.setText("\n    % 7.1f Hz\n%d:%02d:%02d:%03d h:m:s:ms" % (f, h, m, s, ms))
			
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
					if amplitude == 0:
						print("fallback")
						amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_reg_speed.get_linspace(), t0, t1, self.rpm)
					RegLine(self, t0, t1, amplitude, omega, phase, offset)
					self.master_reg_speed.update()
				else:
					if self.trace_mode == "Center of Gravity":
						times, freqs = wow_detection.trace_cog(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
					elif self.trace_mode == "Peak":
						times, freqs = wow_detection.trace_peak(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1, adaptation_mode = self.adapt_mode)
					elif self.trace_mode == "Correlation":
						times, freqs = wow_detection.trace_correlation(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
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
				for trace in self.lines+self.regs:
					if trace.selected:
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
				#actually, we don't need the euclidean distance here, just a relative distance metric, so we can avoid the sqrt and just take the squared distance
				#ind = np.linalg.norm(A-(c[0], c[1]), axis = 1).argmin()
				ind = np.sum((A-c[0:2])**2, axis = 1).argmin()
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
			if self.spectrum:
				#in fact the simple Y mel transform would be enough in any case
				#but this would also support other transforms
				return mel_transform.imap(scene_space)
		
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	dark_palette = QtGui.QPalette()
	WHITE =     QtGui.QColor(255, 255, 255)
	BLACK =     QtGui.QColor(0, 0, 0)
	RED =       QtGui.QColor(255, 0, 0)
	PRIMARY =   QtGui.QColor(53, 53, 53)
	SECONDARY = QtGui.QColor(35, 35, 35)
	TERTIARY =  QtGui.QColor(42, 130, 218)
	dark_palette.setColor(QtGui.QPalette.Window,          PRIMARY)
	dark_palette.setColor(QtGui.QPalette.WindowText,      WHITE)
	dark_palette.setColor(QtGui.QPalette.Base,            SECONDARY)
	dark_palette.setColor(QtGui.QPalette.AlternateBase,   PRIMARY)
	dark_palette.setColor(QtGui.QPalette.ToolTipBase,     WHITE)
	dark_palette.setColor(QtGui.QPalette.ToolTipText,     WHITE)
	dark_palette.setColor(QtGui.QPalette.Text,            WHITE)
	dark_palette.setColor(QtGui.QPalette.Button,          PRIMARY)
	dark_palette.setColor(QtGui.QPalette.ButtonText,      WHITE)
	dark_palette.setColor(QtGui.QPalette.BrightText,      RED)
	dark_palette.setColor(QtGui.QPalette.Link,            TERTIARY)
	dark_palette.setColor(QtGui.QPalette.Highlight,       TERTIARY)
	dark_palette.setColor(QtGui.QPalette.HighlightedText, BLACK)
	appQt.setPalette(dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = MainWindow()
	win.show()
	appQt.exec_()
