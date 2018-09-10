# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtCore, QtWidgets

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, qt_theme, snd

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
		
		tolerance_l = QtWidgets.QLabel("Tolerance")
		self.tolerance_s = QtWidgets.QDoubleSpinBox()
		self.tolerance_s.setRange(.01, 5)
		self.tolerance_s.setSingleStep(.05)
		self.tolerance_s.setValue(.1)
		self.tolerance_s.setToolTip("Intervall to consider in the trace, in semitones.")
		self.tolerance_s.valueChanged.connect(self.update_other_settings)
		
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
		self.mode_c.currentIndexChanged.connect(self.toggle_resampling_quality)
		self.sinc_quality_l = QtWidgets.QLabel("Quality")
		self.sinc_quality_s = QtWidgets.QSpinBox()
		self.sinc_quality_s.setRange(1, 100)
		self.sinc_quality_s.setSingleStep(1)
		self.sinc_quality_s.setValue(50)
		self.sinc_quality_s.setToolTip("Number of input samples that contribute to each output sample.\nMore samples = more quality, but slower. Only for sinc mode.")
		self.toggle_resampling_quality()
		
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
		
		self.audio_widget = snd.AudioWidget()
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(3)
		self.qgrid.setVerticalSpacing(0)
		
		buttons = [(display_l,), (fft_l, self.fft_c), (overlap_l, self.overlap_c), (show_l, self.show_c), (cmap_l,self.cmap_c), (tracing_l,), (trace_l, self.trace_c), (adapt_l, self.adapt_c), (rpm_l,self.rpm_c), (phase_l, self.phase_s), (tolerance_l, self.tolerance_s), (self.autoalign_b, ), (resampling_l, ), (mode_l, self.mode_c), (self.sinc_quality_l, self.sinc_quality_s), (self.scroll,), (self.progressBar,), (self.audio_widget,), (self.inspector_l,) ]
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
				self.parent.canvas.fft_storages = ({},)
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
		
	def merge_traces(self):
		#TODO: the offset handling is hacky here
		#it should be possible without the extra correction step in the end
	
		self.deltraces= []
		t0 = 999999
		t1 = 0
		offset = 0
		for trace in reversed(self.parent.canvas.lines):
			if trace.selected:
				self.deltraces.append(trace)
				t0 = min(t0, trace.speed_data[0, 0])
				t1 = max(t1, trace.speed_data[-1, 0])
				offset += trace.offset
		if self.deltraces:
			for trace in self.deltraces:
				trace.remove()
			offset /= len(self.deltraces)
			sr = self.parent.canvas.sr
			hop = self.parent.canvas.hop
			i0 = int(t0*sr/hop)
			i1 = int(t1*sr/hop)
			data = self.parent.canvas.master_speed.data[i0:i1]
			freqs = np.power(2, data[:,1]+10)
			y0 = self.parent.canvas.master_speed.data[i0,1]
			line = TraceLine(self.parent.canvas, data[:,0], freqs, offset)
			y1 = line.speed_data[0,1]
			d = y0-y1
			line.set_offset(0, d)
			self.parent.canvas.master_speed.update()
			
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
	
	def toggle_resampling_quality(self):
		b = (self.mode_c.currentText() == "Sinc")
		self.sinc_quality_l.setVisible(b)
		self.sinc_quality_s.setVisible(b)
		
	def update_other_settings(self):
		self.parent.canvas.trace_mode = self.trace_c.currentText()
		self.parent.canvas.tolerance = self.tolerance_s.value()
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
				self.resampling_thread.settings = ((self.filename,), speed_curve, self.mode_c.currentText(), self.sinc_quality_s.value(), channels)
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
		"""Toggle this line's selection state"""
		self.selected = True
		for v in self.visuals: v.set_data(color = self.color_sel)
		
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
		for v in self.visuals: v.parent = None
		#note: this has to search the list
		self.container.remove(self)
	
class MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)		
		
		self.resize(720, 400)
		self.setWindowTitle('pyrespeeder')
		try:
			scriptDir = os.path.dirname(os.path.realpath(__file__))
			self.setWindowIcon(QtGui.QIcon(os.path.join(scriptDir,'icons/pyrespeeder.png')))
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
						(editMenu, "Merge Selected", self.props.merge_traces, "CTRL+M"), \
						(editMenu, "Delete Selected", self.props.delete_traces, "DEL"), \
						(editMenu, "Play/Pause", self.props.audio_widget.play_pause, "SPACE"), \
						)
		
		for submenu, name, func, shortcut in button_data:
			button = QtWidgets.QAction(name, self)
			button.triggered.connect(func)
			if shortcut: button.setShortcut(shortcut)
			submenu.addAction(button)
		
	def update_settings_hard(self):
		self.canvas.set_file_or_fft_settings((self.props.filename,),
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
			self.data[:, 1] = mean_with_nans
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
		self.vispy_canvas.master_reg_speed.update()
		
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
		self.vispy_canvas.props.phase_s.setValue(self.offset)
		
	def update_phase(self, v):
		"""Adjust this regressions's phase offset according to the UI input."""
		if self.selected: self.offset = v
		
class TraceLine(BaseMarker):
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, times, freqs, offset=None):
		
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
			if not vispy_canvas.auto_align:
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
		self.vispy_canvas.master_speed.update()

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
		self.auto_align = True
		self.trace_mode = "Center of Gravity"
		self.adapt_mode = "Linear"
		self.rpm = "Unknown"
		self.tolerance = 5
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
			self.master_speed.update()
			self.master_reg_speed.update()
		
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
					#maybe query it here from the button instead of the other way
					if self.trace_mode == "Sine Regression":
						amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_speed.get_linspace(), t0, t1, self.rpm)
						if amplitude == 0:
							print("fallback")
							amplitude, omega, phase, offset = wow_detection.trace_sine_reg(self.master_reg_speed.get_linspace(), t0, t1, self.rpm)
						RegLine(self, t0, t1, amplitude, omega, phase, offset)
						self.master_reg_speed.update()
					else:
						if self.trace_mode in ("Center of Gravity", "Peak", "Correlation", "Freehand Draw"):
							times, freqs = wow_detection.trace_handle(self.trace_mode, self.fft_storages[0][(self.fft_size, self.hop)], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1, adaptation_mode = self.adapt_mode, tolerance = self.tolerance, trail = [self.click_spec_conversion(click) for click in event.trail()])
						if len(freqs) and np.nan not in freqs:
							TraceLine(self, times, freqs)
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
