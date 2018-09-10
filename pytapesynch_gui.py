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
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, qt_theme, snd

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='band', output='sos')
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfiltfilt(sos, data)
	return y
	
def to_mel(val):
	### just to set the image size correctly	
	return np.log(val / 700 + 1) * 1127

def to_Hz(val):
	### just to set the image size correctly	
	return (np.exp(val / 1127) - 1) * 700
	
class ResamplingThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	def run(self):
		names, lag_curve, resampling_mode, sinc_quality, use_channels = self.settings
		resampling.run(names, lag_curve= lag_curve, resampling_mode = resampling_mode, sinc_quality=sinc_quality, use_channels=use_channels, prog_sig=self)
			
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""
	file_or_fft_settings_changed = QtCore.pyqtSignal(name='objectChanged')
	settings_soft_changed = QtCore.pyqtSignal(name='objectChanged2')

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.srcfilename = ""
		self.reffilename = ""
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
		
		self.audio_widget = snd.AudioWidget()
		
		buttons = [(display_l,), (fft_l, self.fft_c), (overlap_l, self.overlap_c), (resampling_l, ), (mode_l, self.mode_c), (self.sinc_quality_l, self.sinc_quality_s), (self.scroll,), (self.progressBar,), (self.audio_widget,), (self.inspector_l,), ]
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
		reffilename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Reference', 'c:\\', "Audio files (*.flac *.wav)")[0]
		srcfilename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Source', 'c:\\', "Audio files (*.flac *.wav)")[0]
		self.load_audio(reffilename, srcfilename)
			
	def load_audio(self, reffilename, srcfilename):

		# is the (dropped) file an audio file, ie. can it be read by pysoundfile?
		try:
			soundob = sf.SoundFile(reffilename)
			self.reffilename = reffilename
		except:
			print(reffilename+" could not be read, is it a valid audio file?")
			return
		try:
			soundob = sf.SoundFile(srcfilename)
			self.srcfilename = srcfilename
		except:
			print(srcfilename+" could not be read, is it a valid audio file?")
			return
				
		#Cleanup of old data
		self.parent.canvas.fft_storages = ({}, {})
		# self.parent.canvas.ref_fft_storage = {}
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
			# self.channel_checkboxes[-1].stateChanged.connect(self.update_other_settings)
			self.channel_layout.addWidget( self.channel_checkboxes[-1] )
		
		#finally - proceed with spectrum stuff elsewhere
		self.parent.setWindowTitle('pytapesynch '+os.path.basename(self.reffilename))
		self.file_or_fft_settings_changed.emit()
		data = resampling.read_lag(self.reffilename)
		for a0, a1, b0, b1, d in data:
			LagSample(self.parent.canvas, (a0, a1), (b0, b1), d)
		self.parent.canvas.lag_line.update()

	def save_traces(self):
		#get the data from the traces and regressions and save it
		resampling.write_lag(self.reffilename, [ (lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.d) for lag in self.parent.canvas.lag_samples ] )

	def improve_lag(self):
		for lag in self.parent.canvas.lag_samples:
			if lag.selected:
				sr = self.parent.canvas.sr
				raw_lag = lag.d*sr
				ref_t0 = int(sr*lag.a[0])
				ref_t1 = int(sr*lag.b[0])
				src_t0 = int(sr*lag.a[0]+lag.d)
				src_t1 = src_t0+ref_t1-ref_t0
				freqs = sorted((lag.a[1], lag.b[1]))
				lower = max(freqs[0], 1)
				upper = min(freqs[1], sr//2-1)
				# channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
				ref_ob = sf.SoundFile(self.reffilename)
				ref_sig = ref_ob.read(always_2d=True, dtype='float32')[ref_t0:ref_t1,0]
				src_ob = sf.SoundFile(self.srcfilename)
				src_sig = src_ob.read(always_2d=True, dtype='float32')[src_t0:src_t1,0]
				res = np.correlate(butter_bandpass_filter(ref_sig, lower, upper, sr, order=3), butter_bandpass_filter(src_sig, lower, upper, sr, order=3), mode="same")
				i_peak = np.argmax(res)
				#interpolate the most accurate fit
				result = wow_detection.parabolic(res, i_peak)[0] -(len(ref_sig)//2)
				lag.d = result/sr
				lag.select()
				self.parent.canvas.lag_line.update()
				print("raw accuracy (smp)",raw_lag)
				print("extra accuracy (smp)",result)
			
	def delete_traces(self, not_only_selected=False):
		self.deltraces= []
		for trace in reversed(self.parent.canvas.lag_samples):
			if (trace.selected and not not_only_selected) or not_only_selected:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.parent.canvas.lag_line.update()
		#this means a file was loaded, so clear the undo stack
		if not_only_selected:
			self.deltraces= []
			
	def toggle_resampling_quality(self):
		b = (self.mode_c.currentText() == "Sinc")
		self.sinc_quality_l.setVisible(b)
		self.sinc_quality_s.setVisible(b)
	
	def run_resample(self):
		if self.srcfilename and self.parent.canvas.lag_samples:
			channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
			if channels and self.parent.canvas.lag_samples:
				lag_curve = self.parent.canvas.lag_line.data
				self.resampling_thread.settings = ((self.srcfilename,), lag_curve, self.mode_c.currentText(), self.sinc_quality_s.value(), channels)
				self.resampling_thread.start()

	def run_resample_batch(self):
		if self.srcfilename and self.parent.canvas.lag_samples:
			reffilenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Files for Batch Resampling', 'c:\\', "Audio files (*.flac *.wav)")[0]
			channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
			if channels and self.parent.canvas.lag_samples:
				lag_curve = self.parent.canvas.lag_line.data
				self.resampling_thread.settings = (reffilenames, lag_curve, self.mode_c.currentText(), self.sinc_quality_s.value(), channels)
				self.resampling_thread.start()
				
	def update_param_hard(self, option):
		self.file_or_fft_settings_changed.emit()

		
class MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)		
		
		self.resize(720, 400)
		self.setWindowTitle('pytapesynch')
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
		# self.props.settings_soft_changed.connect(self.update_settings_soft)
		
		mainMenu = self.menuBar() 
		fileMenu = mainMenu.addMenu('File')
		editMenu = mainMenu.addMenu('Edit')
		#viewMenu = mainMenu.addMenu('View')
		#helpMenu = mainMenu.addMenu('Help')
		
		button_data = ( (fileMenu, "Open", self.props.open_audio, "CTRL+O"), \
						(fileMenu, "Save", self.props.save_traces, "CTRL+S"), \
						(fileMenu, "Resample", self.props.run_resample, "CTRL+R"), \
						(fileMenu, "Batch Resample", self.props.run_resample_batch, "CTRL+B"), \
						(fileMenu, "Exit", self.close, ""), \
						(editMenu, "Improve", self.props.improve_lag, "CTRL+I"), \
						# (editMenu, "Undo", self.props.restore_traces, "CTRL+Z"), \
						# (editMenu, "Redo", self.props.foo, "CTRL+Y"), \
						# (editMenu, "Select All", self.props.select_all, "CTRL+A"), \
						# (editMenu, "Invert Selection", self.props.invert_selection, "CTRL+I"), \
						(editMenu, "Delete Selected", self.props.delete_traces, "DEL"), \
						)
		
		for submenu, name, func, shortcut in button_data:
			button = QtWidgets.QAction(name, self)
			button.triggered.connect(func)
			if shortcut: button.setShortcut(shortcut)
			submenu.addAction(button)
		
	def update_settings_hard(self):
		self.canvas.set_file_or_fft_settings( (self.props.reffilename, self.props.srcfilename),
								 fft_size = int(self.props.fft_c.currentText()),
								 fft_overlap = int(self.props.overlap_c.currentText()))
		
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

class LagLine:
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
		if self.vispy_canvas.lag_samples:
		
			self.vispy_canvas.lag_samples.sort(key=lambda tup: tup.t)
			sample_times = [sample.t for sample in self.vispy_canvas.lag_samples]
			sample_lags = [sample.d for sample in self.vispy_canvas.lag_samples]
			
			num = self.vispy_canvas.num_ffts
			times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
			lag = np.interp(times, sample_times, sample_lags)
			
			#create the speed curve visualization, boost it a bit to distinguish from the raw curves
			self.data = np.zeros((len(times), 2), dtype=np.float32)
			self.data[:, 0] = times
			self.data[:, 1] = lag
		else:
			self.data = np.zeros((2, 2), dtype=np.float32)
			self.data[:, 0] = (0, 999)
			self.data[:, 1] = (0, 0)
		self.line_speed.set_data(pos=self.data)
		
		
class LagSample():
	"""Stores a single sinc regression's data and displays it"""
	def __init__(self, vispy_canvas, a, b, d=None):
		
		self.a = a
		self.b = b
		
		self.t = (a[0]+b[0])/2
		if d is None:
			self.d = vispy_canvas.spectra[-1].delta
		else:
			self.d = d
		self.width= abs(a[0]-b[0])
		self.f = (a[1]+b[1])/2
		self.height= abs(a[1]-b[1])
		self.spec_center = (self.t, self.f)
		self.rect = scene.Rectangle(center=(self.t, self.f), width=self.width, height=self.height, radius=0, parent=vispy_canvas.spec_view.scene)
		self.rect.color = (1, 1, 1, .5)
		self.rect.transform = vispy_canvas.spectra[-1].mel_transform
		self.rect.set_gl_state('additive')
		self.selected = False
		self.vispy_canvas = vispy_canvas
		self.initialize()
		
	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.vispy_canvas.lag_samples.append(self)
		self.vispy_canvas.lag_line.update()

	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		self.rect.color = (1, 1, 1, .5)
		
	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.rect.color = (0, 0, 1, .5)
		new_d = self.vispy_canvas.spectra[-1].delta
		self.vispy_canvas.spectra[-1].translate(self.d-new_d)
		
	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()
			
	def select_handle(self, multi=False):
		if not multi:
			for lag_sample in self.vispy_canvas.lag_samples:
				lag_sample.deselect()
		self.toggle()
	
	def show(self):
		self.rect.parent = self.vispy_canvas.spec_view.scene
		
	def hide(self):
		self.rect.parent = None
		self.deselect()
		
	def remove(self):
		self.hide()
		#note: this has to search the list
		self.vispy_canvas.lag_samples.remove(self)
		
class Canvas(spectrum.SpectrumCanvas):

	def __init__(self):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.unfreeze()
		self.lag_samples = []
		self.lag_line = LagLine(self)
		self.freeze()
		
	#called if either  the file or FFT settings have changed
	def set_file_or_fft_settings(self, files, fft_size = 256, fft_overlap = 1):
		if files:
			self.compute_spectra(files, fft_size, fft_overlap)
			self.lag_line.update()
		
	def on_mouse_press(self, event):
		#selection
		b = self.click_spec_conversion(event.pos)
		#are they in spec_view?
		if b is not None:
			self.props.audio_widget.cursor(b[0])
		if event.button == 2:
			closest_lag_sample = self.get_closest( self.lag_samples, event.pos )
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		if self.props.srcfilename and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				#are they in spec_view?
				if a is not None and b is not None:
					if "Control" in event.modifiers:
						d = b[0]-a[0]
						self.spectra[1].translate(d)
					elif "Shift" in event.modifiers:
						LagSample(self, a, b)
					elif "Alt" in event.modifiers:
						print()
						print("Start")
						#first get the time range for both
						#apply bandpass
						#split into pieces and look up the delay for each
						#correlate all the pieces
						sr = self.sr
						dur = int(0.2 *sr)
						times = sorted((a[0], b[0]))
						ref_t0 = int(sr*times[0])
						ref_t1 = int(sr*times[1])
						# src_t0 = int(sr*lag.a[0]+lag.d)
						# src_t1 = src_t0+ref_t1-ref_t0
						freqs = sorted((a[1], b[1]))
						lower = max(freqs[0], 1)
						upper = min(freqs[1], sr//2-1)
						# channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
						ref_ob = sf.SoundFile(self.props.reffilename)
						ref_sig = ref_ob.read(always_2d=True, dtype='float32')
						src_ob = sf.SoundFile(self.props.srcfilename)
						src_sig = src_ob.read(always_2d=True, dtype='float32')
						sample_times = np.arange(ref_t0, ref_t1, dur//32)
						data = self.lag_line.data
						sample_lags = np.interp(sample_times, data[:, 0]*sr, data[:, 1]*sr)
						
						#could do a stack
						out = np.zeros((len(sample_times), 2), dtype=np.float32)
						out[:, 0] = sample_times/sr
						for i, (x, d) in enumerate(zip(sample_times, sample_lags)):
							
							ref_s = butter_bandpass_filter(ref_sig[x:x+dur,0], lower, upper, sr, order=3)
							src_s = butter_bandpass_filter(src_sig[x-int(d):x-int(d)+dur,0], lower, upper, sr, order=3)
							res = np.correlate(ref_s*np.hanning(dur), src_s*np.hanning(dur), mode="same")
							i_peak = np.argmax(res)
							#interpolate the most accurate fit
							result = wow_detection.parabolic(res, i_peak)[0] -(len(ref_s)//2)
							# print("extra accuracy (smp)",int(d)+result)
							out[i, 1] = (int(d)+result)/sr
						self.lag_line.data =out
						self.lag_line.line_speed.set_data(pos=out)
							

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
