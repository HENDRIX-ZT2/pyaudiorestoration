import os
import numpy as np
import vispy
import sys
from vispy import color
from PyQt5 import QtGui, QtCore, QtWidgets

from util import units, config, qt_theme, colormaps

myFont=QtGui.QFont()
myFont.setBold(True)


def print_version_info():
	print("Running...")
	print("Python:", sys.version)
	print("Numpy:", np.__version__)
	print("Vispy:", vispy.__version__)


def startup(cls):
	print_version_info()
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	appQt.setPalette(qt_theme.dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = cls()
	win.show()
	appQt.exec_()
	config.write_config("config.ini", win.cfg)

def abort_open_new_file(parent, newfile, oldfile):
	# only return True if we should abort
	if newfile == oldfile:
		return True
	if oldfile:
		qm = QtWidgets.QMessageBox
		return qm.No == qm.question(parent.parent,'', "Do you really want to load "+os.path.basename(newfile)+"? You will lose unsaved work on "+os.path.basename(oldfile)+"!", qm.Yes | qm.No)

def showdialog(str):
	msg = QtWidgets.QMessageBox()
	msg.setIcon(QtWidgets.QMessageBox.Information)
	msg.setText(str)
	#msg.setInformativeText("This is additional information")
	msg.setWindowTitle("Error")
	#msg.setDetailedText("The details are as follows:")
	msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
	retval = msg.exec_()

def grid(buttons):
	qgrid = QtWidgets.QGridLayout()
	qgrid.setHorizontalSpacing(3)
	qgrid.setVerticalSpacing(0)
	for i, line in enumerate(buttons):
		for j, element in enumerate(line):
			# we want to stretch that one
			if 1 == len(line):
				qgrid.addWidget(line[j], i, j, 1, 2)
			else:
				qgrid.addWidget(line[j], i, j)
	for i in range(2):
		qgrid.setColumnStretch(i, 1)
	return qgrid
	
def vbox(parent, grid):
	vbox = QtWidgets.QVBoxLayout(parent)
	vbox.addLayout(grid)
	vbox.addStretch(1.0)
	vbox.setSpacing(0)
	vbox.setContentsMargins(0,0,0,0)

def vbox2(parent, buttons):
	vbox = QtWidgets.QVBoxLayout(parent)
	for w in buttons: vbox.addWidget(w)
	vbox.addStretch(1.0)
	


class FileWidget(QtWidgets.QLineEdit):
	"""An entry widget that starts a file selector when clicked and also accepts drag & drop.
	Displays the current file's basename.
	"""

	def __init__(self, parent, cfg, description="", ask_user=True):
		super(FileWidget, self).__init__(parent)
		self.parent = parent
		self.cfg = cfg
		if not self.cfg:
			self.cfg["dir_in"]  = "C://"
		self.setDragEnabled(True)
		self.setReadOnly(True)
		self.filepath = ""
		self.description = description
		self.setToolTip(self.description)
		self.ask_user = ask_user
			
	def abort_open_new_file(self, new_filepath):
		# only return True if we should abort
		if not self.ask_user:
			return False
		if new_filepath == self.filepath:
			return True
		if self.filepath:
			qm = QtWidgets.QMessageBox
			return qm.No == qm.question(self,'', "Do you really want to load "+os.path.basename(new_filepath)+"? You will lose unsaved work on "+os.path.basename(self.filepath)+"!", qm.Yes | qm.No)
			
	def accept_file(self, filepath):
		if os.path.isfile(filepath):
			if os.path.splitext(filepath)[1].lower() in (".flac", ".wav"):
				if not self.abort_open_new_file(filepath):
					self.filepath = filepath
					self.cfg["dir_in"], filename = os.path.split(filepath)
					self.setText(filename)
					self.parent.poll()
			else:
				showdialog("Unsupported File Format")
				
	def get_files(self, event):
		data = event.mimeData()
		urls = data.urls()
		if urls and urls[0].scheme() == 'file':
			return urls
		
	def dragEnterEvent(self, event):
		if self.get_files(event):
			event.acceptProposedAction()
			self.setFocus(True)

	def dragMoveEvent(self, event):
		if self.get_files(event):
			event.acceptProposedAction()
			self.setFocus(True)

	def dropEvent(self, event):
		urls = self.get_files(event)
		if urls:
			filepath = str(urls[0].path())[1:]
			self.accept_file(filepath)
			
	def ask_open(self):
		filepath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open '+self.description, self.cfg["dir_in"], "Audio files (*.flac *.wav)")[0]
		self.accept_file(filepath)
		
	def mousePressEvent(self, event):
		self.ask_open()

class DisplayWidget(QtWidgets.QWidget):
	def __init__(self, with_canvas=True):
		QtWidgets.QWidget.__init__(self,)
		
		display_l = QtWidgets.QLabel("Display")
		display_l.setFont(myFont)
		
		fft_l = QtWidgets.QLabel("FFT Size")
		self.fft_c = QtWidgets.QComboBox(self)
		self.fft_c.addItems(("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072"))
		self.fft_c.setToolTip("This determines the frequency resolution.")
		self.fft_c.setCurrentIndex(5)

		self.clear_storage = QtWidgets.QPushButton("Clear Storage")

		overlap_l = QtWidgets.QLabel("FFT Overlap")
		self.overlap_c = QtWidgets.QComboBox(self)
		self.overlap_c.addItems(("1", "2", "4", "8", "16", "32"))
		self.overlap_c.setToolTip("Increase to improve temporal resolution.")
		self.overlap_c.setCurrentIndex(2)
		
		buttons = [(display_l,), (fft_l, self.fft_c), (overlap_l, self.overlap_c)]
		
		if with_canvas:
			show_l = QtWidgets.QLabel("Show")
			self.show_c = QtWidgets.QComboBox(self)
			self.show_c.addItems(("Both", "Traces", "Regressions"))
			
			cmap_l = QtWidgets.QLabel("Colors")
			self.cmap_c = QtWidgets.QComboBox(self)
			self.cmap_c.addItems(sorted(colormaps.cmaps.keys()))
			self.cmap_c.setCurrentText("izo")

			buttons.extend( ((show_l, self.show_c), (cmap_l,self.cmap_c)) )

		buttons.append((self.clear_storage,))
		vbox(self, grid(buttons))
		
		if with_canvas:
			# only connect in the end
			self.fft_c.currentIndexChanged.connect(self.update_fft_settings)
			self.overlap_c.currentIndexChanged.connect(self.update_fft_settings)
			self.show_c.currentIndexChanged.connect(self.update_show_settings)
			self.cmap_c.currentIndexChanged.connect(self.update_cmap)
			self.clear_storage.clicked.connect(self.force_clear_storage)

	@property
	def fft_size(self): return int(self.fft_c.currentText())
	
	@property
	def fft_overlap(self): return int(self.overlap_c.currentText())
	
	def update_fft_settings(self,):
		self.canvas.compute_spectra(self.canvas.filenames,
									fft_size = self.fft_size,
									fft_overlap = self.fft_overlap)
		
	def update_show_settings(self):
		show = self.show_c.currentText()
		if show == "Traces":
			self.canvas.show_regs = False
			self.canvas.show_lines = True
			self.canvas.master_speed.show()
			for trace in self.canvas.lines:
				trace.show()
			self.canvas.master_reg_speed.hide()
			for reg in self.canvas.regs:
				reg.hide()
		elif show == "Regressions":
			self.canvas.show_regs = True
			self.canvas.show_lines = False
			self.canvas.master_speed.hide()
			for trace in self.canvas.lines:
				trace.hide()
			self.canvas.master_reg_speed.show()
			for reg in self.canvas.regs:
				reg.show()
		elif show == "Both":
			self.canvas.show_regs = True
			self.canvas.show_lines = True
			self.canvas.master_speed.show()
			for trace in self.canvas.lines:
				trace.show()
			self.canvas.master_reg_speed.show()
			for reg in self.canvas.regs:
				reg.show()
				
	def update_cmap(self):
		self.canvas.set_colormap(self.cmap_c.currentText())	

	def force_clear_storage(self):
		self.canvas.clear_fft_storage()

class TracingWidget(QtWidgets.QWidget):
	def __init__(self,):
		QtWidgets.QWidget.__init__(self,)
		tracing_l = QtWidgets.QLabel("\nTracing")
		tracing_l.setFont(myFont)
		trace_l = QtWidgets.QLabel("Mode")
		self.trace_c = QtWidgets.QComboBox(self)
		self.trace_c.addItems(("Center of Gravity","Peak","Partials","Correlation","Freehand Draw", "Sine Regression"))
		self.trace_c.currentIndexChanged.connect(self.toggle_trace_mode)
		
		self.rpm_l = QtWidgets.QLabel("Source RPM")
		self.rpm_c = QtWidgets.QComboBox(self)
		self.rpm_c.setEditable(True)
		self.rpm_c.addItems(("Unknown","33.333","45","78"))
		self.rpm_c.setToolTip("This helps avoid bad values in the sine regression. \nIf you don't know the source, measure the duration of one wow cycle. \nRPM = 60/cycle length")
		
		self.phase_l = QtWidgets.QLabel("Phase Offset")
		self.phase_s = QtWidgets.QSpinBox()
		self.phase_s.setRange(-20, 20)
		self.phase_s.setSingleStep(1)
		self.phase_s.setValue(0)
		self.phase_s.valueChanged.connect(self.update_phase_offset)
		self.phase_s.setToolTip("Adjust the phase of the selected sine regression to match the surrounding regions.")
		
		tolerance_l = QtWidgets.QLabel("Tolerance")
		self.tolerance_s = QtWidgets.QDoubleSpinBox()
		self.tolerance_s.setRange(.01, 5)
		self.tolerance_s.setSingleStep(.05)
		self.tolerance_s.setValue(.5)
		self.tolerance_s.setToolTip("Intervall to consider in the trace, in semitones.")
		
		adapt_l = QtWidgets.QLabel("Adaptation")
		self.adapt_c = QtWidgets.QComboBox(self)
		self.adapt_c.addItems(("Average", "Linear", "Constant", "None"))
		self.adapt_c.setToolTip("Used to predict the next frequencies when tracing.")
		# might as well hide it until it is re-implemented
		adapt_l.setVisible(False)
		self.adapt_c.setVisible(False)
		
		band0_l = QtWidgets.QLabel("Highpass")
		self.band0_s = QtWidgets.QDoubleSpinBox()
		self.band0_s.setRange(0, 10000)
		self.band0_s.setSingleStep(.1)
		self.band0_s.setValue(0)
		self.band0_s.setToolTip("Cull wow below this frequency from the final speed curve.")
		self.band0_s.valueChanged.connect(self.update_bands)
		
		
		band1_l = QtWidgets.QLabel("Lowpass")
		self.band1_s = QtWidgets.QDoubleSpinBox()
		self.band1_s.setRange(.01, 10000)
		self.band1_s.setSingleStep(.1)
		self.band1_s.setValue(20)
		self.band1_s.setToolTip("Cull flutter above this frequency from the final speed curve.")
		self.band1_s.valueChanged.connect(self.update_bands)
		
		target_l = QtWidgets.QLabel("Target Frequency")
		self.target_s = QtWidgets.QDoubleSpinBox()
		self.target_s.setRange(0, 30000)
		self.target_s.setSingleStep(.1)
		self.target_s.setValue(0)
		self.target_s.setToolTip("The selected traces' mean frequency.")
		
		self.target_b = QtWidgets.QPushButton("Set Freq")
		self.target_b.clicked.connect(self.update_target)
		self.target_b.setToolTip("Set mean frequency to selected traces.")
		
		self.autoalign_b = QtWidgets.QCheckBox("Auto-Align")
		self.autoalign_b.setChecked(True)
		self.autoalign_b.setToolTip("Should new traces be aligned with existing ones?")
		
		buttons = ((tracing_l,), (trace_l, self.trace_c), (adapt_l, self.adapt_c), (self.rpm_l,self.rpm_c), (self.phase_l, self.phase_s), (tolerance_l, self.tolerance_s), (band0_l, self.band0_s), (band1_l, self.band1_s), (target_l, self.target_s), (self.target_b, ), (self.autoalign_b, ))
		vbox(self, grid(buttons))
		
		self.toggle_trace_mode()

	@property
	def mode(self): return self.trace_c.currentText()
	
	@property
	def tolerance(self): return self.tolerance_s.value()
	
	@property
	def adapt(self): return self.adapt_c.currentText()
	
	@property
	def auto_align(self): return self.autoalign_b.isChecked()
	
	@property
	def rpm(self): return self.rpm_c.currentText()
	
	def toggle_trace_mode(self):
		b = (self.trace_c.currentText() == "Sine Regression")
		self.rpm_l.setVisible(b)
		self.rpm_c.setVisible(b)
		self.phase_l.setVisible(b)
		self.phase_s.setVisible(b)
		
	def update_bands(self):
		self.canvas.master_speed.bands = (self.band0_s.value(), self.band1_s.value())
		self.canvas.master_speed.update()
		
	def update_phase_offset(self):
		v = self.phase_s.value()
		for reg in self.canvas.regs:
			reg.update_phase(v)
		self.canvas.master_reg_speed.update()
	
	def update_target(self):
		f = self.target_s.value()
		for reg in self.canvas.lines:
			reg.lock_to(f)
		self.canvas.master_speed.update()

class AlignmentWidget(QtWidgets.QWidget):
	def __init__(self,):
		QtWidgets.QWidget.__init__(self,)
		align_l = QtWidgets.QLabel("\nAlignment")
		align_l.setFont(myFont)
		
		self.align_abs_b = QtWidgets.QCheckBox("Absolute match")
		self.align_abs_b.setChecked(False)
		self.align_abs_b.setToolTip("Turn on if phase does not match.")
		
		buttons = ((align_l,), (self.align_abs_b, ), )
		vbox(self, grid(buttons))
	
	@property
	def align_abs(self): return self.align_abs_b.isChecked()

class DropoutWidget(QtWidgets.QWidget):
	def __init__(self, ):
		QtWidgets.QWidget.__init__(self,)
		
		dropouts_l = QtWidgets.QLabel("\nDropouts")
		dropouts_l.setFont(myFont)
		
		mode_l = QtWidgets.QLabel("Mode")
		self.mode_c = QtWidgets.QComboBox(self)
		self.mode_c.addItems(("Heuristic", "MaxMono"))
		self.mode_c.currentIndexChanged.connect(self.toggle_mode)
		
		self.num_bands_l = QtWidgets.QLabel("Bands")
		self.num_bands_s = QtWidgets.QSpinBox()
		self.num_bands_s.setRange(1, 6)
		self.num_bands_s.setSingleStep(1)
		self.num_bands_s.setValue(3)
		self.num_bands_s.setToolTip("Number of bands across which the intensity of a dropout is evaluated")
		
		self.f_lower_l = QtWidgets.QLabel("Lower")
		self.f_lower_s = QtWidgets.QSpinBox()
		self.f_lower_s.setRange(1, 20000)
		self.f_lower_s.setSingleStep(500)
		self.f_lower_s.setValue(3000)
		self.f_lower_s.setSuffix(" Hz")
		self.f_lower_s.setToolTip("Lower boundary frequency for dropout detection.")
		
		self.f_upper_l = QtWidgets.QLabel("Upper")
		self.f_upper_s = QtWidgets.QSpinBox()
		self.f_upper_s.setRange(1, 20000)
		self.f_upper_s.setSingleStep(500)
		self.f_upper_s.setValue(12000)
		self.f_upper_s.setSuffix(" Hz")
		self.f_upper_s.setToolTip("Upper boundary frequency for dropout detection.")
		
		self.max_slope_l = QtWidgets.QLabel("Max Slope")
		self.max_slope_s = QtWidgets.QDoubleSpinBox()
		self.max_slope_s.setRange(0.0, 10)
		self.max_slope_s.setSingleStep(.1)
		self.max_slope_s.setValue(0.5)
		self.max_slope_s.setSuffix(" dB")
		self.max_slope_s.setToolTip("Absolute slope between dB to the left and right of a dropout candidate.")
		
		self.max_width_l = QtWidgets.QLabel("Max Width")
		self.max_width_s = QtWidgets.QDoubleSpinBox()
		self.max_width_s.setRange(0.000001, 2)
		self.max_width_s.setSingleStep(.01)
		self.max_width_s.setValue(.02)
		self.max_width_s.setSuffix(" s")
		self.max_width_s.setToolTip("Maximum length of a dropout - increase to capture wider dropouts")
		
		self.bottom_freedom_l = QtWidgets.QLabel("Bottom Freedom")
		self.bottom_freedom_s = QtWidgets.QDoubleSpinBox()
		self.bottom_freedom_s.setRange(0.0000001, 5)
		self.bottom_freedom_s.setSingleStep(.1)
		self.bottom_freedom_s.setValue(2)
		self.bottom_freedom_s.setToolTip("Clips the band's factor to x*gain of the band above")
		
		buttons = ((dropouts_l,), (mode_l, self.mode_c,), (self.num_bands_l, self.num_bands_s), (self.f_upper_l, self.f_upper_s), (self.f_lower_l, self.f_lower_s), (self.max_slope_l, self.max_slope_s), (self.max_width_l, self.max_width_s), (self.bottom_freedom_l, self.bottom_freedom_s)	)
		vbox(self, grid(buttons))
		
	def toggle_mode(self):
		b = (self.mode_c.currentText() == "Heuristic")
		self.num_bands_l.setVisible(b)
		self.num_bands_s.setVisible(b)
		self.f_lower_l.setVisible(b)
		self.f_lower_s.setVisible(b)
		self.f_upper_l.setVisible(b)
		self.f_upper_s.setVisible(b)
		self.max_slope_l.setVisible(b)
		self.max_slope_s.setVisible(b)
		self.max_width_l.setVisible(b)
		self.max_width_s.setVisible(b)
		
	@property
	def mode(self, ): return self.mode_c.currentText()
	
	@property
	def f_lower(self): return self.f_lower_s.value()
	
	@property
	def f_upper(self): return self.f_upper_s.value()
	
	@property
	def num_bands(self): return self.num_bands_s.value()
	
	@property
	def max_slope(self): return self.max_slope_s.value()
	
	@property
	def max_width(self): return self.max_width_s.value()

	@property
	def bottom_freedom(self): return self.bottom_freedom_s.value()


class HPSSWidget(QtWidgets.QWidget):
	def __init__(self, ):
		QtWidgets.QWidget.__init__(self, )

		dropouts_l = QtWidgets.QLabel("\nHPSS")
		dropouts_l.setFont(myFont)

		self.h_kernel_l = QtWidgets.QLabel("Harmonic Kernel")
		self.h_kernel_s = QtWidgets.QSpinBox()
		self.h_kernel_s.setRange(1, 99)
		self.h_kernel_s.setSingleStep(1)
		self.h_kernel_s.setValue(31)
		self.h_kernel_s.setToolTip("Kernel size for the harmonic median filter.")
		
		self.p_kernel_l = QtWidgets.QLabel("Percussive Kernel")
		self.p_kernel_s = QtWidgets.QSpinBox()
		self.p_kernel_s.setRange(1, 99)
		self.p_kernel_s.setSingleStep(1)
		self.p_kernel_s.setValue(31)
		self.p_kernel_s.setToolTip("Kernel size for the percussive median filter.")

		self.power_l = QtWidgets.QLabel("Power")
		self.power_s = QtWidgets.QDoubleSpinBox()
		self.power_s.setRange(0.0, 10)
		self.power_s.setSingleStep(.1)
		self.power_s.setValue(2.0)
		self.power_s.setToolTip("Exponent for the Wiener filter when constructing soft mask matrices.")

		self.margin_l = QtWidgets.QLabel("Margin")
		self.margin_s = QtWidgets.QDoubleSpinBox()
		self.margin_s.setRange(0.000001, 99)
		self.margin_s.setSingleStep(.01)
		self.margin_s.setValue(1.0)
		self.margin_s.setToolTip("margin size(s) for the masks")

		buttons = (
		(dropouts_l,), (self.h_kernel_l, self.h_kernel_s), (self.p_kernel_l, self.p_kernel_s), (self.power_l, self.power_s), (self.margin_l, self.margin_s))
		vbox(self, grid(buttons))

	@property
	def h_kernel(self): return self.h_kernel_s.value()

	@property
	def p_kernel(self): return self.p_kernel_s.value()

	@property
	def power(self): return self.power_s.value()

	@property
	def margin(self): return self.margin_s.value()


class ResamplingWidget(QtWidgets.QWidget):
	def __init__(self, ):
		QtWidgets.QWidget.__init__(self,)
		
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
		
		
		self.mygroupbox = QtWidgets.QGroupBox('Channels')
		self.mygroupbox.setToolTip("Only selected channels will be resampled.")
		self.channel_layout = QtWidgets.QVBoxLayout()
		self.channel_layout.setSpacing(0)
		self.mygroupbox.setLayout(self.channel_layout)
		self.scroll = QtWidgets.QScrollArea()
		self.scroll.setWidget(self.mygroupbox)
		self.scroll.setWidgetResizable(True)
		self.channel_checkboxes = [ ]
		
		buttons = ((resampling_l,), (mode_l, self.mode_c,), (self.sinc_quality_l, self.sinc_quality_s), (self.scroll,))
		vbox(self, grid(buttons))
		
	def toggle_resampling_quality(self):
		b = (self.mode_c.currentText() == "Sinc")
		self.sinc_quality_l.setVisible(b)
		self.sinc_quality_s.setVisible(b)
		
	def refill(self, num_channels):
		for channel in self.channel_checkboxes:
			self.channel_layout.removeWidget(channel)
			channel.deleteLater()
		self.channel_checkboxes = []
		
		#fill the channel UI
		channel_names = ("Front Left", "Front Right", "Center", "LFE", "Back Left", "Back Right")
		for i in range(0, num_channels):
			name = channel_names[i] if i < 6 else str(i)
			self.channel_checkboxes.append(QtWidgets.QCheckBox(name))
			# set the startup option to just resample channel 0
			self.channel_checkboxes[-1].setChecked(True if i == 0 else False)
			self.channel_layout.addWidget( self.channel_checkboxes[-1] )
			
	@property
	def channels(self, ): return [i for i, channel in enumerate(self.channel_checkboxes) if channel.isChecked()]
	
	@property
	def sinc_quality(self, ): return self.sinc_quality_s.value()
	
	@property
	def mode(self, ): return self.mode_c.currentText()

class ProgressWidget(QtWidgets.QWidget):
	def __init__(self, ):
		QtWidgets.QWidget.__init__(self,)
	
		self.progressBar = QtWidgets.QProgressBar(self)
		self.progressBar.setRange(0,100)
		self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
		
		buttons = ((self.progressBar,),)
		vbox(self, grid(buttons))
		
	def onProgress(self, i):
		self.progressBar.setValue(i)


A4 = 440
C0 = A4*np.power(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	
def pitch(freq):
	try:
		h = round(12*np.log2(freq/C0))
		octave = int(h // 12)
		n = int(h % 12)
		if -1 < octave < 10:
			return note_names[n] + str(octave)
	except:
		pass
	return "-"

class InspectorWidget(QtWidgets.QLabel):
	def __init__(self, ):
		QtWidgets.QLabel.__init__(self, )
		self.def_text = "\n          - Note\n        -.- Hz\n-:--:--:--- h:m:s:ms"
		myFont2=QtGui.QFont("Monospace")
		myFont2.setStyleHint(QtGui.QFont.TypeWriter)
		self.setFont(myFont2)
		
	def update_text(self, click, sr):
		self.setText(self.def_text)
		if click is not None:
			t, f = click[0:2]
			if t >= 0 and  sr/2 > f >= 0:
				self.setText("\n%11s Note\n   % 8.1f Hz\n" % (pitch(f), f)+units.sec_to_timestamp(t))
				

class MainWindow(QtWidgets.QMainWindow):

	def __init__(self, name, object_widget, canvas_widget, count):
		QtWidgets.QMainWindow.__init__(self)		
		
		self.name = name
		self.resize(720, 400)
		self.setWindowTitle(name)
		try:
			base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
			self.setWindowIcon(QtGui.QIcon(os.path.join(base_dir,'icons/'+name+'.png')))
		except: pass
		
		self.cfg = config.read_config("config.ini")
		
		self.props = object_widget(parent=self, count=count)
		self.canvas = canvas_widget(parent=self)
		self.canvas.props = self.props
		self.props.file_widget.on_load_file = self.canvas.load_audio

		splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		splitter.addWidget(self.canvas.native)
		splitter.addWidget(self.props)
		self.setCentralWidget(splitter)

	def update_file(self, filepath):
		self.cfg["dir_in"], file_name = os.path.split(filepath)
		self.setWindowTitle(self.name+" "+ file_name)
		
	def add_to_menu(self, button_data):
		for submenu, name, func, shortcut in button_data:
			button = QtWidgets.QAction(name, self)
			button.triggered.connect(func)
			if shortcut: button.setShortcut(shortcut)
			submenu.addAction(button)
			
class FilesWidget(QtWidgets.QWidget):
	"""
	Holds several file widgets
	controls what happens when they are loaded
	"""

	def __init__(self, parent, count, cfg={}, ask_user=True):
		super(FilesWidget, self).__init__(parent)
		self.parent = parent
		#note: count must be 1 or 2
		# idiosyncratic order here so the complicated stuff can remain as is
		descriptions = ("Reference", "Source")
		self.files = [FileWidget(self, cfg, description, ask_user) for description in descriptions[-count:] ]
		vbox2(self, self.files)
		self.poll()
		
	def ask_open(self):
		# propagates the open action onto child widgets which then call a file selector
		for w in self.files:
			w.ask_open()
			
	def poll(self):
		# called by the child widgets after they have received a file
		self.filepaths = [ w.filepath for w in self.files]
		# only continue if all slots are filled with files
		if all(self.filepaths):
			self.load()
			
	def on_load_file(self, filepaths):
		print("No loading function defined!")
		
	def load(self):
		self.on_load_file(self.filepaths)
		
class ParamWidget(QtWidgets.QWidget):
	"""
	Widget for editing parameters
	"""

	def __init__(self, parent=None, count=1):
		super(ParamWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.file_widget = FilesWidget(self, count, self.parent.cfg)
		self.display_widget = DisplayWidget()
		self.tracing_widget = TracingWidget()
		self.resampling_widget = ResamplingWidget()
		self.progress_widget = ProgressWidget()
		#self.audio_widget = snd.AudioWidget()
		self.inspector_widget = InspectorWidget()
		self.alignment_widget = AlignmentWidget()
		buttons = [ self.file_widget, self.display_widget, self.tracing_widget, self.alignment_widget, self.resampling_widget, self.progress_widget,# self.audio_widget, 
					self.inspector_widget ]
		vbox2(self, buttons)
	
