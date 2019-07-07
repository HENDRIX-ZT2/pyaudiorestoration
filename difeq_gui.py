import numpy as np
import soundfile as sf
import xml.etree.ElementTree as ET
import os
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from util import qt_theme, fourier, io_ops, filters, widgets, units

# todo: make global sr set by the first file that is loaded, make all others fit

   
def spectrum_from_audio(filename, fft_size=4096, hop=256, channel_mode="L", start=None, end=None):
	print("reading",filename)
	signal, sr, channels = io_ops.read_file(filename)
	print(sr)
	spectra = []
	channel_map = {"L":(0,), "R":(1,), "L+R":(0,1)}
	for channel in channel_map[channel_mode]:
		print("channel",channel)
		if channel == channels:
			print("not enough channels for L/R comparison  - fallback to mono")
			break
		#get the magnitude spectrum
		#avoid divide by 0 error in log
		imdata = units.to_dB(fourier.stft(signal[:,channel], fft_size, hop, "hann"))
		spec = np.mean(imdata, axis=1)
		spectra.append(spec)
	#pad the data so we can compare this in a stereo setting if required
	if len(spectra) < 2:
		spectra.append(spectra[0])
	# return np.mean(spectra, axis=0), sr
	return spectra, sr

def indent(e, level=0):
	i = "\n" + level*"	"
	if len(e):
		if not e.text or not e.text.strip(): e.text = i + "	"
		if not e.tail or not e.tail.strip(): e.tail = i
		for e in e: indent(e, level+1)
		if not e.tail or not e.tail.strip(): e.tail = i
	else:
		if level and (not e.tail or not e.tail.strip()): e.tail = i
		
def write_eq(file_path, freqs, dB):
		tree=ET.ElementTree()
		equalizationeffect = ET.Element('equalizationeffect')
		curve=ET.SubElement(equalizationeffect, 'curve')
		curve.attrib["name"] = os.path.basename(file_path)[:-4]
		for f,d in zip(freqs,dB):
			point=ET.SubElement(curve, 'point')
			point.attrib["f"] = str(f)
			point.attrib["d"] = str(d)
		tree._setroot(equalizationeffect)
		indent(equalizationeffect)
		tree.write(file_path)
		
def get_eq(file_src, file_ref, channel_mode):
	print("Comparing channels:",channel_mode)
	#get the averaged spectrum for this audio file
	fft_size=16384
	hop=8192
	#todo: set custom times for both, if given
	spectra_src, sr_src = spectrum_from_audio(file_src, fft_size, hop, channel_mode)
	spectra_ref, sr_ref = spectrum_from_audio(file_ref, fft_size, hop, channel_mode)

	freqs = fourier.fft_freqs(fft_size, sr_src)
	#resample the ref spectrum to match the source
	if sr_src != sr_ref:
		spectra_ref = np.interp(freqs, fourier.fft_freqs(fft_size, sr_ref), spectra_ref)
	return freqs, np.asarray(spectra_ref)-np.asarray(spectra_src)
	
	
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		
		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)
		
		self.setWindowTitle('Differential EQ')
		self.src_dir = "C:\\"
		self.ref_dir = "C:\\"
		self.out_dir = "C:\\"
		self.names = []
		self.src_noise = None
		self.ref_noise = None
		self.eq_noise = None
		self.freqs = None
		self.eqs = []
		self.av = []
		self.freqs_av = []

		# a figure instance to plot on
		self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
		
		# the range is not automatically fixed
		# self.fig.patch.set_facecolor(SECONDARY.getRgb())
		# self.ax.set_facecolor(SECONDARY.getRgb())
		self.fig.patch.set_facecolor((53/255, 53/255, 53/255))
		self.ax.set_facecolor((35/255, 35/255, 35/255))
		# this is the Canvas Widget that displays the `figure`
		# it takes the `fig` instance as a parameter to __init__
		self.canvas = FigureCanvas(self.fig)

		# this is the Navigation widget
		# it takes the Canvas widget and a parent
		self.toolbar = NavigationToolbar(self.canvas, self)

		# Just some button connected to `plot` method
		self.b_add = QtWidgets.QPushButton('+')
		self.b_add.setToolTip("Add a source - reference pair to the list.")
		self.b_add.clicked.connect(self.add)
		self.b_delete = QtWidgets.QPushButton('-')
		self.b_delete.setToolTip("Delete the selected source - reference pair from the list.")
		self.b_delete.clicked.connect(self.delete)
		self.b_save = QtWidgets.QPushButton('=')
		self.b_save.setToolTip("Write the average EQ curve to an XML file.")
		self.b_save.clicked.connect(self.write)
		self.s_rolloff_start = QtWidgets.QSpinBox()
		self.s_rolloff_start.valueChanged.connect(self.plot)
		self.s_rolloff_start.setRange(0, 22000)
		self.s_rolloff_start.setSingleStep(1000)
		self.s_rolloff_start.setValue(21000)
		self.s_rolloff_start.setToolTip("At this frequency, the EQ still has full influence.")
		self.s_rolloff_end = QtWidgets.QSpinBox()
		self.s_rolloff_end.valueChanged.connect(self.plot)
		self.s_rolloff_end.setRange(1000, 22000)
		self.s_rolloff_end.setSingleStep(1000)
		self.s_rolloff_end.setValue(22000)
		self.s_rolloff_end.setToolTip("At this frequency, the effect of the EQ becomes zero.")
		self.c_channels = QtWidgets.QComboBox(self)
		self.c_channels.addItems(list(("L+R","L","R")))
		self.c_channels.setToolTip("Which channels should be analyzed?")
		self.s_output_res = QtWidgets.QSpinBox()
		self.s_output_res.valueChanged.connect(self.plot)
		self.s_output_res.setRange(20, 2000)
		self.s_output_res.setSingleStep(100)
		self.s_output_res.setValue(200)
		self.s_output_res.setToolTip("Resolution of the output curve.")
		self.s_smoothing = QtWidgets.QSpinBox()
		self.s_smoothing.valueChanged.connect(self.plot)
		self.s_smoothing.setRange(1, 200)
		self.s_smoothing.setSingleStep(10)
		self.s_smoothing.setValue(50)
		self.s_smoothing.setToolTip("Smoothing factor. Hint: Increase this if your sample size is small.")
		
		self.s_strength = QtWidgets.QSpinBox()
		self.s_strength.valueChanged.connect(self.plot)
		self.s_strength.setRange(10, 150)
		self.s_strength.setSingleStep(10)
		self.s_strength.setValue(100)
		self.s_strength.setToolTip("EQ Gain [%]. Adjust the strength of the output EQ curve.")
		
		self.b_noise = QtWidgets.QPushButton('Noise Floor')
		self.b_noise.setToolTip("Load a source - reference pair of noise floor samples.")
		self.b_noise.clicked.connect(self.add_noise)

		self.listWidget = QtWidgets.QListWidget()
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.toolbar, 0, 0, 1, 2)
		self.qgrid.addWidget(self.canvas, 1, 0, 1, 2)
		self.qgrid.addWidget(self.listWidget, 2, 0, 8, 1)
		self.qgrid.addWidget(self.b_add, 2, 1)
		self.qgrid.addWidget(self.b_delete, 3, 1)
		self.qgrid.addWidget(self.b_save, 4, 1)
		self.qgrid.addWidget(self.s_rolloff_start, 5, 1)
		self.qgrid.addWidget(self.s_rolloff_end, 6, 1)
		self.qgrid.addWidget(self.c_channels, 7, 1)
		self.qgrid.addWidget(self.s_output_res, 8, 1)
		self.qgrid.addWidget(self.s_smoothing, 9, 1)
		self.qgrid.addWidget(self.s_strength, 10, 1)
		self.qgrid.addWidget(self.b_noise, 11, 1)
		
		self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		self.central_widget.setLayout(self.qgrid)
		
		
	def add(self):
		file_src = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Source', self.src_dir, "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		if file_src:
			self.src_dir, src_name = os.path.split(file_src)
			file_ref = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Reference', self.ref_dir, "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
			if file_ref:
				channel_mode = self.c_channels.currentText()
				self.ref_dir, ref_name = os.path.split(file_ref)
				eq_name = src_name +" ("+channel_mode+") -> " + ref_name+" ("+channel_mode+")"
				self.freqs, eq = get_eq(file_src, file_ref, channel_mode)
				self.listWidget.addItem(eq_name)
				self.names.append(eq_name)
				self.eqs.append( eq )
				self.update_color(eq_name)
				self.plot()
				
	def add_noise(self):
		file_src = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Source', self.src_dir, "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		if file_src:
			file_ref = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Reference', self.ref_dir, "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
			if file_ref:
				channel_mode = self.c_channels.currentText()
				self.freqs, self.eq_noise = get_eq(file_src, file_ref, channel_mode)
				# self.listWidget.addItem(eq_name)
				# self.names.append(eq_name)
				# self.eqs.append( eq )
				# self.update_color(eq_name)
				self.plot()
			
	def update_color(self, eq_name):
		item = self.listWidget.findItems(eq_name, QtCore.Qt.MatchFixedString)[-1]
		#don't include the first (blue) -> reserved for the bold line
		item.setForeground( QtGui.QColor(self.colors[self.names.index(eq_name)+1]) )
		
	def delete(self):
		for item in self.listWidget.selectedItems():
			for i in reversed(range(0, len(self.names))):
				if self.names[i] == item.text():
					self.names.pop(i)
					self.eqs.pop(i)
			self.listWidget.takeItem(self.listWidget.row(item))
		
		for eq_name in self.names:
			self.update_color(eq_name)
		self.plot()
		
	def write(self):
		file_out = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Average EQ', self.out_dir, "XML files (*.xml)")[0]
		file_base = ".".join(file_out.split(".")[:-1])
		if file_out:
			try:
				self.out_dir, eq_name = os.path.split(file_out)
				write_eq(file_base+"_AV.xml", self.freqs_av, np.mean(self.av, axis=0))
				write_eq(file_base+"_L.xml", self.freqs_av, self.av[0])
				write_eq(file_base+"_R.xml", self.freqs_av, self.av[1])
			except PermissionError:
				widgets.showdialog("Could not write files - do you have writing permissions there?")
	
	def plot(self):

		# discards the old graph
		self.ax.clear()
		if self.freqs is not None:
			# todo: just calculate it from SR and bin count
			#again, just show from 20Hz
			from20Hz = (np.abs(self.freqs-20)).argmin()
		
		if self.names:
			num_in = 2000
			#average over n samples, then reduce the step according to the desired output
			n = self.s_smoothing.value()
			num_out = self.s_output_res.value()
			reduction_step = num_in // num_out
			#take the average curve of all differential EQs
			av_in = np.mean( np.asarray(self.eqs), axis=0)
			rolloff_start = self.s_rolloff_start.value()
			rolloff_end = self.s_rolloff_end.value()
			
			#audacity EQ starts at 20Hz
			freqs_spaced = np.power(2, np.linspace(np.log2(20), np.log2(self.freqs[-1]), num=num_in))
			
			avs = []
			#smoothen the curves, and reduce the points with step indexing
			self.freqs_av = filters.moving_average(freqs_spaced, n=n)[::reduction_step]
			for channel in (0,1):
				#interpolate this channel's EQ, then smoothen and reduce keys for this channel
				avs.append( filters.moving_average(np.interp(freqs_spaced, self.freqs, av_in[channel]), n=n)[::reduction_step] )
			self.av = np.asarray(avs)
			
			#get the gain of the filtered  EQ
			idx1 = np.abs(self.freqs_av-70).argmin()
			idx2 = np.abs(self.freqs_av-rolloff_end).argmin()
			gain = np.mean(self.av[:,idx1:idx2])
			strength = self.s_strength.value() / 100
			self.av -= gain
			self.av *= strength
			
			#fade out
			for channel in (0,1):
				self.av[channel] *= np.interp(self.freqs_av, (rolloff_start, rolloff_end), (1, 0) )
				
			#plot the contributing raw curves
			for name, eq in zip(self.names, np.mean(np.asarray(self.eqs), axis=1)):
				self.ax.semilogx(self.freqs[from20Hz:], eq[from20Hz:], basex=2, linestyle="--", linewidth=.5, alpha=.5, color=self.colors[self.names.index(name)+1])
			#take the average
			self.ax.semilogx(self.freqs_av, np.mean(self.av, axis=0), basex=2, linewidth=2.5, alpha=1, color= self.colors[0])
		if self.eq_noise is not None:
			self.ax.semilogx(self.freqs[from20Hz:], np.mean(self.eq_noise, axis=0)[from20Hz:], basex=2, linestyle="-.", linewidth=.5, alpha=.5, color="white")
		# refresh canvas
		self.canvas.draw()


if __name__ == '__main__':
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	appQt.setPalette(qt_theme.dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = MainWindow()
	win.show()
	appQt.exec_()