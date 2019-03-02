import numpy as np
import soundfile as sf
import os
import sys
import resampy
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from util import qt_theme, fourier

def spectrum_from_audio(filename, fft_size=4096, hop=256, channel_mode="L", start=None, end=None):
	print("reading",filename)
	soundob = sf.SoundFile(filename)
	sig = soundob.read(always_2d=True)
	sr = soundob.samplerate
	num_channels = sig.shape[1]
	spectra = []
	channels = {"L":(0,), "R":(1,), "L+R":(0,1)}
	for channel in channels[channel_mode]:
		print("channel",channel)
		if channel == num_channels:
			print("not enough channels for L/R comparison  - fallback to mono")
			break
		signal = sig[:,channel]
		
		#get the magnitude spectrum
		#avoid divide by 0 error in log
		imdata = 20 * np.log10(fourier.stft(signal, fft_size, hop, "hann"))
		spec = np.mean(imdata, axis=1)
		spectra.append(spec)
	#pad the data so we can compare this in a stereo setting if required
	if len(spectra) < 2:
		spectra.append(spectra[0])
	# return np.mean(spectra, axis=0), sr
	return spectra, sr
	
def parabolic(f, x):
	"""Helper function to refine a peak position in an array"""
	xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
	yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
	return (xv, yv)
	
def get_spectrum(file_src, channel_mode, fft_size):
	print("Comparing channels:",channel_mode)
	#get the averaged spectrum for this audio file
	hop = fft_size*2
	spectra_src, sr_src = spectrum_from_audio(file_src, fft_size, hop, channel_mode)

	freqs = fourier.fft_freqs(fft_size, sr_src)
	return freqs, np.mean(spectra_src, axis=0), sr_src
	
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		
		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)
		
		self.setWindowTitle('Hum Speed Analysis')
		self.file_src = ""
		self.freqs = None
		self.spectrum = None
		self.fft_size = 131072
		self.marker_freqs = []
		self.marker_dBs = []
		self.percentages = []
		self.hum_freqs = []
		
		self.cb = QtWidgets.QApplication.clipboard()
			
		# a figure instance to plot on
		self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
		
		self.ax.set_xlabel('Frequency (Hz)')
		self.ax.set_ylabel('Volume (dB)')
		
		# the range is not automatically fixed
		# self.fig.patch.set_facecolor(SECONDARY.getRgb())
		# self.ax.set_facecolor(SECONDARY.getRgb())
		self.fig.patch.set_facecolor((53/255, 53/255, 53/255))
		self.ax.set_facecolor((35/255, 35/255, 35/255))
		# this is the Canvas Widget that displays the `figure`
		# it takes the `fig` instance as a parameter to __init__
		self.canvas = FigureCanvas(self.fig)
		self.canvas.mpl_connect('button_press_event', self.onclick)
		
		# this is the Navigation widget
		# it takes the Canvas widget and a parent
		self.toolbar = NavigationToolbar(self.canvas, self)

		# Just some button connected to `plot` method
		self.b_open = QtWidgets.QPushButton('Open')
		self.b_open.setToolTip("Load an audio source to plot.")
		self.b_open.clicked.connect(self.open_file)
		
		self.b_resample = QtWidgets.QPushButton('Resample')
		self.b_resample.setToolTip("Write speed-corrected audio to a new file.")
		self.b_resample.clicked.connect(self.resample)
		
		self.c_channels = QtWidgets.QComboBox(self)
		self.c_channels.addItems(list(("L+R","L","R")))
		self.c_channels.setToolTip("Which channels should be analyzed?")
		self.s_base_hum = QtWidgets.QSpinBox()
		self.s_base_hum.setRange(10, 500)
		self.s_base_hum.setSingleStep(10)
		self.s_base_hum.setValue(50)
		self.s_base_hum.setSuffix(" Hz")
		self.s_base_hum.setToolTip("Base frequency of the hum.")

		self.s_num_harmonies = QtWidgets.QSpinBox()
		self.s_num_harmonies.setRange(0, 5)
		self.s_num_harmonies.setSingleStep(1)
		self.s_num_harmonies.setValue(2)
		self.s_num_harmonies.setToolTip("Number of hum harmonies to consider.")
		
		self.s_tolerance = QtWidgets.QSpinBox()
		self.s_tolerance.setRange(1, 20)
		self.s_tolerance.setSingleStep(1)
		self.s_tolerance.setValue(8)
		self.s_tolerance.setSuffix(" %")
		self.s_tolerance.setToolTip("Maximum derivation from target hum frequency to look for peaks in spectrum.")
		
		self.l_result = QtWidgets.QLabel("Result: ")
		
		self.qgrid = QtWidgets.QGridLayout()
		# self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.toolbar, 0, 0, 1, 7)
		self.qgrid.addWidget(self.canvas, 1, 0, 1, 7)
		self.qgrid.addWidget(self.b_open, 2, 0)
		self.qgrid.addWidget(self.b_resample, 2, 1)
		self.qgrid.addWidget(self.c_channels, 2, 2)
		self.qgrid.addWidget(self.s_base_hum, 2, 3)
		self.qgrid.addWidget(self.s_num_harmonies, 2, 4)
		self.qgrid.addWidget(self.s_tolerance, 2, 5)
		self.qgrid.addWidget(self.l_result, 2, 6)
		
		self.central_widget.setLayout(self.qgrid)
		
		self.s_base_hum.valueChanged.connect(self.on_hum_param_changed)
		self.s_num_harmonies.valueChanged.connect(self.on_hum_param_changed)
		self.s_tolerance.valueChanged.connect(self.on_hum_param_changed)
		
	def on_hum_param_changed(self,):
		
		base_hum = self.s_base_hum.value()
		num_harmonies = self.s_num_harmonies.value()
		self.hum_freqs = np.arange(base_hum, base_hum+base_hum*num_harmonies+1, base_hum)
		self.marker_freqs = []
		self.marker_dBs = []
		self.percentages = []
		for hum_freq in self.hum_freqs:
			self.track_to(hum_freq)
			self.plot()
		
	def open_file(self):
		file_src = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Source', os.path.dirname(self.file_src), "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		if file_src:
			self.file_src = file_src
			channel_mode = self.c_channels.currentText()
			self.freqs, self.spectrum, self.sr = get_spectrum(file_src, channel_mode, self.fft_size)
			self.on_hum_param_changed()
			self.plot()
	
	def onclick(self, event):
		if event.xdata and event.ydata:
			#right click
			if event.button == 3:
				self.marker_freqs = []
				self.marker_dBs = []
				self.percentages = []
				self.track_to(event.xdata)
				self.plot()
					
	def track_to(self, xpos):
		if self.freqs is not None:
			# a constant around the click; maybe make this into a log2-based value?
			WIDTH = self.s_tolerance.value()
			
			l_ratio = 1-WIDTH/100
			r_ratio = 1+WIDTH/100
			# print(l_ratio,r_ratio)
			# constant highest frequency for hum harmonies
			# MAX_FREQ = 500
			
			#get the closest index to the click's x position in the frequency array
			
			border_L = max(np.argmin(np.abs(self.freqs-xpos*l_ratio)), 0)
			border_R = min(np.argmin(np.abs(self.freqs-xpos*r_ratio)), len(self.freqs))
			
			# get peak from the selected region 
			raw_index = np.argmax(self.spectrum[border_L:border_R]) + border_L
			interp_index, dB = parabolic(self.spectrum, raw_index)
			
			# convert to frequency
			freq = interp_index * self.sr / self.fft_size
			
			#store coordinate
			self.marker_freqs.append( freq )
			self.marker_dBs.append( dB )
			
			# # get base hum frequency
			# base_hum = self.s_base_hum.value()
			
			# # get harmonies
			# hum_freqs = np.arange(base_hum, MAX_FREQ, base_hum)
			
			# find closest index and hum freq
			closest_index = np.argmin(np.abs(self.hum_freqs-freq))
			closest_hum = self.hum_freqs[closest_index]
			
			# # todo: tolerance should be a lot smaller
			# # is it close enough?
			# if abs(closest_hum-freq) > base_hum:
				# self.l_result.setText("hum was not close enough")
				# return
			# print("good", closest_hum, freq)
			# get percentage
			percent = ((closest_hum / freq) -1) * 100
			self.percentages.append(closest_hum / freq)
			# format as string
			percent_str = "%.3f" % percent
			
			# set to debug label
			self.l_result.setText("Percent Change: " + percent_str)
			
			# copy to clipboard
			self.cb.clear(mode=self.cb.Clipboard )
			self.cb.setText(percent_str, mode=self.cb.Clipboard)
	
	def resample(self,):
		if self.file_src and self.percentages:
			print("Resampling...")
			# get input
			ratio = self.percentages[-1]
			soundob = sf.SoundFile(self.file_src)
			sig = soundob.read(always_2d=True)
			sr = soundob.samplerate
			
			# resample, first axis is time!
			res = resampy.resample(sig, sr*ratio, sr, axis=0, filter='sinc_window', num_zeros=8)
			
			print("Saving...")
			# store output
			outfilename = self.file_src.rsplit('.', 1)[0]+'_resampled.wav'
			with sf.SoundFile(outfilename, 'w+', sr, soundob.channels, subtype='FLOAT') as outfile:
				outfile.write( res )
			print("Done!")
			
	def plot(self):
		# discards the old graph
		self.ax.clear()
		self.ax.set_xlabel('Frequency (Hz)')
		self.ax.set_ylabel('Volume (dB)')
		if self.freqs is not None:
			#take the average
			# self.ax.semilogx(self.freqs, self.spectrum, basex=2, linewidth=0.5, alpha=0.5)
			# self.ax.semilogx(self.marker_freqs, self.marker_dBs, 'b+', ms=14)
			end = np.argmin(np.abs(self.freqs-500))
			self.ax.plot(self.freqs[:end], self.spectrum[:end], linewidth=0.5, alpha=0.5)
			self.ax.plot(self.marker_freqs, self.marker_dBs, 'b+', ms=14)
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