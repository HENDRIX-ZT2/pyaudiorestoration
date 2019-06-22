import numpy as np
import soundfile as sf
import os
import scipy.signal
from PyQt5 import QtWidgets, QtGui, QtCore

from util import qt_theme, fourier, widgets, config, filters
	
def pairwise(iterable):
	it = iter(iterable)
	a = next(it, None)
	for b in it:
		yield (a, b)
		a = b

def to_dB(a):
	return 20 * np.log10(a)
	
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setAcceptDrops(True)
		
		self.cfg = config.read_config("config.ini")
		self.file_names = []
		self.names_to_full_paths = {}
		
		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)
		
		self.setWindowTitle('Dropouts')

		self.dropout_widget = widgets.DropoutWidget()
		self.display_widget = widgets.DisplayWidget()
		self.display_widget.fft_c.setCurrentIndex(3) # 512 smp
		
		
		self.b_add = QtWidgets.QPushButton('Load Audio')
		self.b_add.setToolTip("Load audio files you want to process.")
		self.b_add.clicked.connect(self.open_audio)
		
		self.b_remove = QtWidgets.QPushButton('Remove Audio')
		self.b_remove.setToolTip("Remove audio files you do not want to process.")
		self.b_remove.clicked.connect(self.remove_files)
		
		self.b_process = QtWidgets.QPushButton('Process')
		self.b_process.setToolTip("Process these files according to the current settings.")
		self.b_process.clicked.connect(self.process)
		
		self.file_widget = QtWidgets.QListWidget()
		self.file_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.b_add, 0, 0)
		self.qgrid.addWidget(self.b_remove, 0, 1)
		self.qgrid.addWidget(self.b_process, 0, 2)
		self.qgrid.addWidget(self.file_widget, 1, 0, 1, 3)
		self.qgrid.addWidget(self.dropout_widget, 2, 0, 1, 3)
		self.qgrid.addWidget(self.display_widget, 3, 0, 1, 3)
		
		self.central_widget.setLayout(self.qgrid)
		
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
				self.load_audio( str(url.toLocalFile()) )
		else:
			event.ignore()
			
	def open_audio(self):
		#just a wrapper around load_audio so we can access that via drag & drop and button
		#pyqt5 returns a tuple
		src_files = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Sources', self.cfg["dir_in"], "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		for audio_path in src_files:
			self.load_audio(audio_path)
			
	def load_audio(self, audio_path):
		#called whenever a potential audio file is added - via drag& drop or open_audio
		if audio_path:
			self.cfg["dir_in"], file_name = os.path.split(audio_path)
			if file_name not in self.file_names:
				self.file_names.append(file_name)
				self.file_widget.addItem(file_name)
				self.names_to_full_paths[file_name] = audio_path
		
	def remove_files(self):
		for item in self.file_widget.selectedItems():
			file_name = item.text()
			for i in reversed(range(0, len(self.file_names))):
				if self.file_names[i] == file_name:
					self.file_names.pop(i)
					break
			self.file_widget.takeItem(self.file_widget.row(item))

	def process(self):
		# get params from gui
		fft_size = self.display_widget.fft_size
		fft_overlap = self.display_widget.fft_overlap
		hop = fft_size // fft_overlap
		
		max_width = self.dropout_widget.max_width
		max_slope = self.dropout_widget.max_slope
		num_bands = self.dropout_widget.num_bands
		f_upper = self.dropout_widget.f_upper
		f_lower = self.dropout_widget.f_lower
		
		#split the range up into n bands
		bands = np.logspace(np.log2(f_lower), np.log2(f_upper), num=num_bands, endpoint=True, base=2, dtype=np.uint16)
		
		for file_name in self.file_names:
			print("Processing",file_name)
			# load audio
			file_path = self.names_to_full_paths[file_name]
			soundob = sf.SoundFile(file_path)
			sr = soundob.samplerate
			channels = soundob.channels
			signal = soundob.read(always_2d=True)
			
			# distance to look around current fft
			# divide by two because we are looking around the center
			d = int(max_width/1.5 * sr / hop )
			
			for channel in range(channels):
				print("Processing channel",channel)
				#which range should dropouts be detected in?
				imdata = fourier.stft(signal[:,channel], fft_size, hop, "hann")
				imdata = to_dB(imdata)
				#now what we generally don't want to do is "fix" dropouts of the lower bands only
				#basically, the gain of a band should be always controlled by that of the band above
				# only the top band acts freely
				
				# initialize correction
				correction_fac = np.ones( imdata.shape[1] ) * 1000
				
				# go over all bands
				for f_lower_band, f_upper_band in reversed(list(pairwise(bands))):
					
					# get the bin indices for this band
					bin_lower = int(f_lower_band * fft_size / sr)
					bin_upper = int(f_upper_band * fft_size / sr)
					
					# take the mean volume across this band
					vol = np.mean(imdata[bin_lower:bin_upper], axis=0)
				
					# detect valleys in the volume curve
					peaks, properties = scipy.signal.find_peaks(-vol, height=None, threshold=None, distance=None, prominence=5, wlen=None, rel_height=0.5, plateau_size=None)
					
					# initialize the gain curve for this band
					gain_curve = np.zeros( imdata.shape[1] )
					
					# go over all peak candidates and use good ones
					for peak_i in peaks:
						# avoid errors at the very ends
						if 2*d < peak_i < imdata.shape[1]-2*d-1:
							# make sure we are not blurring the left side of a transient
							# sample mean volume around +-d samples on either side of the potential dropout
							# patch_region = np.asarray( (peak_i-d, peak_i+d) )
							# patch_coords = vol[patch_region]
							left = np.mean(vol[peak_i-2*d:peak_i-d])
							right = np.mean(vol[peak_i+d:peak_i+2*d])
							m = (left-right) / (2*d)
							# only use it if slant is desirable
							#actually better make this abs() to avoid adding reverb
							# if not m < -.5:
							if abs(m) < max_slope:
								# now interpolate a new patch and get gain from difference to original volume curve
								gain_curve[peak_i-d:peak_i+d+1] = np.interp( range(2*d+1), (0, 2*d), (left, right) ) - vol[peak_i-d:peak_i+d+1]
					# gain_curve = gain_curve.clip(0)
					
					# we don't want to make pops more quiet, so clip at 1
					# clip the upper boundary according to band above (was processed before)
					# -> clip the factor to be between 1 and the factor of the band above (with some tolerance)
					correction_fac = np.clip(np.power(10, gain_curve/20), 1, correction_fac*2)
					# resample to match the signal
					vol_corr = signal[:,channel] * np.interp(np.linspace(0,1, len(signal[:,channel])), np.linspace(0,1, len(correction_fac)), correction_fac - 1)
					# add the extra bits to the signal
					signal[:,channel] += filters.butter_bandpass_filter(vol_corr, f_lower_band, f_upper_band, sr, order=3)
			
			# write the final signal
			with sf.SoundFile( os.path.splitext(file_path)[0]+"_out.wav", 'w+', sr, channels, subtype='FLOAT') as outfile:
				outfile.write(signal)
			print("Finished",file_path)

if __name__ == '__main__':
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	appQt.setPalette(qt_theme.dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = MainWindow()
	win.show()
	appQt.exec_()
config.write_config("config.ini", win.cfg)