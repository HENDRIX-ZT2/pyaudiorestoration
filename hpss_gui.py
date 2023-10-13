import numpy as np
import os

from PyQt5 import QtWidgets, QtCore

from util import fourier, widgets, config, io_ops, decompose
	
def pairwise(iterable):
	it = iter(iterable)
	a = next(it, None)
	for b in it:
		yield (a, b)
		a = b
	
class MainWindow(QtWidgets.QMainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setAcceptDrops(True)
		
		self.cfg = config.load_config()
		self.file_names = []
		self.names_to_full_paths = {}
		
		self.central_widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.central_widget)
		
		self.setWindowTitle('Harmonic Percussive Separation')

		self.hpss_widget = widgets.HPSSWidget()
		self.display_widget = widgets.SpectrumSettingsWidget(with_canvas=False)
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
		
		self.files_widget = QtWidgets.QListWidget()
		self.files_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		
		self.qgrid = QtWidgets.QGridLayout()
		self.qgrid.setHorizontalSpacing(0)
		self.qgrid.setVerticalSpacing(0)
		self.qgrid.addWidget(self.b_add, 0, 0)
		self.qgrid.addWidget(self.b_remove, 0, 1)
		self.qgrid.addWidget(self.b_process, 0, 2)
		self.qgrid.addWidget(self.files_widget, 1, 0, 1, 3)
		self.qgrid.addWidget(self.hpss_widget, 2, 0, 1, 3)
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
			src_files = [str(url.toLocalFile()) for url in event.mimeData().urls()]
			for audio_path in sorted(src_files):
				self.load_audio(audio_path)
		else:
			event.ignore()
			
	def open_audio(self):
		#just a wrapper around load_audio so we can access that via drag & drop and button
		#pyqt5 returns a tuple
		src_files = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Sources', self.cfg.get("dir_in", "C:/"), "Audio files (*.flac *.wav *.ogg *.aiff)")[0]
		for audio_path in src_files:
			self.load_audio(audio_path)
			
	def load_audio(self, audio_path):
		# called whenever a potential audio file is added - via drag& drop or open_audio
		if audio_path:
			self.cfg["dir_in"], file_name = os.path.split(audio_path)
			if file_name not in self.file_names:
				self.file_names.append(file_name)
				self.files_widget.addItem(file_name)
				self.names_to_full_paths[file_name] = audio_path
		
	def remove_files(self):
		for item in self.files_widget.selectedItems():
			file_name = item.text()
			for i in reversed(range(0, len(self.file_names))):
				if self.file_names[i] == file_name:
					self.file_names.pop(i)
					break
			self.files_widget.takeItem(self.files_widget.row(item))

	def process(self):
		# get params from gui
		fft_size = self.display_widget.fft_size
		fft_overlap = self.display_widget.fft_overlap
		hop = fft_size // fft_overlap
		self.process_hpss(fft_size, hop)

	def process_hpss(self, fft_size, hop):
		for file_name in self.file_names:
			file_path = self.names_to_full_paths[file_name]
			signal, sr, channels = io_ops.read_file(file_path)

			kernel = (self.hpss_widget.h_kernel, self.hpss_widget.p_kernel)
			power = self.hpss_widget.power
			margin = self.hpss_widget.margin

			n = len(signal)
			# pad input stereo signal
			y_pad = fourier.fix_length(signal, n + fft_size // 2, axis=0)
			h_out = np.empty(signal.shape, signal.dtype)
			p_out = np.empty(signal.shape, signal.dtype)
			if margin != 1.0:
				print("has residue", margin)
				r_out = np.empty(signal.shape, signal.dtype)
			for channel in range(channels):
				print("channel",channel)
				# take FFT for each channel
				spectrum = fourier.stft(y_pad[:, channel], n_fft=fft_size, step=hop)

				harmonics, percussives = decompose.hpss(spectrum, kernel_size=kernel, power=power, margin=margin)

				# take iFFT
				h_out[:, channel] = fourier.istft(harmonics, length=n, hop_length=hop)
				p_out[:, channel] = fourier.istft(percussives, length=n, hop_length=hop)
				# implement residue: R = D - (H + P)
				if margin != 1.0:
					r_out[:, channel] = signal[:, channel] - (h_out[:, channel] + p_out[:, channel])
			io_ops.write_file(file_path, h_out, sr, channels, suffix="_H")
			io_ops.write_file(file_path, p_out, sr, channels, suffix="_P")
			if margin != 1.0:
				io_ops.write_file(file_path, r_out, sr, channels, suffix="_R")


if __name__ == '__main__':
	widgets.startup(MainWindow)