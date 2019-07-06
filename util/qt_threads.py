from PyQt5 import QtCore
from util import resampling, io_ops, fourier

# nb. in QThreads, run() is called when start() is called on class

class BaseThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	
	def __init__(self, progress_callback=None):
		super().__init__()
		if progress_callback:
			self.notifyProgress.connect(progress_callback)

class ResamplingThread(BaseThread):

	def run(self):
		# name, speed_curve, resampling_mode, sinc_quality, use_channels = 
		resampling.run(prog_sig=self, **self.settings)
		
class FourierThread(BaseThread):
	jobs = []
	result = {}
	def run(self):
		for filename, channel, fft_size, hop, window, num_cores, key in self.jobs:
			signal, self.sr, self.channels = io_ops.read_file(filename)
			self.result[key] = fourier.stft(signal[:,channel], fft_size, hop, window, num_cores, prog_sig=self.notifyProgress)
		self.jobs = []
	