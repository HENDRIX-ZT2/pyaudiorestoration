import logging

from PyQt5 import QtCore
from util import resampling, fourier


# nb. in QThreads, run() is called when start() is called on class

class BaseThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)

	def __init__(self, progress_callback=None):
		super().__init__()
		if progress_callback:
			self.notifyProgress.connect(progress_callback)


class ResamplingThread(BaseThread):

	def run(self):
		resampling.run(prog_sig=self, **self.settings)


class FourierThread(BaseThread):
	jobs = []
	result = {}

	def run(self):
		for i, (signal_1d, fft_size, hop, window_name, key, filename) in enumerate(self.jobs):
			self.notifyProgress.emit(i/len(self.jobs)*100)
			self.result[filename, key] = fourier.get_mag(signal_1d, fft_size, hop, window_name)
		# logging.info("Cleared Fourier jobs")
		self.jobs.clear()
		self.notifyProgress.emit(100)
