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
		for signal_1d, fft_size, hop, window_name, num_cores, key, filename in self.jobs:
			self.result[filename, key] = fourier.get_mag(
				signal_1d, fft_size, hop, window_name, num_cores, prog_sig=self.notifyProgress)
		# logging.info("Cleared Fourier jobs")
		self.jobs.clear()
