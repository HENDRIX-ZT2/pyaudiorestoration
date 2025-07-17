import logging
import time

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
        for i, (signal_1d, fft_size, hop, window_name, zeropad, key, filename) in enumerate(self.jobs):
            self.notifyProgress.emit(i/len(self.jobs)*100)
            self.result[filename, key] = fourier.get_mag(signal_1d, fft_size, hop, window_name, zeropad)
        # logging.info("Cleared Fourier jobs")
        self.jobs.clear()
        self.notifyProgress.emit(100)


class CursorUpdater(QtCore.QObject):
    """Object representing a complex data producer."""
    new_pos = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._should_end = False
        self._time = 0.0
        self.is_playing = False

    def update_cursor(self):
        logging.info("Run data creation is starting")
        step = 0.04
        while not self._should_end:
            # todo this is inaccurate, but looks ok with step 0.04 on my computer
            time.sleep(step)
            if self.is_playing:
                self._time += step
                self.new_pos.emit(self._time)
        self.finished.emit()

    def update_time(self, t):
        self._time = t
        self.new_pos.emit(self._time)

    def update_playing(self, b):
        self.is_playing = b

    def stop_data(self):
        self._should_end = True
