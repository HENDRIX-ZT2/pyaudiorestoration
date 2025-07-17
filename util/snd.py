import logging
import math

from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QBuffer, QIODevice, Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QSlider, QWidget, QLabel, QCheckBox
from PyQt5.QtMultimedia import QAudio, QAudioFormat, QAudioOutput

from util import io_ops


class AudioWidget(QWidget):

	cursor_set = QtCore.pyqtSignal(float)
	is_playing = QtCore.pyqtSignal(bool)

	def __init__(self, parent = None):
	
		QWidget.__init__(self, parent)

		self.format = None
		self.output = None
		self.buffer = QBuffer()
		
		self.volumeSlider = QSlider(Qt.Horizontal)
		self.volumeSlider.setMinimum(0)
		self.volumeSlider.setMaximum(100)
		self.volumeSlider.setPageStep(1)
		self.volumeSlider.setValue(50)
		self.volumeSlider.valueChanged.connect(self.change_volume)
		
		self.playButton = QPushButton()
		self.playButton.setIcon(QIcon("icons/play.png"))

		self.stopButton = QPushButton()
		self.stopButton.setIcon(QIcon("icons/stop.png"))

		self.playButton.clicked.connect(self.play_pause)
		self.stopButton.clicked.connect(self.stop)

		self.volume_label = QLabel()
		self.volume_label.setPixmap(QIcon("icons/volume.svg").pixmap(16))

		self.scrub_button = QCheckBox("Scroll")

		layout = QHBoxLayout(self)
		layout.addWidget(self.playButton)
		layout.addWidget(self.stopButton)
		layout.addWidget(self.volume_label)
		layout.addWidget(self.volumeSlider)
		layout.addWidget(self.scrub_button)
		layout.addStretch()
	
	def stop(self):
		if self.output:
			if self.output.state() != QAudio.StoppedState:
				self.is_playing.emit(False)
				self.output.stop()
				self.set_cursor(0.0)

	def set_data(self, mono_sig, sr, channels):
		# print(mono_sig.shape, sr, channels)
		mono_sig = mono_sig[:, channels]
		# if not self.format:
		self.format = QAudioFormat()
		self.format.setChannelCount(len(channels))
		self.format.setSampleRate(sr)
		# numpy is in bytes, qt in bits
		self.format.setSampleSize(mono_sig.dtype.itemsize*8)
		self.format.setCodec("audio/pcm")
		self.format.setByteOrder(QAudioFormat.LittleEndian)
		self.format.setSampleType(QAudioFormat.Float)
		self.output = QAudioOutput(self.format, self)
		self.output.stateChanged.connect(self.audio_state_changed)
		self.change_volume(self.volumeSlider.value())
		# change the content without stopping playback
		p = self.buffer.pos()
		if self.buffer.isOpen():
			self.buffer.close()
		
		self.data = mono_sig.tobytes()
		self.buffer.setData(self.data)
		self.buffer.open(QIODevice.ReadWrite)
		self.buffer.seek(p)
		
	def audio_state_changed(self, new_state):
		#adjust the button icon
		if new_state != QAudio.ActiveState:
			self.playButton.setIcon(QIcon("icons/play.png"))
		else:
			self.playButton.setIcon(QIcon("icons/pause.png"))
		
	def set_cursor(self, t):
		"""seek towards time t in playback buffer"""
		if self.format:
			t = max(0.0, t)
			self.cursor_set.emit(t)
			pos = self.format.bytesForDuration(int(t*1000000))
			if pos < self.buffer.size():
				self.buffer.seek(pos)

	@property
	def cursor(self):
		"""Get current cursor position in seconds"""
		if self.output:
			return self.format.durationForBytes(self.buffer.pos())/1000000  # microseconds
		return 0.0

	@property
	def scroll_view(self):
		return self.scrub_button.isChecked()

	def load_audio(self, fp):
		logging.info(f"Reading audio for playback")
		signal, sr, num_channels = io_ops.read_file(fp)
		self.set_data(signal, sr, list(range(num_channels)))

	def play_pause(self):
		if self.output:
			#(un)pause the audio output, keeps the buffer intact
			if self.output.state() == QAudio.ActiveState:
				self.output.suspend()
				self.is_playing.emit(False)
			elif self.output.state() == QAudio.SuspendedState:
				self.output.resume()
				self.is_playing.emit(True)
			else:
				self.is_playing.emit(True)
				self.output.start(self.buffer)
			self.cursor_set.emit(self.cursor)
			logging.info(f"Seek is at {self.cursor} seconds")
			
	def change_volume(self, value):
		if self.output:
			linearVolume = QAudio.convertVolume(value / self.volumeSlider.maximum(), QAudio.LogarithmicVolumeScale, QAudio.LinearVolumeScale)
			self.output.setVolume(linearVolume)
	

if __name__ == "__main__":
	fp = "C:/Users/arnfi/Music/CSNY/CSNY WoodenNickel/02 Guinevere.flac"
	app = QApplication([])
	window = AudioWidget()
	window.load_audio(fp)
	window.show()
	app.exec_()