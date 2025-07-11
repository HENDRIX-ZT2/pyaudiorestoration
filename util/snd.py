import logging

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QBuffer, QIODevice, Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QSlider, QWidget
from PyQt5.QtMultimedia import QAudio, QAudioFormat, QAudioOutput

from util import io_ops


class AudioWidget(QWidget):

	def __init__(self, parent = None):
	
		QWidget.__init__(self, parent)

		self.format = None
		self.output = None
		self.buffer = QBuffer()
		
		self.volumeSlider = QSlider(Qt.Horizontal)
		self.volumeSlider.setMaximum(10)
		self.volumeSlider.setPageStep(1)
		self.volumeSlider.setValue(5)
		
		self.playButton = QPushButton()
		self.playButton.setIcon(QIcon("icons/play.png"))

		self.stopButton = QPushButton()
		self.stopButton.setIcon(QIcon("icons/stop.png"))
		
		self.volumeSlider.valueChanged.connect(self.change_volume)
		self.playButton.clicked.connect(self.play_pause)
		self.stopButton.clicked.connect(self.stop)
		
		layout = QHBoxLayout(self)
		layout.addWidget(self.playButton)
		layout.addWidget(self.stopButton)
		layout.addWidget(self.volumeSlider)
		layout.addStretch()
	
	def stop(self):
		if self.output:
			if self.output.state() != QAudio.StoppedState:
				self.output.stop()

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
			t = max(0, t)
			pos = self.format.bytesForDuration(int(t*1000000))
			if pos < self.buffer.size():
				self.buffer.seek(pos)

	@property
	def cursor(self):
		"""Get current cursor position in seconds"""
		if self.output:
			return self.format.durationForBytes(self.buffer.pos())/1000000  # microseconds
		return 0.0

	def load_audio(self, fp):
		logging.info(f"Reading audio for playback")
		signal, sr, num_channels = io_ops.read_file(fp)
		self.set_data(signal, sr, list(range(num_channels)))

	def play_pause(self):
		if self.output:
			#(un)pause the audio output, keeps the buffer intact
			if self.output.state() == QAudio.ActiveState:
				self.output.suspend()
			elif self.output.state() == QAudio.SuspendedState:
				self.output.resume()
			else:
				self.buffer.seek(0)
				self.output.start(self.buffer)
			logging.info(f"Seek is at {self.cursor} seconds")
			
	def change_volume(self, value):
		if self.output:
			# need to wrap this because slider gives not float output
			self.output.setVolume(value/10)
	

if __name__ == "__main__":
	fp = "C:/Users/arnfi/Music/CSNY/CSNY WoodenNickel/02 Guinevere.flac"
	app = QApplication([])
	window = AudioWidget()
	window.load_audio(fp)
	window.show()
	app.exec_()