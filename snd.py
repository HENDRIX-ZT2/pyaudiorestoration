import numpy as np
import soundfile as sf

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QBuffer, QIODevice, Qt
from PyQt5.QtWidgets import QApplication, QFormLayout, QLineEdit, QHBoxLayout, QPushButton, QSlider, QVBoxLayout, QWidget
from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioOutput

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
		
		buttonLayout = QVBoxLayout()
		buttonLayout.addWidget(self.playButton)
		buttonLayout.addWidget(self.stopButton)
		buttonLayout.addWidget(self.volumeSlider)
		buttonLayout.addStretch()
		
		horizontalLayout = QHBoxLayout(self)
		horizontalLayout.addLayout(buttonLayout)
	
		
	def stop(self):
		if self.output:
			if self.output.state() != QAudio.StoppedState:
				self.output.stop()
				self.playButton.setIcon(QIcon("icons/play.png"))
				

	def set_data(self, mono_sig, sr):
		# if not self.format:
		self.format = QAudioFormat()
		self.format.setChannelCount(1)
		self.format.setSampleRate(sr)
		#numpy is in bites, qt in bits
		self.format.setSampleSize(mono_sig.dtype.itemsize*8)
		self.format.setCodec("audio/pcm")
		self.format.setByteOrder(QAudioFormat.LittleEndian)
		self.format.setSampleType(QAudioFormat.Float)
		self.output = QAudioOutput(self.format, self)
		#change the content without stopping playback
		p = self.buffer.pos()
		if self.buffer.isOpen():
			self.buffer.close()
		
		print("pos",p)
			
		self.data = mono_sig.tobytes()
		self.buffer.setData(self.data)
		self.buffer.open(QIODevice.ReadWrite)
		self.buffer.seek(p)
		
	def cursor(self, t):
		#seek towards the time t
		#todo: handle EOF case
		try:
			if self.format:
				t = max(0, t)
				b = self.format.bytesForDuration(t*1000000)
				self.buffer.seek(b)
		except:
			print("cursor error")
		
	def play_pause(self):
		if self.output:
			#(un)pause the audio output, keeps the buffer intact
			if self.output.state() == QAudio.ActiveState:
				self.output.suspend()
				self.playButton.setIcon(QIcon("icons/play.png"))
			elif self.output.state() == QAudio.SuspendedState:
				self.output.resume()
				self.playButton.setIcon(QIcon("icons/pause.png"))
			else:
				self.buffer.seek(0)
				self.output.start(self.buffer)
				self.playButton.setIcon(QIcon("icons/pause.png"))
			
	def change_volume(self, value):
		if self.output:
			#need to wrap this because slider gives not float output
			self.output.setVolume(value/10)
	

if __name__ == "__main__":

	app = QApplication([])
	window = AudioWidget()
	window.show()
	app.exec_()