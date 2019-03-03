import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtCore, QtWidgets
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz
from scipy import interpolate

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, qt_theme, snd, widgets

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='band', output='sos')
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfiltfilt(sos, data)
	return y
	
class ResamplingThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	def run(self):
		names, lag_curve, resampling_mode, sinc_quality, use_channels = self.settings
		resampling.run(names, lag_curve= lag_curve, resampling_mode = resampling_mode, sinc_quality=sinc_quality, use_channels=use_channels, prog_sig=self)
			
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.srcfilename = ""
		self.reffilename = ""
		self.deltraces = []
		
		
		self.display_widget = widgets.DisplayWidget(self.parent.canvas)
		self.resampling_widget = widgets.ResamplingWidget()
		self.audio_widget = snd.AudioWidget()
		self.inspector_widget = widgets.InspectorWidget()
		buttons = [self.display_widget, self.resampling_widget, self.audio_widget, self.inspector_widget ]

		vbox = QtWidgets.QVBoxLayout()
		for w in buttons: vbox.addWidget(w)
		vbox.addStretch(1.0)
		self.setLayout(vbox)

		self.resampling_thread = ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.resampling_widget.onProgress)

		
	def open_audio(self):
		#just a wrapper around load_audio so we can access that via drag & drop and button
		#pyqt5 returns a tuple
		reffilename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Reference', 'c:\\', "Audio files (*.flac *.wav)")[0]
		srcfilename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Source', 'c:\\', "Audio files (*.flac *.wav)")[0]
		self.load_audio(reffilename, srcfilename)
			
	def load_audio(self, reffilename, srcfilename):

		# is the (dropped) file an audio file, ie. can it be read by pysoundfile?
		try:
			soundob = sf.SoundFile(reffilename)
			self.reffilename = reffilename
		except:
			print(reffilename+" could not be read, is it a valid audio file?")
			return
		try:
			soundob = sf.SoundFile(srcfilename)
			self.srcfilename = srcfilename
		except:
			print(srcfilename+" could not be read, is it a valid audio file?")
			return
				
		#Cleanup of old data
		self.parent.canvas.init_fft_storages()
		self.delete_traces(not_only_selected=True)
		self.resampling_widget.refill(soundob.channels)
		
		#finally - proceed with spectrum stuff elsewhere
		self.parent.setWindowTitle('pytapesynch '+os.path.basename(self.reffilename))

		self.parent.canvas.set_file_or_fft_settings((self.reffilename, self.srcfilename),
													 fft_size = self.display_widget.fft_size,
													 fft_overlap = self.display_widget.fft_overlap)
											 
		data = resampling.read_lag(self.reffilename)
		for a0, a1, b0, b1, d in data:
			LagSample(self.parent.canvas, (a0, a1), (b0, b1), d)
		self.parent.canvas.lag_line.update()

	def save_traces(self):
		#get the data from the traces and regressions and save it
		resampling.write_lag(self.reffilename, [ (lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.d) for lag in self.parent.canvas.lag_samples ] )

	def improve_lag(self):
		for lag in self.parent.canvas.lag_samples:
			if lag.selected:
				try:
					#prepare some values
					sr = self.parent.canvas.sr
					raw_lag = int(lag.d*sr)
					ref_t0 = int(sr*lag.a[0])
					ref_t1 = int(sr*lag.b[0])
					src_t0 = ref_t0-raw_lag
					src_t1 = ref_t1-raw_lag
					freqs = sorted((lag.a[1], lag.b[1]))
					lower = max(freqs[0], 1)
					upper = min(freqs[1], sr//2-1)
					ref_pad_l = 0
					ref_pad_r = 0
					src_pad_l = 0
					src_pad_r = 0
					# channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
					
					#trim and pad both sources
					ref_ob = sf.SoundFile(self.reffilename)
					ref_sig = ref_ob.read(always_2d=True, dtype='float32')[:,0]
					if ref_t0 < 0:
						ref_pad_l = abs(ref_t0)
						ref_t0 = 0
					if ref_t1 > len(ref_sig):
						ref_pad_r = ref_t1 - len(ref_sig)
					ref_sig = np.pad( ref_sig[ref_t0:ref_t1], (ref_pad_l, ref_pad_r), "constant", constant_values = 0)
					
					src_ob = sf.SoundFile(self.srcfilename)
					src_sig = src_ob.read(always_2d=True, dtype='float32')[:,0]
					if src_t0 < 0:
						src_pad_l = abs(src_t0)
						src_t0 = 0
					if src_t1 > len(src_sig):
						src_pad_r = src_t1 - len(src_sig)
					src_sig = np.pad( src_sig[src_t0:src_t1], (src_pad_l, src_pad_r), "constant", constant_values = 0)
					# with sf.SoundFile(self.reffilename+"2.wav", 'w+', sr, 1, subtype='FLOAT') as outfile: outfile.write( ref_sig )
					# with sf.SoundFile(self.srcfilename+"2.wav", 'w+', sr, 1, subtype='FLOAT') as outfile: outfile.write( src_sig )
					
					#correlate both sources
					res = np.correlate(butter_bandpass_filter(ref_sig, lower, upper, sr, order=3), butter_bandpass_filter(src_sig, lower, upper, sr, order=3), mode="same")
					#interpolate to get the most accurate fit
					i_peak = wow_detection.parabolic(res, np.argmax(res))[0]
					result = raw_lag + i_peak - len(ref_sig)//2
					#update the lag marker
					lag.d = result/sr
					lag.select()
					self.parent.canvas.lag_line.update()
					print("raw accuracy (smp)",raw_lag)
					print("extra accuracy (smp)",result)
				except:
					print("Refining error!")
	def delete_traces(self, not_only_selected=False):
		self.deltraces= []
		for trace in reversed(self.parent.canvas.lag_samples):
			if (trace.selected and not not_only_selected) or not_only_selected:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.parent.canvas.lag_line.update()
		#this means a file was loaded, so clear the undo stack
		if not_only_selected:
			self.deltraces= []
	
	def run_resample(self):
		if self.srcfilename and self.parent.canvas.lag_samples:
			channels = self.resampling_widget.channels
			if channels and self.parent.canvas.lag_samples:
				lag_curve = self.parent.canvas.lag_line.data
				self.resampling_thread.settings = ((self.srcfilename,), lag_curve, self.resampling_widget.mode, self.resampling_widget.sinc_quality, channels)
				self.resampling_thread.start()
		
	def run_resample_batch(self):
		if self.srcfilename and self.parent.canvas.lag_samples:
			filenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Files for Batch Resampling', 'c:\\', "Audio files (*.flac *.wav)")[0]
			channels = self.resampling_widget.channels
			if channels and self.parent.canvas.lag_samples:
				lag_curve = self.parent.canvas.lag_line.data
				self.resampling_thread.settings = (filenames, lag_curve, self.resampling_widget.mode, self.resampling_widget.sinc_quality, channels)
				self.resampling_thread.start()
					
class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pytapesynch", ObjectWidget, Canvas, accept_drag=False)
		mainMenu = self.menuBar() 
		fileMenu = mainMenu.addMenu('File')
		editMenu = mainMenu.addMenu('Edit')
		button_data = ( (fileMenu, "Open", self.props.open_audio, "CTRL+O"), \
						(fileMenu, "Save", self.props.save_traces, "CTRL+S"), \
						(fileMenu, "Resample", self.props.run_resample, "CTRL+R"), \
						(fileMenu, "Batch Resample", self.props.run_resample_batch, "CTRL+B"), \
						(fileMenu, "Exit", self.close, ""), \
						(editMenu, "Improve", self.props.improve_lag, "CTRL+I"), \
						(editMenu, "Delete Selected", self.props.delete_traces, "DEL"), \
						)
		self.add_to_menu(button_data)

class LagLine:
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		
		self.vispy_canvas = vispy_canvas
		
		#create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.data[:, 1] = (0, 0)
		self.line_speed = scene.Line(pos=self.data, color=(0, 0, 1, .5), method='gl')
		self.line_speed.parent = vispy_canvas.speed_view.scene
		
	def update(self):
		if self.vispy_canvas.lag_samples:
			try:
				self.vispy_canvas.lag_samples.sort(key=lambda tup: tup.t)
				sample_times = [sample.t for sample in self.vispy_canvas.lag_samples]
				sample_lags = [sample.d for sample in self.vispy_canvas.lag_samples]
				
				num = self.vispy_canvas.num_ffts
				times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
				# lag = np.interp(times, sample_times, sample_lags)
				lag = interpolate.interp1d(sample_times, sample_lags, fill_value="extrapolate")(times)

				#quadratic interpolation does not give usable results
				# interolator = scipy.interpolate.interp1d(sample_times, sample_lags, kind='quadratic', bounds_error=False, fill_value="extrapolate")
				# lag = interolator(times)
				
				#using bezier splines; probably needs to be done segment by segment
				# tck,u = interpolate.splprep([times,lag],k=2,s=0)
				# # u=np.linspace(0,1,num=10000,endpoint=True)
				# out = interpolate.splev(u,tck)
				# x=out[0]
				# y=out[1]
				# #create the speed curve visualization, boost it a bit to distinguish from the raw curves
				# self.data = np.zeros((len(x), 2), dtype=np.float32)
				# self.data[:, 0] = x
				# self.data[:, 1] = y
				#create the speed curve visualization, boost it a bit to distinguish from the raw curves
				self.data = np.zeros((len(times), 2), dtype=np.float32)
				self.data[:, 0] = times
				self.data[:, 1] = lag
			except: pass
		else:
			self.data = np.zeros((2, 2), dtype=np.float32)
			self.data[:, 0] = (0, 999)
			self.data[:, 1] = (0, 0)
		self.line_speed.set_data(pos=self.data)
				
class LagSample():
	"""Stores a single sinc regression's data and displays it"""
	def __init__(self, vispy_canvas, a, b, d=None):
		
		self.a = a
		self.b = b
		
		self.t = (a[0]+b[0])/2
		if d is None:
			self.d = vispy_canvas.spectra[-1].delta
		else:
			self.d = d
		self.width= abs(a[0]-b[0])
		self.f = (a[1]+b[1])/2
		self.height= abs(a[1]-b[1])
		self.spec_center = (self.t, self.f)
		self.speed_center = (self.t, self.d)
		self.rect = scene.Rectangle(center=(self.t, self.f), width=self.width, height=self.height, radius=0, parent=vispy_canvas.spec_view.scene)
		self.rect.color = (1, 1, 1, .5)
		self.rect.transform = vispy_canvas.spectra[-1].mel_transform
		self.rect.set_gl_state('additive')
		self.selected = False
		self.vispy_canvas = vispy_canvas
		self.initialize()
		
	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.vispy_canvas.lag_samples.append(self)
		self.vispy_canvas.lag_line.update()

	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		self.rect.color = (1, 1, 1, .5)
		
	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.rect.color = (0, 0, 1, .5)
		new_d = self.vispy_canvas.spectra[-1].delta
		self.vispy_canvas.spectra[-1].translate(self.d-new_d)
		
	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()
			
	def select_handle(self, multi=False):
		if not multi:
			for lag_sample in self.vispy_canvas.lag_samples:
				lag_sample.deselect()
		self.toggle()
	
	def show(self):
		self.rect.parent = self.vispy_canvas.spec_view.scene
		
	def hide(self):
		self.rect.parent = None
		self.deselect()
		
	def remove(self):
		self.hide()
		#note: this has to search the list
		self.vispy_canvas.lag_samples.remove(self)
		
class Canvas(spectrum.SpectrumCanvas):

	def __init__(self):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.unfreeze()
		self.lag_samples = []
		self.lag_line = LagLine(self)
		self.freeze()
		
	#called if either  the file or FFT settings have changed
	def set_file_or_fft_settings(self, files, fft_size = 256, fft_overlap = 1):
		if files:
			self.compute_spectra(files, fft_size, fft_overlap)
			self.lag_line.update()
		
	def on_mouse_press(self, event):
		#selection
		b = self.click_spec_conversion(event.pos)
		#are they in spec_view?
		if b is not None:
			self.props.audio_widget.cursor(b[0])
		if event.button == 2:
			closest_lag_sample = self.get_closest( self.lag_samples, event.pos )
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		if self.props.srcfilename and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				#are they in spec_view?
				if a is not None and b is not None:
					if "Control" in event.modifiers:
						d = b[0]-a[0]
						self.spectra[1].translate(d)
					elif "Shift" in event.modifiers:
						LagSample(self, a, b)
					# elif "Alt" in event.modifiers:
						# print()
						# print("Start")
						# #first get the time range for both
						# #apply bandpass
						# #split into pieces and look up the delay for each
						# #correlate all the pieces
						# sr = self.sr
						# dur = int(0.2 *sr)
						# times = sorted((a[0], b[0]))
						# ref_t0 = int(sr*times[0])
						# ref_t1 = int(sr*times[1])
						# # src_t0 = int(sr*lag.a[0]+lag.d)
						# # src_t1 = src_t0+ref_t1-ref_t0
						# freqs = sorted((a[1], b[1]))
						# lower = max(freqs[0], 1)
						# upper = min(freqs[1], sr//2-1)
						# # channels = [i for i in range(len(self.channel_checkboxes)) if self.channel_checkboxes[i].isChecked()]
						# ref_ob = sf.SoundFile(self.props.reffilename)
						# ref_sig = ref_ob.read(always_2d=True, dtype='float32')
						# src_ob = sf.SoundFile(self.props.srcfilename)
						# src_sig = src_ob.read(always_2d=True, dtype='float32')
						# sample_times = np.arange(ref_t0, ref_t1, dur//32)
						# data = self.lag_line.data
						# sample_lags = np.interp(sample_times, data[:, 0]*sr, data[:, 1]*sr)
						
						# #could do a stack
						# out = np.zeros((len(sample_times), 2), dtype=np.float32)
						# out[:, 0] = sample_times/sr
						# for i, (x, d) in enumerate(zip(sample_times, sample_lags)):
							
							# ref_s = butter_bandpass_filter(ref_sig[x:x+dur,0], lower, upper, sr, order=3)
							# src_s = butter_bandpass_filter(src_sig[x-int(d):x-int(d)+dur,0], lower, upper, sr, order=3)
							# res = np.correlate(ref_s*np.hanning(dur), src_s*np.hanning(dur), mode="same")
							# i_peak = np.argmax(res)
							# #interpolate the most accurate fit
							# result = wow_detection.parabolic(res, i_peak)[0] -(len(ref_s)//2)
							# # print("extra accuracy (smp)",int(d)+result)
							# out[i, 1] = (int(d)+result)/sr
						# self.lag_line.data =out
						# self.lag_line.line_speed.set_data(pos=out)
							
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	appQt = QtWidgets.QApplication([])
	
	#style
	appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
	appQt.setPalette(qt_theme.dark_palette)
	appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")
	
	win = MainWindow()
	win.show()
	appQt.exec_()
