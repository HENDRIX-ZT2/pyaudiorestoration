import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtCore, QtWidgets
from scipy import interpolate

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, qt_threads, snd, widgets, filters, io_ops, markers

class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pytapesynch", widgets.ParamWidget, Canvas, accept_drag=False)
		mainMenu = self.menuBar() 
		fileMenu = mainMenu.addMenu('File')
		editMenu = mainMenu.addMenu('Edit')
		button_data = ( (fileMenu, "Open", self.canvas.open_audio, "CTRL+O"), \
						(fileMenu, "Save", self.canvas.save_traces, "CTRL+S"), \
						(fileMenu, "Resample", self.canvas.run_resample, "CTRL+R"), \
						(fileMenu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B"), \
						(fileMenu, "Exit", self.close, ""), \
						(editMenu, "Select All", self.canvas.select_all, "CTRL+A"), \
						(editMenu, "Improve", self.canvas.improve_lag, "CTRL+I"), \
						(editMenu, "Delete Selected", self.canvas.delete_traces, "DEL"), \
						)
		self.add_to_menu(button_data)
	
class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.create_native()
		self.native.setParent(parent)
		
		self.unfreeze()
		self.parent = parent
		self.deltraces = []
		self.lag_samples = []
		self.lag_line = markers.LagLine(self)
		
		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_widget.onProgress)
		self.fourier_thread.notifyProgress.connect( self.parent.props.progress_widget.onProgress )
		self.parent.props.display_widget.canvas = self
		self.freeze()
		
	def load_visuals(self,):
		for a0, a1, b0, b1, d in io_ops.read_lag(self.filenames[0]):
			markers.LagSample(self, (a0, a1), (b0, b1), d)
		self.lag_line.update()

	def save_traces(self):
		#get the data from the traces and regressions and save it
		io_ops.write_lag(self.filenames[0], [ (lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.d) for lag in self.lag_samples ] )

	def improve_lag(self):
		for lag in self.lag_samples:
			if lag.selected:
				try:
					#prepare some values
					sr = self.sr
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
					ref_sig = self.signals[0]
					src_sig = self.signals[1]
					if ref_t0 < 0:
						ref_pad_l = abs(ref_t0)
						ref_t0 = 0
					if ref_t1 > len(ref_sig):
						ref_pad_r = ref_t1 - len(ref_sig)
					
					if src_t0 < 0:
						src_pad_l = abs(src_t0)
						src_t0 = 0
					if src_t1 > len(src_sig):
						src_pad_r = src_t1 - len(src_sig)
						
					ref_sig = np.pad( ref_sig[ref_t0:ref_t1, 0], (ref_pad_l, ref_pad_r), "constant", constant_values = 0)
					src_sig = np.pad( src_sig[src_t0:src_t1, 0], (src_pad_l, src_pad_r), "constant", constant_values = 0)
					
					#correlate both sources
					res = np.correlate(filters.butter_bandpass_filter(ref_sig, lower, upper, sr, order=3), filters.butter_bandpass_filter(src_sig, lower, upper, sr, order=3), mode="same")
					#interpolate to get the most accurate fit
					# we are not necessarily interested in the largest positive value if the correlation is negative
					# todo: add a toggle for this
					# i_peak = wow_detection.parabolic(res, np.argmax(np.abs(res)))[0]
					i_peak = wow_detection.parabolic(res, np.argmax(res))[0]
					result = raw_lag + i_peak - len(ref_sig)//2
					#update the lag marker
					lag.d = result/sr
					lag.select()
					self.lag_line.update()
					print("raw accuracy (smp)",raw_lag)
					print("extra accuracy (smp)",result)
				except:
					print("Refining error!")
	
	def select_all(self):
		for trace in self.lag_samples:
			trace.select()
			
	def delete_traces(self, not_only_selected=False):
		self.deltraces= []
		for trace in reversed(self.lag_samples):
			if (trace.selected and not not_only_selected) or not_only_selected:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.lag_line.update()
		#this means a file was loaded, so clear the undo stack
		if not_only_selected:
			self.deltraces= []
	
	def run_resample(self):
		self.resample_files( (self.filenames[1],) )
	
	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Files for Batch Resampling', 'c:\\', "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files( filenames )
	
	def resample_files(self, files):
		channels = self.parent.props.resampling_widget.channels
		if self.filenames[1] and self.lag_samples and channels:
			lag_curve = self.lag_line.data
			self.resampling_thread.settings = {"filenames"			:files,
												"lag_curve"			:lag_curve, 
												"resampling_mode"	:self.parent.props.resampling_widget.mode,
												"sinc_quality"		:self.parent.props.resampling_widget.sinc_quality,
												"use_channels"		:channels}
			self.resampling_thread.start()
		
	def on_mouse_press(self, event):
		# #selection
		# b = self.click_spec_conversion(event.pos)
		# #are they in spec_view?
		# if b is not None:
			# self.props.audio_widget.cursor(b[0])
		if event.button == 2:
			closest_lag_sample = self.get_closest( self.lag_samples, event.pos )
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		if self.filenames[1] and (event.trail() is not None) and event.button == 1:
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
						markers.LagSample(self, a, b)
						self.lag_line.update()
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
							
							# ref_s = filters.butter_bandpass_filter(ref_sig[x:x+dur,0], lower, upper, sr, order=3)
							# src_s = filters.butter_bandpass_filter(src_sig[x-int(d):x-int(d)+dur,0], lower, upper, sr, order=3)
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
	widgets.startup( MainWindow )
