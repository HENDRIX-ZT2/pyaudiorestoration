import os
import numpy as np
import soundfile as sf
from vispy import scene, color
from PyQt5 import QtGui, QtCore, QtWidgets
from scipy import interpolate

#custom modules
from util import vispy_ext, fourier, spectrum, resampling, wow_detection, widgets, io_ops, units, markers

class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pypan", widgets.ParamWidget, Canvas, 1)
		mainMenu = self.menuBar() 
		fileMenu = mainMenu.addMenu('File')
		editMenu = mainMenu.addMenu('Edit')
		button_data = ( (fileMenu, "Open", self.props.file_widget.ask_open, "CTRL+O"), \
						(fileMenu, "Save", self.canvas.save_traces, "CTRL+S"), \
						(fileMenu, "Resample", self.canvas.run_resample, "CTRL+R"), \
						(fileMenu, "Exit", self.close, ""), \
						(editMenu, "Delete Selected", self.canvas.delete_traces, "DEL"), \
						)
		self.add_to_menu(button_data)
		
class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.unfreeze()
		self.parent = parent
		self.deltraces = []
		self.pan_samples = []
		self.pan_line = markers.PanLine(self)
		
		# threading & links
		self.fourier_thread.notifyProgress.connect( self.parent.props.progress_widget.onProgress )
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.setVisible(False)
		self.freeze()
		
	def load_visuals(self,):
		#read pan curve
		for a0, a1, b0, b1, d in io_ops.read_lag(self.filenames[0]):
			markers.PanSample(self, (a0, a1), (b0, b1), d)
		self.pan_line.update()

	def save_traces(self):
		#get the data from the traces and regressions and save it
		io_ops.write_lag(self.filenames[0], [ (lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.pan) for lag in self.pan_samples ] )
			
	def delete_traces(self, delete_all=False):
		self.deltraces= []
		for trace in reversed(self.pan_samples):
			if (trace.selected and not delete_all) or delete_all:
				self.deltraces.append(trace)
		for trace in self.deltraces:
			trace.remove()
		self.pan_line.update()
		#this means a file was loaded, so clear the undo stack
		if delete_all:
			self.deltraces= []
	
	def run_resample(self):
		if self.filenames[0] and self.pan_samples:
			channels = self.parent.props.resampling_widget.channels
			if channels and self.pan_samples:
				lag_curve = self.pan_line.data
				signal, sr, channels = io_ops.read_file(self.filenames[0])
				af = np.interp(np.arange(len(signal[:,0])), lag_curve[:,0]*sr, lag_curve[:,1])
				io_ops.write_file(self.filenames[0], signal[:,1]*af, sr, 1)
					
	def on_mouse_press(self, event):
		# #selection
		# b = self.click_spec_conversion(event.pos)
		# #are they in spec_view?
		# if b is not None:
			# self.props.audio_widget.cursor(b[0])
		if event.button == 2:
			closest_lag_sample = self.get_closest( self.pan_samples, event.pos )
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				event.handled = True
	
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		if self.filenames[0] and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.click_spec_conversion(last_click)
				b = self.click_spec_conversion(click)
				#are they in spec_view?
				if a is not None and b is not None:
					if "Shift" in event.modifiers:
						L = self.fft_storage[ self.keys[0] ]
						R = self.fft_storage[ self.keys[1] ]
						
						t0, t1 = sorted((a[0], b[0]))
						freqs = sorted((a[1], b[1]))
						fL = max(freqs[0], 1)
						fU = min(freqs[1], self.sr//2-1)
						first_fft_i = 0
						num_bins, last_fft_i = L.shape
						#we have specified start and stop times, which is the usual case
						if t0:
							#make sure we force start and stop at the ends!
							first_fft_i = max(first_fft_i, int(t0*self.sr/self.hop)) 
						if t1:
							last_fft_i = min(last_fft_i, int(t1*self.sr/self.hop))

						def freq2bin(f): return max(1, min(num_bins-3, int(round(f * self.fft_size / self.sr))) )
						bL = freq2bin(fL)
						bU = freq2bin(fU)
						
						# dBs = np.nanmean(units.to_dB(L[bL:bU,first_fft_i:last_fft_i])-units.to_dB(R[bL:bU,first_fft_i:last_fft_i]), axis=0)
						# fac = units.to_fac(dBs)
						# out_times = np.arange(first_fft_i, last_fft_i)*hop/sr
						# PanSample(self, a, b, np.mean(fac) )
						
						# faster and simpler equivalent avoiding fac - dB - fac conversion
						fac = np.nanmean(L[bL:bU,first_fft_i:last_fft_i] / R[bL:bU,first_fft_i:last_fft_i])
						markers.PanSample(self, a, b, fac )
						self.pan_line.update()
						
			
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	# monkey patch the file load function
	def load_file_spectrum(self):
		self.parent.parent.canvas.load_audio( (self.filepaths[0], self.filepaths[0]), (0, 1) )
	widgets.FilesWidget.load = load_file_spectrum
	widgets.startup( MainWindow )
