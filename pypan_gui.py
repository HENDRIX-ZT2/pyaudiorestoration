import numpy as np

# custom modules
from util.undo import AddAction, DeleteAction
from util import spectrum, widgets, io_ops, markers


class MainWindow(widgets.MainWindow):

	def __init__(self):
		widgets.MainWindow.__init__(self, "pypan", widgets.ParamWidget, Canvas, 1)
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		button_data = (
			(file_menu, "Open", self.props.files_widget.ask_open, "CTRL+O", "dir"),
			(file_menu, "Save", self.canvas.save_traces, "CTRL+S", "save"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R", "curve"),
			(file_menu, "Exit", self.close, "", "exit"),
			(edit_menu, "Undo", self.props.undo_stack.undo, "CTRL+Z", "undo"),
			(edit_menu, "Redo", self.props.undo_stack.redo, "CTRL+Y", "redo"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A", "select_extend"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL", "x"),
		)
		self.add_to_menu(button_data)


class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black")
		self.unfreeze()
		self.parent = parent
		self.pan_samples = []
		self.pan_line = markers.PanLine(self)

		# threading & links
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.parent.props.display_widget.canvas = self
		self.parent.props.tracing_widget.setVisible(False)
		self.freeze()

	def update_lines(self):
		self.pan_line.update()

	def select_all(self):
		for trace in self.pan_samples:
			trace.select()

	def deselect_all(self):
		for trace in self.pan_samples:
			trace.deselect()

	def load_visuals(self, ):
		for a0, a1, b0, b1, d in io_ops.read_lag(self.filenames[0]):
			yield markers.PanSample(self, (a0, a1), (b0, b1), d)

	def save_traces(self):
		# get the data from the traces and regressions and save it
		io_ops.write_lag(
			self.filenames[0], [(lag.a[0], lag.a[1], lag.b[0], lag.b[1], lag.pan) for lag in self.pan_samples])
		self.parent.props.undo_stack.setClean()

	def delete_traces(self, delete_all=False):
		deltraces = []
		for trace in reversed(self.pan_samples):
			if delete_all or trace.selected:
				deltraces.append(trace)
		self.props.undo_stack.push(DeleteAction(deltraces))

	def run_resample(self):
		if self.filenames[0] and self.pan_samples:
			lag_curve = self.pan_line.data
			signal, sr, channels = io_ops.read_file(self.filenames[0])
			af = np.interp(np.arange(len(signal[:, 0])), lag_curve[:, 0] * sr, lag_curve[:, 1])
			io_ops.write_file(self.filenames[0], signal[:, 1] * af, sr, 1)

	def on_mouse_press(self, event):
		# #selection
		if event.button == 2:
			closest_lag_sample = self.get_closest(self.pan_samples, event.pos)
			if closest_lag_sample:
				closest_lag_sample.select_handle()
				event.handled = True

	def on_mouse_release(self, event):
		# coords of the click on the vispy canvas
		if self.filenames[0] and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.px_to_spectrum(last_click)
				b = self.px_to_spectrum(click)
				# are they in spec_view?
				if a is not None and b is not None:
					if "Shift" in event.modifiers:
						L, R = [self.fft_storage[spectrum.key] for spectrum in self.spectra]

						t0, t1 = sorted((a[0], b[0]))
						freqs = sorted((a[1], b[1]))
						fL = max(freqs[0], 1)
						fU = min(freqs[1], self.sr // 2 - 1)
						first_fft_i = 0
						num_bins, last_fft_i = L.shape
						# we have specified start and stop times, which is the usual case
						if t0:
							# make sure we force start and stop at the ends!
							first_fft_i = max(first_fft_i, int(t0 * self.sr / self.hop))
						if t1:
							last_fft_i = min(last_fft_i, int(t1 * self.sr / self.hop))

						def freq2bin(f):
							return max(1, min(num_bins - 3, int(round(f * self.fft_size / self.sr))))

						bL = freq2bin(fL)
						bU = freq2bin(fU)

						# dBs = np.nanmean(units.to_dB(L[bL:bU,first_fft_i:last_fft_i])-units.to_dB(R[bL:bU,first_fft_i:last_fft_i]), axis=0)
						# fac = units.to_fac(dBs)
						# faster and simpler equivalent avoiding fac - dB - fac conversion
						fac = np.nanmean(L[bL:bU, first_fft_i:last_fft_i] / R[bL:bU, first_fft_i:last_fft_i])
						self.props.undo_stack.push(AddAction((markers.PanSample(self, a, b, fac),)))


if __name__ == '__main__':
	# monkey patch the file load function
	def load_file_spectrum(self):
		self.parent.parent.canvas.load_audio((self.filepaths[0], self.filepaths[0]), (0, 1))

	widgets.FilesWidget.load = load_file_spectrum
	widgets.startup(MainWindow)
