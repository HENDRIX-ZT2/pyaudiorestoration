import logging

import matplotlib.pyplot as plt
import numpy as np
import resampy
import scipy
from PyQt5 import QtWidgets
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

# custom modules
from util.correlation import find_delay
from util.filters import make_odd
from util.fourier import to_mag
from util.undo import AddAction, DeltaAction
from util import spectrum, qt_threads, widgets, filters, io_ops, markers, fourier

from util.config import logging_setup
from util.units import to_fac, to_dB
from util.wow_detection import interp_nans

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
logging_setup()


class MainWindow(widgets.MainWindow):
	EXT = ".drop"
	STORE = {"dropouts": markers.DropoutSample, }

	def __init__(self):
		widgets.MainWindow.__init__(self, "Dropout Healer", widgets.ParamWidget, Canvas, 1)
		self.props.display_widget.fft_size = 512
		self.props.display_widget.fft_overlap = 16
		main_menu = self.menuBar()
		file_menu = main_menu.addMenu('File')
		edit_menu = main_menu.addMenu('Edit')
		button_data = (
			(file_menu, "Open", self.props.load, "CTRL+O", "dir"),
			(file_menu, "Save", self.props.save, "CTRL+S", "save"),
			(file_menu, "Resample", self.canvas.run_resample, "CTRL+R", "curve"),
			(file_menu, "Batch Resample", self.canvas.run_resample_batch, "CTRL+B", "curve2"),
			(file_menu, "Exit", self.close, "", "exit"),
			(edit_menu, "Select All", self.canvas.select_all, "CTRL+A", "select_extend"),
			# (edit_menu, "Improve", self.canvas.improve_lag, "CTRL+I"),
			(edit_menu, "Delete Selected", self.canvas.delete_traces, "DEL", "x"),
			(edit_menu, "Undo", self.props.undo_stack.undo, "CTRL+Z", "undo"),
			(edit_menu, "Redo", self.props.undo_stack.redo, "CTRL+Y", "redo"),
			(edit_menu, "Play/Pause", self.props.audio_widget.play_pause, "SPACE"),
		)
		self.add_to_menu(button_data)


class Canvas(spectrum.SpectrumCanvas):

	def __init__(self, parent):
		spectrum.SpectrumCanvas.__init__(self, bgcolor="black", y_axis='Intensity')
		self.create_native()
		self.native.setParent(parent)

		self.unfreeze()
		self.parent = parent
		self.lag_line = markers.DropoutLine(self)

		# threading & links
		self.resampling_thread = qt_threads.ResamplingThread()
		self.resampling_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.fourier_thread.notifyProgress.connect(self.parent.props.progress_bar.setValue)
		self.parent.props.display_widget.canvas = self
		self.parent.props.filters_widget.bands_changed.connect(self.lag_line.update_bands)
		self.parent.props.tracing_widget.setVisible(False)
		self.parent.props.alignment_widget.setVisible(False)
		self.parent.props.filters_widget.setVisible(False)
		self.parent.props.output_widget.mode_l.setVisible(False)
		self.parent.props.output_widget.mode_c.setVisible(False)
		self.freeze()
		self.parent.props.dropout_widget.surrounding_s.editingFinished.connect(self.update_surrounding)

	@property
	def dropouts(self):
		return [m for m in self.markers if isinstance(m, markers.DropoutSample)]

	def update_lines(self):
		self.lag_line.update()

	def update_surrounding(self):
		f = self.parent.props.dropout_widget.surrounding
		for reg in self.dropouts:
			if reg.selected:
				reg.surrounding = f

	def load_visuals(self, ):
		"""legacy code path"""
		for a0, a1, b0, b1, d in io_ops.read_lag(self.filenames[0]):
			yield markers.LagSample(self, (a0, a1), (b0, b1), d)

	def run_resample(self):
		self.resample_files((self.filenames[1],))

	def run_resample_batch(self):
		filenames = QtWidgets.QFileDialog.getOpenFileNames(
			self.parent, 'Open Files for Batch Resampling',
			self.parent.cfg.get("dir_in", ""), "Audio files (*.flac *.wav)")[0]
		if filenames:
			self.resample_files(filenames)

	def time_2_frame(self, t):
		# todo unify on spectrum class
		return int(t * self.sr / self.hop)

	def freq_2_bin(self, f):
		# todo unify on spectrum class
		return max(1, min(self.fft_size//2, int(round(f * self.fft_size / self.sr))))

	def resample_files(self, files):
		channels = self.props.files_widget.files[0].channel_widget.channels
		if self.filenames[0] and self.markers and channels:
			# lag_curve = self.lag_line.data
			# self.resampling_thread.settings = {
			# 	"filenames"			: files,
			# 	"lag_curve"			: lag_curve,
			# 	"use_channels"		: channels}
			self.props.output_widget.bump_index()
			# self.props.output_widget.to_cfg(self.resampling_thread.settings)
			# self.resampling_thread.start()
			fft_size = self.fft_size
			hop = self.hop

			for file_path in files:
				signal, sr, num_channels = io_ops.read_file(file_path)
				# if num_channels != 2:
				# 	print("expects stereo input")
				# 	continue
				n = len(signal)
				# pad input stereo signal
				y_pad = fourier.fix_length(signal, n + fft_size // 2, axis=0)
				# take FFT for each channel
				D_L = np.array(fourier.stft(y_pad[:, 0], n_fft=fft_size, step=hop))
				# print(D_L.shape, fft_size)
				# D_R = np.array(fourier.stft(y_pad[:, 1], n_fft=fft_size, step=hop))
				DL_mag = to_dB(to_mag(D_L))
				boost_mask = np.zeros(D_L.shape, dtype=float)
				# peaks =
				for drop in self.markers:
					frame_l = self.time_2_frame(drop.t - (drop.width / 2))
					frame_r = self.time_2_frame(drop.t + (drop.width / 2))
					# parametrize frame_surrounding as percentage, set from UI for selected or as preset
					frame_surrounding = self.time_2_frame(drop.width * drop.surrounding)
					bin_l = self.freq_2_bin(drop.f - (drop.height / 2))
					bin_u = self.freq_2_bin(drop.f + (drop.height / 2))

					# take mean of left and right frames
					mag_left = np.mean(DL_mag[bin_l:bin_u, frame_l-frame_surrounding:frame_l], axis=1)
					mag_right = np.mean(DL_mag[bin_l:bin_u, frame_r:frame_r+frame_surrounding], axis=1)

					# interpolate the desired spectrum for each bin in the selected region
					region_mag = DL_mag[bin_l:bin_u, frame_l:frame_r]
					fp_frames = np.linspace(frame_l, frame_r, num=frame_r-frame_l)
					fp_bins = np.linspace(bin_l, bin_u, num=bin_u-bin_l)

					interp = RegularGridInterpolator(((frame_l, frame_r), fp_bins), (mag_left, mag_right))
					mp_bins, mp_frames = np.meshgrid(fp_bins, fp_frames)  # 2D grid for interpolation
					fp_dB = interp((mp_frames, mp_bins))
					fp_dB = np.swapaxes(fp_dB, 0, 1)
					# plt.pcolormesh(fp_dB, shading='auto')
					# calculate boost to bring dropout up to desired volume
					diff = fp_dB - region_mag
					# take at least as much as the previous gain for each bin
					np.clip(diff, boost_mask[bin_l:bin_u, frame_l:frame_r], 255, out=diff)
					# print(diff)
					# plt.pcolormesh(diff, shading='auto')
					# plt.show()
					# store boost in mask
					boost_mask[bin_l:bin_u, frame_l:frame_r] = diff
				# correct fft data with boost
				D_out = D_L * to_fac(boost_mask)
				# take iFFT
				y_out = fourier.istft(D_out, length=n, hop_length=hop)

				io_ops.write_file(file_path, y_out, sr, 1, suffix=f"_drops{self.props.output_widget.suffix}")
		
	def on_mouse_press(self, event):
		# selection
		if event.button == 1:
			# audio cursor
			b = self.px_to_spectrum(event.pos)
			# are they in spec_view?
			if b is not None:
				self.props.audio_widget.set_cursor(b[0])
		if event.button == 2:
			closest_marker = self.get_closest(self.markers, event.pos)
			if closest_marker:
				closest_marker.select_handle("Shift" in event.modifiers)
				self.update_corr_view(closest_marker)
				event.handled = True

	def update_corr_view(self, closest_marker):
		v = "None" if closest_marker.corr is None else f"{closest_marker.corr:.3f}"
		self.parent.props.alignment_widget.corr_l.setText(v)

	def get_speed_at(self, t):
		width = 0.05
		# calc speed across range
		data = self.lag_line.data
		# smooth / lowpass lag curve to get a better derivative
		filtered = data[:, 1]
		filtered = filters.butter_bandpass_filter(filtered, 0, 15, self.lag_line.marker_sr, order=3)
		# for input in t, sample lag at t+-range (0.5 s?)
		before = np.interp(t-width, data[:, 0], filtered)
		after = np.interp(t+width, data[:, 0], filtered)
		speed = (after - before) / (2 * width) + 1.0
		logging.info(f"Source runs {(speed-1)*100:0.2f}% wrong")
		return speed
		# from matplotlib import pyplot as plt
		# plt.plot(filtered, label=f"filtered")
		# plt.plot(data[:, 1], label=f"raw")
		# plt.legend(frameon=True, framealpha=0.75)
		# plt.show()

	def on_mouse_release(self, event):
		# coords of the click on the vispy canvas
		if self.filenames[1] and (event.trail() is not None) and event.button == 1:
			last_click = event.trail()[0]
			click = event.pos
			if last_click is not None:
				a = self.px_to_spectrum(last_click)
				b = self.px_to_spectrum(click)
				# are they in spec_view?
				if a is not None and b is not None:
					# direct mode
					if "Shift" in event.modifiers:
						marker = markers.DropoutSample(self, a, b)
						self.props.undo_stack.push(AddAction((marker,)))
					# batch detection
					elif "Alt" in event.modifiers:
						pass
						# self.props.undo_stack.push(AddAction((markers,)))

	def get_times_freqs(self, a, b, sr):
		ref_t0, ref_t1 = sorted((a[0], b[0]))
		freqs = sorted((a[1], b[1]))
		lower = max(freqs[0], 1)
		upper = min(freqs[1], sr // 2 - 1)
		return ref_t0, ref_t1, lower, upper


if __name__ == '__main__':
	widgets.startup(MainWindow)
