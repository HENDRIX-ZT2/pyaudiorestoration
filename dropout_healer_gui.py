import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy
from PyQt5 import QtWidgets
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter

# custom modules
from util.fourier import to_mag
from util.undo import AddAction
from util import spectrum, qt_threads, widgets, io_ops, markers, fourier

from util.config import logging_setup
from util.units import to_fac, to_dB

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

	def frame_2_time(self, f):
		# todo unify on spectrum class
		return f / self.sr * self.hop

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
				output = np.empty(signal.shape, dtype=signal.dtype)
				n = len(signal)
				# pad input stereo signal
				y_pad = fourier.fix_length(signal, n + fft_size // 2, axis=0)
				for channel in channels:
					# take FFT for each channel
					spectrum_complex = np.array(fourier.stft(y_pad[:, channel], n_fft=fft_size, step=hop))
					spectrum_db = to_dB(to_mag(spectrum_complex))
					gain_db_whole = np.zeros(spectrum_complex.shape, dtype=float)
					for drop in self.markers:
						frame_b = self.time_2_frame(drop.t - (drop.width / 2))
						frame_a = self.time_2_frame(drop.t + (drop.width / 2))
						# parametrize frame_surrounding as percentage, set from UI for selected or as preset, at least 1
						frame_surrounding = max(1, self.time_2_frame(drop.width * drop.surrounding))
						bin_l = self.freq_2_bin(drop.f - (drop.height / 2))
						bin_u = self.freq_2_bin(drop.f + (drop.height / 2))
	
						# take mean of left and right frames
						mag_before = np.mean(spectrum_db[bin_l:bin_u, frame_b-frame_surrounding:frame_b], axis=1)
						mag_after = np.mean(spectrum_db[bin_l:bin_u, frame_a:frame_a+frame_surrounding], axis=1)
	
						# interpolate the desired spectrum for each bin in the selected region
						fp_frames = np.linspace(frame_b, frame_a, num=frame_a-frame_b)
						fp_bins = np.linspace(bin_l, bin_u, num=bin_u-bin_l)
	
						interp = RegularGridInterpolator(((frame_b, frame_a), fp_bins), (mag_before, mag_after))
						mp_bins, mp_frames = np.meshgrid(fp_bins, fp_frames)  # 2D grid for interpolation
						fp_db = interp((mp_frames, mp_bins))
						fp_db = np.swapaxes(fp_db, 0, 1)
						# calculate boost to bring dropout up to desired volume
						gain_db = fp_db - spectrum_db[bin_l:bin_u, frame_b:frame_a]
						# take at least as much as the previous gain for each bin
						np.clip(gain_db, gain_db_whole[bin_l:bin_u, frame_b:frame_a], 255, out=gain_db)
						# store boost in mask
						gain_db_whole[bin_l:bin_u, frame_b:frame_a] = gain_db
					# correct fft data with boost
					spectrum_complex *= to_fac(gain_db_whole)
					# take iFFT
					output[:, channel] = fourier.istft(spectrum_complex, length=n, hop_length=hop)

				io_ops.write_file(file_path, output, sr, len(channels), suffix=f"_drops{self.props.output_widget.suffix}")

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
					spec = self.spectra[0]
					t_0, t_1, f_lower, f_upper = spec.get_times_freqs(a, b)
					# direct mode
					if "Shift" in event.modifiers:
						marker = markers.DropoutSample(self, a, b)
						self.props.undo_stack.push(AddAction((marker,)))
					# batch detection
					elif "Alt" in event.modifiers:
						imdata = spec.fft_storage[spec.key]
						imdata = np.array(imdata)
						imdata = to_dB(imdata)
						# which range should dropouts be detected in?
						frame_b = self.time_2_frame(t_0)
						frame_a = self.time_2_frame(t_1)
						bin_l = self.freq_2_bin(f_lower)
						bin_u = self.freq_2_bin(f_upper)
						# take the mean volume across this band
						vol = np.mean(imdata[bin_l:bin_u, frame_b:frame_a], axis=0)
						half_width = self.props.dropout_widget.width / 1000 / 2
						frames_half_width = self.time_2_frame(half_width)
						vol_lt = savgol_filter(vol, frames_half_width*12, 5)
						vol_st = savgol_filter(vol, frames_half_width, 5)

						# detect valleys in the volume curve
						peaks, properties = scipy.signal.find_peaks(-vol, height=None, threshold=None, distance=None,
																	prominence=10.0-self.props.dropout_widget.sensitivity, wlen=None, rel_height=0.5,
																	plateau_size=None)
						# plt.vlines(peaks, -100, 1, colors=(0, 1.0, 0, 0.2), linestyles='--', label='peaks',)
						# plt.plot(vol, label='raw vol',)
						# plt.plot(vol_lt, label='vol_lt',)
						# plt.plot(vol_st, label='vol_st',)
						# plt.show()
						_markers = []
						for f_peak in peaks:
							# get intersection of variously interpolated curves
							t_center = self.frame_2_time(frame_b + f_peak)
							try:
								# optimize hw by measuring the width of a parabola fit through the dropout and the low-passed signal
								# get frames for fitting the parabola
								f_qw = self.time_2_frame(half_width / 4)
								f_before = f_peak - f_qw
								f_after = f_peak + f_qw
								xp = np.arange(f_before, f_after)
								# fit the parabola
								parabola_coeff = np.polyfit(xp, vol_st[f_before:f_after], 2)
								parabola = np.poly1d(parabola_coeff, r=False, variable=None)
								# get frames for sampling the parabola
								f_hw = self.time_2_frame(half_width)
								f_before = f_peak - f_hw
								f_after = f_peak + f_hw
								xp = np.arange(f_before, f_after)
								fp = parabola(xp)
								# get intersection and calculate width from peaks
								f_intersection = scipy.signal.argrelmin(np.abs(fp-vol_lt[f_before:f_after]))[0]
								assert len(f_intersection)==2
								# plt.plot(xp, fp, label='parab')
								# plt.plot(f_intersection+f_before, fp[f_intersection], 'ro', label='x',)
								half_width = self.frame_2_time(f_intersection[1]-f_intersection[0])
							except:
								logging.exception(f"Could not refine width at peak {f_peak}")
							t_before = t_center-half_width
							t_after = t_center+half_width
							marker = markers.DropoutSample(self, (t_before, f_lower), (t_after, f_upper))
							_markers.append(marker)
						self.props.undo_stack.push(AddAction(_markers))



if __name__ == '__main__':
	widgets.startup(MainWindow)
