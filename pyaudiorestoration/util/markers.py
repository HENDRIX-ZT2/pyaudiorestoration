import logging

import numpy as np
from vispy import scene
from scipy import interpolate

# custom modules
from util import wow_detection, filters


class BaseMarker:
	"""Stores and visualizes a trace fragment, including its speed offset."""

	def __init__(self, vispy_canvas, container, color_def, color_sel):
		self.selected = False
		self.vispy_canvas = vispy_canvas
		self.visuals = []
		self.color_def = color_def
		self.color_sel = color_sel
		self.spec_center = (0, 0)
		self.speed_center = (0, 0)
		self.offset = None
		self.container = container
		self.parents = (self.vispy_canvas.speed_view.scene, self.vispy_canvas.spec_view.scene)

	def initialize(self):
		"""Called when first created, or revived via undo."""
		self.show()
		self.container.append(self)

	def show(self):
		for v, p in zip(self.visuals, self.parents):
			v.parent = p

	def hide(self):
		for v, p in zip(self.visuals, self.parents):
			v.parent = None
		self.deselect()

	def set_color(self, c):
		for v in self.visuals:
			v.set_data(color=c)

	def deselect(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = False
		self.set_color(self.color_def)

	def select(self):
		"""Select this line"""
		self.selected = True
		self.set_color(self.color_sel)

	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()

	def select_handle(self, multi=False):
		if not multi:
			# for trace in self.vispy_canvas.regs+self.vispy_canvas.lines:
			for trace in self.container:
				trace.deselect()
		self.toggle()

	def remove(self):
		for v in self.visuals:
			v.parent = None
		# note: this has to search the list
		self.container.remove(self)


class RegLine(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	def __init__(self, vispy_canvas, t0, t1, amplitude, omega, phase, offset):

		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.regs, color_def, color_sel)
		# the extents on which this regression operated
		self.t0 = t0
		self.t1 = t1
		# here, the reg values are most accurate
		self.t_center = (t0 + t1) / 2
		self.speed_center = np.array((self.t_center, 0))
		self.spec_center = np.array((self.t_center, 2000))

		# the following is more or less duped in the tracer - resolve?
		speed_curve = vispy_canvas.master_speed.get_linspace()

		times = speed_curve[:, 0]
		# speeds = speed_curve[:, 1]

		# which part to process?
		period = times[1] - times[0]
		ind_start = int(self.t0 / period)
		ind_stop = int(self.t1 / period)
		clipped_times = times[ind_start:ind_stop]

		# set the properties
		self.amplitude = amplitude
		self.omega = omega
		self.phase = phase
		self.offset = offset

		# some conventions are needed
		# correct the amplitude & phase so we can interpolate properly
		if self.amplitude < 0:
			self.amplitude *= -1
			self.phase += np.pi
		# phase should be in 0 < x< 2pi
		# this is not completely guaranteed by this
		# if self.phase < 0:
		# self.phase += (2*np.pi)
		# if self.phase > 2*np.pi:
		# self.phase -= (2*np.pi)
		# self.phase = self.phase % (2*np.pi)

		# create the speed curve visualization
		self.speed_data = np.stack(
			(clipped_times, self.amplitude * np.sin(self.omega * clipped_times + self.phase)), axis=-1)
		# sine_on_hz = np.power(2, sine + np.log2(2000))
		self.visuals.append(scene.Line(pos=self.speed_data, color=(0, 0, 1, .5), method='gl'))
		self.initialize()

	def set_offset(self, a, b):
		# user manipulation: custom amplitude for sample
		self.amplitude *= (b / a)
		self.speed_data[:, 1] *= (b / a)
		self.visuals[0].set_data(pos=self.speed_data)

	def select(self):
		"""Deselect this line, ie. restore its colors to their original state"""
		self.selected = True
		self.set_color(self.color_sel)
		# set the offset in the ui
		self.vispy_canvas.props.tracing_widget.phase_s.setValue(self.offset)

	def update_phase(self, v):
		"""Adjust this regressions's phase offset according to the UI input."""
		if self.selected:
			self.offset = v


class TraceLine(BaseMarker):
	"""Stores and visualizes a trace fragment, including its speed offset."""

	def __init__(self, vispy_canvas, times, freqs, offset=None, auto_align=False):

		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.lines, color_def, color_sel)
		self.times = np.asarray(times)
		self.freqs = np.asarray(freqs)

		# note: the final, output speed curve output should be linscale and centered on 1
		self.speed = np.log2(freqs)
		self.speed -= np.mean(self.speed)
		# we don't want to overwrite existing offsets loaded from files
		if offset is None:
			if not auto_align:
				offset = 0
			else:
				# create the array for sampling
				out = np.ones((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
				# lerp and sample all lines, use NAN for missing parts
				for i, line in enumerate(self.vispy_canvas.lines):
					line_sampled = np.interp(times, line.times, line.speed, left=np.nan, right=np.nan)
					out[:, i] = line_sampled
				# take the mean and ignore nans
				mean_with_nans = np.nanmean(out, axis=1)
				offset = np.nanmean(mean_with_nans - self.speed)
				offset = 0 if np.isnan(offset) else offset
		self.offset = offset
		self.speed += offset

		# calculate the centers
		mean_times = np.mean(self.times)
		self.spec_center = np.array((mean_times, np.mean(self.freqs)))
		self.speed_center = np.array((mean_times, np.mean(self.speed)))

		# create the speed curve visualization
		self.speed_data = np.stack((self.times, self.speed), axis=-1)
		self.visuals.append(scene.Line(pos=self.speed_data, color=color_def, method='gl'))

		# create the spectral visualization
		# could also do a stack here; note the z coordinate!
		spec_data = np.stack((self.times, self.freqs, np.ones(len(self.times), dtype=np.float32) * -2), axis=-1)
		self.visuals.append(scene.Line(pos=spec_data, color=color_def, method='gl'))
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform

		self.initialize()

	@property
	def start(self):
		return self.times[0]

	@property
	def end(self):
		return self.times[-1]

	def toggle(self):
		"""Toggle this line's selection state"""
		if self.selected:
			self.deselect()
		else:
			self.select()

		# TODO: evaluate performance penalty of looping here!

		# go over all selected markers
		target_freqs = [marker.spec_center[1] / (2 ** marker.offset) for marker in self.container if marker.selected]
		# to set the offset properly, we need to know
		# we have a current mean Hz + offset
		# and a target mean Hz

		# offset works in log2 scale
		# offset +1 halfes the final frequency
		# offset -1 doubles the final frequency
		self.vispy_canvas.props.tracing_widget.target_s.setValue(np.mean(target_freqs))

	def lock_to(self, f):
		if self.selected:
			# print("lock_to")
			# print(f, np.log2(f))
			# print(self.spec_center[1], np.log2(self.spec_center[1]))
			# print()
			offset = np.log2(self.spec_center[1]) - np.log2(f)

			old_offset = self.offset
			self.offset = offset
			self.speed_center[1] += offset - old_offset
			self.speed += offset - old_offset
			self.speed_data[:, 1] = self.speed
			self.visuals[0].set_data(pos=self.speed_data)

	def set_offset(self, a, b):
		offset = b - a
		self.offset += offset
		self.speed_center[1] += offset
		self.speed += offset
		self.speed_data[:, 1] = self.speed
		self.visuals[0].set_data(pos=self.speed_data)


class PanSample(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	def __init__(self, vispy_canvas, a, b, pan):
		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.pan_samples, color_def, color_sel)
		self.parents = (self.vispy_canvas.spec_view.scene,)

		self.a = a
		self.b = b
		self.t = (a[0] + b[0]) / 2
		self.width = abs(a[0] - b[0])
		self.f = (a[1] + b[1]) / 2
		self.height = abs(a[1] - b[1])
		self.spec_center = (self.t, self.f)
		self.speed_center = (self.t, pan)

		self.pan = pan

		# create & store visual
		rect = scene.Rectangle(
			center=(self.t, self.f), width=self.width, height=self.height, radius=0,
			parent=vispy_canvas.spec_view.scene)
		rect.color = (1, 1, 1, .5)
		rect.transform = vispy_canvas.spectra[-1].mel_transform
		rect.set_gl_state('additive')
		self.visuals.append(rect)

		self.initialize()

	def set_color(self, c):
		for v in self.visuals:
			v.color = c


class LagSample(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	color_def = (1, 1, 1, .5)
	color_sel = (0, 0, 1, 1)

	def __init__(self, vispy_canvas, a, b, d=None, corr=None):
		BaseMarker.__init__(self, vispy_canvas, vispy_canvas.lag_samples, self.color_def, self.color_sel)
		self.parents = (self.vispy_canvas.spec_view.scene,)

		self.a = a
		self.b = b
		self.corr = corr
		if d is None:
			self.d = vispy_canvas.spectra[-1].delta
		else:
			self.d = d
		self.width = abs(a[0] - b[0])
		self.t = (a[0] + b[0]) / 2
		self.f = (a[1] + b[1]) / 2
		self.height = abs(a[1] - b[1])
		self.spec_center = (self.t, self.f)
		self.speed_center = (self.t, self.d)
		# create & store visual
		rect = scene.Rectangle(
			center=(self.t, self.f), width=self.width, height=self.height, radius=0,
			parent=vispy_canvas.spec_view.scene)
		rect.color = self.color_def
		rect.transform = vispy_canvas.spectra[-1].mel_transform
		rect.set_gl_state('additive')
		self.visuals.append(rect)

		# create & store visual
		r = 0.1
		rect = scene.Rectangle(
			center=(self.t, self.d), width=r, height=r, radius=0,
			parent=vispy_canvas.speed_view.scene)
		rect.color = self.color_def
		self.visuals.append(rect)

		self.initialize()

	def set_color(self, c):
		for v in self.visuals:
			v.color = c

	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.set_color(self.color_sel)


class BaseLine:
	def __init__(self, vispy_canvas, color=(1, 0, 0, .5)):
		self.vispy_canvas = vispy_canvas

		# create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.empty = np.array(self.data)
		self.bands = (0, 9999999)
		self.line_speed = scene.Line(pos=self.data, color=color, method='gl')
		self.line_speed.parent = self.vispy_canvas.speed_view.scene

	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene

	def hide(self):
		self.line_speed.parent = None

	def get_times(self):
		# num = self.vispy_canvas.spectra[0].num_ffts
		num = int(self.vispy_canvas.duration * self.vispy_canvas.sr / self.vispy_canvas.hop)
		# get the times at which the average should be sampled
		return np.linspace(0, self.vispy_canvas.duration, num=num)

	def get_linspace(self):
		"""Convert the log2 spaced speed curve back into linspace for further processing"""
		out = np.array(self.data)
		np.power(2, out[:, 1], out[:, 1])
		return out


class MasterSpeedLine(BaseLine):
	"""Stores and displays the average, ie. master speed curve."""

	def update(self):
		if self.vispy_canvas.lines:
			times = self.get_times()
			# create the array for sampling
			out = np.zeros((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
			# lerp and sample all lines, use NAN for missing parts
			for i, line in enumerate(self.vispy_canvas.lines):
				line_sampled = np.interp(times, line.times, line.speed, left=np.nan, right=np.nan)
				out[:, i] = line_sampled
			# take the mean and ignore nans
			mean_with_nans = np.nanmean(out, axis=1)
			# lerp over nan areas
			nans, x = wow_detection.nan_helper(mean_with_nans)
			mean_with_nans[nans] = np.interp(x(nans), x(~nans), mean_with_nans[~nans])

			# bandpass filter the output
			fs = self.vispy_canvas.sr / self.vispy_canvas.hop
			lowcut, highcut = sorted(self.bands)
			samples = filters.butter_bandpass_filter(mean_with_nans, lowcut, highcut, fs, order=3)

			self.data = np.stack((times, samples), axis=-1)
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data)

	def get_overlapping_lines(self):
		"""Return a list of lists with overlapping lines each"""
		if not self.vispy_canvas.lines:
			return
		# sort the lines by their start time
		sorted_lines = sorted(self.vispy_canvas.lines, key=lambda line_w: line_w.start)
		# start with just the first line
		merged = [
			[sorted_lines[0]]
		]
		# go over the other lines
		for higher_line in sorted_lines[1:]:
			current_group = merged[-1]
			upper_bound = max([line_w.end for line_w in current_group])
			# example data
			# 1---  3------------
			#   2-----    4-
			# this starts before any of the lines already merged end, it is part of this group
			if higher_line.start <= upper_bound:
				current_group.append(higher_line)
			# else it is a new group
			else:
				merged.append([higher_line, ])
		return merged


class MasterRegLine(BaseLine):
	"""Stores and displays the average, ie. master speed curve."""

	def update(self):
		if self.vispy_canvas.regs:

			# here we want to interpolate smoothly between the regressed sines around their centers
			# https://stackoverflow.com/questions/11199509/sine-wave-that-slowly-ramps-up-frequency-from-f1-to-f2-for-a-given-time
			# https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time

			times = self.get_times()

			# sort the regressions by their time
			self.vispy_canvas.regs.sort(key=lambda tup: tup.t_center)

			pi2 = 2 * np.pi
			t_centers = []
			amp_centers = []
			phi_centers = []
			for i, reg in enumerate(self.vispy_canvas.regs):
				if i == 0:
					phi_centers.append(reg.omega * times[0] + reg.phase % pi2 + reg.offset * pi2)
					t_centers.append(times[0])
					amp_centers.append(reg.amplitude)
				phi_centers.append(reg.omega * reg.t_center + reg.phase % pi2 + reg.offset * pi2)
				t_centers.append(reg.t_center)
				amp_centers.append(reg.amplitude)
				if i == len(self.vispy_canvas.regs) - 1:
					phi_centers.append(reg.omega * times[-1] + reg.phase % pi2 + reg.offset * pi2)
					t_centers.append(times[-1])
					amp_centers.append(reg.amplitude)
			sine_curve = np.sin(np.interp(times, t_centers, phi_centers))
			amplitudes_sampled = np.interp(times, t_centers, amp_centers)

			# create the speed curve visualization, boost it a bit to distinguish from the raw curves
			self.data = np.stack((times, 1.5 * amplitudes_sampled * sine_curve), axis=-1)
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data)


class PanLine(BaseLine):
	"""Stores and displays the average, ie. master speed curve."""

	def update(self):
		if self.vispy_canvas.pan_samples:
			# sort before we start
			self.vispy_canvas.pan_samples.sort(key=lambda tup: tup.t)

			# get the times at which the average should be sampled
			times = self.get_times()
			sample_times = [sample.t for sample in self.vispy_canvas.pan_samples]
			sample_pans = [sample.pan for sample in self.vispy_canvas.pan_samples]
			pan = np.interp(times, sample_times, sample_pans)
			self.data = np.stack((times, pan), axis=-1)
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data)


class LagLine(BaseLine):
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		super().__init__(vispy_canvas)
		self.smoothing = 3

	def sample_at(self, times):
		self.vispy_canvas.lag_samples.sort(key=lambda tup: tup.t)
		sample_times = [sample.t for sample in self.vispy_canvas.lag_samples]
		sample_lags = [sample.d for sample in self.vispy_canvas.lag_samples]

		if len(self.vispy_canvas.lag_samples) == 1:
			return np.interp(times, sample_times, sample_lags)
		else:
			# using bezier splines
			s = interpolate.InterpolatedUnivariateSpline(sample_times, sample_lags, k=self.smoothing)
			return s(times)

	def update(self):
		logging.info(f"Updating sync line")
		if self.vispy_canvas.lag_samples:
			times = self.get_times()
			try:
				lag = self.sample_at(times)
				self.data = np.stack((times, lag), axis=-1)
			except:
				logging.exception(f"LagLine.update failed")
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data)
