import logging

import numpy as np
from vispy import scene
from scipy import interpolate

# custom modules
from util import wow_detection, filters
from util.wow_detection import nan_helper

line_settings = {"method": 'gl', "antialias": False, "width": 2.0}
wide_line_settings = {"method": 'gl', "antialias": False, "width": 5.0}
colors_val = [(1, 0, 0, 1), (0.5, 0.5, 0.5, 1), (0, 1, 0, 1)]
colors_range = (-1, 0, 1)
color_interp = interpolate.CubicSpline(colors_range, colors_val, bc_type="clamped")


def get_colors(corr):
	color = color_interp(corr)
	return color.clip(0.0, 1.0)


class BaseMarker:
	"""Stores and visualizes a trace fragment, including its speed offset."""

	def __init__(self, vispy_canvas, color_def, color_sel):
		self.selected = False
		self.vispy_canvas = vispy_canvas
		self.visuals = []
		self.color_def = color_def
		self.color_sel = color_sel
		self.spec_center = (0, 0)
		self.speed_center = (0, 0)
		self.offset = None
		self.container = vispy_canvas.markers
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
			self.vispy_canvas.deselect_all()
		self.toggle()

	def remove(self):
		for v in self.visuals:
			v.parent = None
		# note: this has to search the list
		self.container.remove(self)

	@classmethod
	def from_cfg(cls, canvas, *args):
		return cls(canvas, *args)


class RegLine(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	def __init__(self, vispy_canvas, t0, t1, amplitude, omega, phase, offset):

		color_def = (0, 0, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_sel)
		# the extents on which this regression operated
		self.t0 = t0
		self.t1 = t1
		# here, the reg values are most accurate
		self.t_center = (t0 + t1) / 2
		self.speed_center = np.array((self.t_center, 0))
		self.spec_center = np.array((self.t_center, 2000))

		# the following is more or less duped in the tracer - resolve?
		# clipped_times = self.get_time_range(vispy_canvas)
		samples_per_second = 200
		clipped_times = np.linspace(t0, t1, num=int((t1-t0)*samples_per_second))

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
		speed = self.amplitude * np.sin(self.omega * clipped_times + self.phase)
		self.speed_data = np.stack((clipped_times, speed), axis=-1)
		self.visuals.append(scene.Line(pos=self.speed_data, color=color_def, **line_settings))

		self.spec_data = np.stack((clipped_times, self.speed_to_hz(speed), np.ones(len(clipped_times), dtype=np.float32) * -2), axis=-1)
		self.visuals.append(scene.Line(pos=self.spec_data, color=color_def, **line_settings))
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform

	def get_time_range(self, vispy_canvas):
		speed_curve = vispy_canvas.master_speed.get_linspace()
		times = speed_curve[:, 0]
		# speeds = speed_curve[:, 1]
		# which part to process?
		period = times[1] - times[0]
		ind_start = int(self.t0 / period)
		ind_stop = int(self.t1 / period)
		clipped_times = times[ind_start:ind_stop]
		return clipped_times

	def speed_to_hz(self, speed):
		return np.power(2, speed + np.log2(2000))

	def set_offset(self, a, b):
		# user manipulation: custom amplitude for sample
		self.amplitude *= (b / a)
		self.speed_data[:, 1] *= (b / a)
		self.visuals[0].set_data(pos=self.speed_data)
		self.spec_data[:, 1] = self.speed_to_hz(self.speed_data[:, 1])
		self.visuals[1].set_data(pos=self.spec_data)

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

	def to_cfg(self):
		return self.t0, self.t1, self.amplitude, self.omega, self.phase, self.offset


class TraceLine(BaseMarker):
	"""Stores and visualizes a trace fragment, including its speed offset."""

	def __init__(self, vispy_canvas, times, freqs, offset=None, auto_align=False):

		color_def = (1, 0, 0, .5)
		color_sel = (0, 1, 0, .5)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_sel)
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
		self.visuals.append(scene.Line(pos=self.speed_data, color=color_def, **line_settings))
		self.visuals[0].set_gl_state('additive')

		# create the spectral visualization
		# could also do a stack here; note the z coordinate!
		spec_data = np.stack((self.times, self.freqs, np.ones(len(self.times), dtype=np.float32) * -2), axis=-1)
		self.visuals.append(scene.Line(pos=spec_data, color=color_def, **line_settings))
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform

	@property
	def start(self):
		return self.times[0]

	@property
	def end(self):
		return self.times[-1]

	def toggle(self):
		"""Toggle this line's selection state"""
		super().toggle()

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

	def to_cfg(self):
		return list(self.times), list(self.freqs), self.offset


class PanSample(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	def __init__(self, vispy_canvas, a, b, pan):
		color_def = (1, 1, 1, .5)
		color_sel = (0, 1, 0, 1)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_sel)
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

	def set_color(self, c):
		for v in self.visuals:
			v.color = c

	def to_cfg(self):
		return self.a[0], self.a[1], self.b[0], self.b[1], self.pan

	@classmethod
	def from_cfg(cls, canvas, a0, a1, b0, b1, pan):
		return cls(canvas, (a0, a1), (b0, b1), pan)


class DropoutSample(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	color_def = (1, 1, 1, .5)
	color_sel = (0, 0, 1, 1)

	def __init__(self, vispy_canvas, a, b, d=None, corr=0):
		BaseMarker.__init__(self, vispy_canvas, self.color_def, self.color_sel)
		self.a = a
		self.b = b
		self.corr = corr
		if d is None:
			self.d = vispy_canvas.spectra[-1].offset
		else:
			self.d = d
		self.width = abs(a[0] - b[0])
		self.t = (a[0] + b[0]) / 2
		self.f = (a[1] + b[1]) / 2
		self.height = abs(a[1] - b[1])
		self.speed_center = np.array((self.t, self.d))
		self.spec_center = np.array((self.t, self.f))
		# create & store speed visual
		r = 0.1
		rect = scene.Rectangle(center=self.speed_center, width=r, height=r, radius=0)
		rect.color = self.color_def
		self.visuals.append(rect)

		# create & store spec visual
		rect = scene.Rectangle(center=self.spec_center, width=self.width, height=self.height, radius=0)
		rect.color = self.color_def
		rect.transform = vispy_canvas.spectra[-1].mel_transform
		rect.set_gl_state('additive')
		self.visuals.append(rect)

	def set_offset(self, d):
		self.d += d
		speed_vis = self.visuals[0]
		speed_vis.pos[:, 1] += d
		speed_vis.pos = np.array(speed_vis.pos)

	def set_color(self, c):
		for v in self.visuals:
			v.color = c

	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.set_color(self.color_sel)

	def to_cfg(self):
		return self.a[0], self.a[1], self.b[0], self.b[1], self.d, self.corr

	@classmethod
	def from_cfg(cls, canvas, a0, a1, b0, b1, d, corr):
		return cls(canvas, (a0, a1), (b0, b1), d, corr)


class LagSample(BaseMarker):
	"""Stores a single sinc regression's data and displays it"""

	color_def = (1, 1, 1, .5)
	color_sel = (0, 0, 1, 1)

	def __init__(self, vispy_canvas, a, b, d=None, corr=0):
		BaseMarker.__init__(self, vispy_canvas, self.color_def, self.color_sel)
		self.a = a
		self.b = b
		self.corr = corr
		if d is None:
			self.d = vispy_canvas.spectra[-1].offset
		else:
			self.d = d
		self.width = abs(a[0] - b[0])
		self.t = (a[0] + b[0]) / 2
		self.f = (a[1] + b[1]) / 2
		self.height = abs(a[1] - b[1])
		self.speed_center = np.array((self.t, self.d))
		self.spec_center = np.array((self.t, self.f))
		# create & store speed visual
		r = 0.1
		rect = scene.Rectangle(center=self.speed_center, width=r, height=r, radius=0)
		rect.color = self.color_def
		self.visuals.append(rect)

		# create & store spec visual
		rect = scene.Rectangle(center=self.spec_center, width=self.width, height=self.height, radius=0)
		rect.color = self.color_def
		rect.transform = vispy_canvas.spectra[-1].mel_transform
		rect.set_gl_state('additive')
		self.visuals.append(rect)

	def set_offset(self, d):
		self.d += d
		speed_vis = self.visuals[0]
		speed_vis.pos[:, 1] += d
		speed_vis.pos = np.array(speed_vis.pos)

	def set_color(self, c):
		for v in self.visuals:
			v.color = c

	def select(self):
		"""Toggle this line's selection state"""
		self.selected = True
		self.set_color(self.color_sel)

	def to_cfg(self):
		return self.a[0], self.a[1], self.b[0], self.b[1], self.d, self.corr

	@classmethod
	def from_cfg(cls, canvas, a0, a1, b0, b1, d, corr):
		return cls(canvas, (a0, a1), (b0, b1), d, corr)


class AzimuthLine(BaseMarker):

	def __init__(self, vispy_canvas, times, lags, corrs, lower, upper):

		color_def = (1, 0, 0, .5)
		color_sel = (0, 1, 0, .5)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_sel)
		self.times = np.asarray(times)
		self.lags = np.asarray(lags)
		self.corrs = np.asarray(corrs)
		self.lower = lower
		self.upper = upper

		self.d = np.mean(self.lags)
		self.corr = np.mean(self.corrs)
		self.width = self.end - self.start
		self.height = abs(upper - lower)

		# calculate the centers
		mean_times = np.mean(self.times)
		self.spec_center = np.array((mean_times, (lower + upper)/2))
		self.speed_center = np.array((mean_times, np.mean(self.lags)))

		# create the speed curve visualization
		self.lags_data = np.stack((self.times, self.lags), axis=-1)
		color_line = get_colors(self.corrs)
		self.visuals.append(scene.Line(pos=self.lags_data, color=color_line, **line_settings))
		# self.visuals[0].set_gl_state('additive')

		# create the spectral visualization
		rect = scene.Rectangle(center=self.spec_center, width=self.width, height=self.height, radius=0)
		rect.color = self.color_def
		rect.transform = vispy_canvas.spectra[-1].mel_transform
		self.visuals.append(rect)
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform

	@property
	def start(self):
		return self.times[0]

	@property
	def end(self):
		return self.times[-1]

	@property
	def t(self):
		return (self.start+self.end)/2

	def set_color(self, c):
		# don't override the  line color
		# api inconsistency
		# self.visuals[0].set_data(color=c)
		self.visuals[1].color = c

	def toggle(self):
		"""Toggle this line's selection state"""
		super().toggle()
		# todo - do stuff
		# self.vispy_canvas.props.tracing_widget.target_s.setValue(np.mean(target_freqs))

	def to_cfg(self):
		return list(self.times), list(self.lags), list(self.corrs), float(self.lower), float(self.upper)


class BaseLine:
	def __init__(self, vispy_canvas, color=(1, 0, 0, .5)):
		self.vispy_canvas = vispy_canvas

		# create the speed curve visualization
		self.data = np.zeros((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.empty = np.array(self.data)
		self.bands = (0, 9999999)
		self.line_speed = scene.Line(pos=self.data, color=color, **wide_line_settings)
		self.line_speed.parent = self.vispy_canvas.speed_view.scene
		self.line_speed.set_gl_state('additive')

	def show(self):
		self.line_speed.parent = self.vispy_canvas.speed_view.scene

	def hide(self):
		self.line_speed.parent = None

	def get_times(self):
		num = int(self.vispy_canvas.duration * self.marker_sr)
		# get the times at which the average should be sampled
		return np.linspace(0, self.vispy_canvas.duration, num=num)

	@property
	def marker_sr(self):
		"""Amount of marker samples per second"""
		return self.vispy_canvas.sr / self.vispy_canvas.hop

	def get_linspace(self):
		"""Convert the log2 spaced speed curve back into linspace for further processing"""
		out = np.array(self.data)
		np.power(2, out[:, 1], out[:, 1])
		return out

	def filter_bandpass(self, samples_in):
		# bandpass filter the output
		lowcut, highcut = sorted(self.bands)
		samples = filters.butter_bandpass_filter(samples_in, lowcut, highcut, self.marker_sr, order=3)
		return samples

	def sample_lines(self, times, lines_times, lines_values):
		# create the array for sampling
		out = np.zeros((len(times), len(lines_times)), dtype=np.float32)
		# lerp and sample all lines, use NAN for missing parts
		for i, (line_times, line_values) in enumerate(zip(lines_times, lines_values)):
			line_sampled = np.interp(times, line_times, line_values, left=np.nan, right=np.nan)
			out[:, i] = line_sampled
		# take the mean and ignore nans
		return np.nanmean(out, axis=1)

	def update(self):
		pass

	def update_bands(self, bands):
		self.bands = bands
		self.update()


class MasterSpeedLine(BaseLine):
	"""Stores and displays the average, ie. master speed curve."""

	def update(self):
		if self.vispy_canvas.lines:
			times = self.get_times()
			lines_times = [line.times for line in self.vispy_canvas.lines]
			lines_values = [line.speed for line in self.vispy_canvas.lines]
			mean_with_nans = self.sample_lines(times, lines_times, lines_values)
			# lerp over nan areas
			wow_detection.interp_nans(mean_with_nans)
			self.data = np.stack((times, self.filter_bandpass(mean_with_nans)), axis=-1)
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

	def __repr__(self):
		return f"{type(self).__name__}[{len(self.vispy_canvas.lines)}]"


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
		if self.vispy_canvas.markers:
			# sort before we start
			self.vispy_canvas.markers.sort(key=lambda tup: tup.t)

			# get the times at which the average should be sampled
			times = self.get_times()
			sample_times = [sample.t for sample in self.vispy_canvas.markers]
			sample_pans = [sample.pan for sample in self.vispy_canvas.markers]
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

	def interp(self, times, keys, values):
		# ensure that k doesn't exceed the order possible to infer from amount of samples
		if len(keys) == 0:
			return np.interp(times, (0,), (0,))
		elif len(keys) == 1:
			return np.interp(times, keys, values)
		else:
			# using bezier splines
			k = min(self.smoothing, len(keys)-1)
			lags_s = interpolate.InterpolatedUnivariateSpline(keys, values, k=k)
			return lags_s(times)

	def get_times(self):
		dur = self.vispy_canvas.duration
		lag, corr = self.sample_at((dur,))
		# print(lag)
		dur += lag[0]
		num = int(dur * self.marker_sr)
		# get the times at which the average should be sampled
		return np.linspace(0, dur, num=num)

	def sample_at(self, times):
		self.vispy_canvas.markers.sort(key=lambda marker: marker.t)
		lags = self.vispy_canvas.lags
		sample_times = [sample.t for sample in lags]
		sample_lags = [sample.d for sample in lags]
		sample_corrs = [sample.corr for sample in lags]
		azimuths = self.vispy_canvas.azimuths
		azimuths_times = [sample.times for sample in azimuths]
		azimuths_lags = [sample.lags for sample in azimuths]
		azimuths_corrs = [sample.corrs for sample in azimuths]
		azimuths_sampled_with_nans = self.sample_lines(times, azimuths_times, azimuths_lags)
		corrs_sampled_with_nans = self.sample_lines(times, azimuths_times, azimuths_corrs)
		lags_sampled = self.interp(times, sample_times, sample_lags)
		corrs_sampled = self.interp(times, sample_times, sample_corrs)
		nans, x = nan_helper(azimuths_sampled_with_nans)
		azimuths_sampled_with_nans[nans] = lags_sampled[nans]
		corrs_sampled_with_nans[nans] = corrs_sampled[nans]
		return azimuths_sampled_with_nans, corrs_sampled_with_nans

	def update(self):
		logging.info(f"Updating sync line")
		color = (1, 0, 0, 1)
		if self.vispy_canvas.markers:
			times = self.get_times()
			try:
				lag, corr = self.sample_at(times)
				lag = self.filter_bandpass(lag)
				self.data = np.stack((times, lag), axis=-1)
				# map the correlation to color range
				color = get_colors(corr)
			except:
				logging.exception(f"LagLine.update failed")
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data, color=color)


class DropoutLine(LagLine):
	"""Stores and displays the average, ie. master speed curve."""

	def __init__(self, vispy_canvas):
		super().__init__(vispy_canvas)
		self.smoothing = 3

	def interp(self, times, keys, values):
		# ensure that k doesn't exceed the order possible to infer from amount of samples
		if len(keys) == 0:
			return np.interp(times, (0,), (0,))
		elif len(keys) == 1:
			return np.interp(times, keys, values)
		else:
			# using bezier splines
			k = min(self.smoothing, len(keys)-1)
			lags_s = interpolate.InterpolatedUnivariateSpline(keys, values, k=k)
			return lags_s(times)

	def get_times(self):
		dur = self.vispy_canvas.duration
		num = int(dur * self.marker_sr)
		# get the times at which the average should be sampled
		return np.linspace(0, dur, num=num)

	def sample_at(self, times):
		return np.zeros(len(times)), np.zeros(len(times))

	def update(self):
		logging.info(f"Updating sync line")
		color = (1, 0, 0, 1)
		if self.vispy_canvas.markers:
			times = self.get_times()
			try:
				lag, corr = self.sample_at(times)
				lag = self.filter_bandpass(lag)
				self.data = np.stack((times, lag), axis=-1)
				# map the correlation to color range
				color = get_colors(corr)
			except:
				logging.exception(f"LagLine.update failed")
		else:
			self.data = self.empty
		self.line_speed.set_data(pos=self.data, color=color)
