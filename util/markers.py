import logging

import numpy as np
import scipy
from vispy import scene
from scipy import interpolate

# custom modules
from util import wow_detection, filters
from util.filters import make_odd
from util.wow_detection import nan_helper, interp_nans

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
		self.parents = [view.scene for view in self.vispy_canvas.views]

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
		self.spec_center = np.array((self.t_center, np.mean(self.freqs)))
		self.speed_center = np.array((self.t_center, np.mean(self.speed)))

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
	def t_center(self):
		return np.mean(self.times)

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


class PointDraft(BaseMarker):
	"""A trace draft built from draggable control points, committed as a TraceLine"""

	# max distance in pixels to grab a control point for dragging
	GRAB_RADIUS = 12
	# cap on the preview curve samples, to keep dragging responsive on long selections
	MAX_PREVIEW = 5000

	def __init__(self, vispy_canvas):

		color_def = (1, 1, 0, .9)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_def)
		# control points in spectrum coordinates, in placement order
		self.points = []
		self.drag_index = None

		# create the speed curve preview
		self.visuals.append(scene.Line(pos=np.empty((0, 2)), color=color_def, **wide_line_settings))
		# create the spectral preview, the data is in Hz, so to visualize correctly,
		# it has to be mel'ed, like the trace lines
		self.visuals.append(scene.Line(pos=np.empty((0, 3)), color=color_def, **wide_line_settings))
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform
		# create the control point handles
		self.visuals.append(scene.visuals.Markers(pos=np.empty((0, 3)), face_color=color_def, size=10))

	def show(self):
		# the curve previews live in their own views, the handles only in the spectrum view
		self.visuals[0].parent = self.parents[0]
		self.visuals[1].parent = self.parents[1]
		self.visuals[2].parent = self.parents[1]

	def remove(self):
		# a draft is never part of the marker container, so only unparent the visuals
		for v in self.visuals:
			v.parent = None

	def add_point(self, t, f):
		self.points.append((t, f))
		self.rebuild()

	def move_point(self, index, t, f):
		self.points[index] = (t, f)
		self.rebuild()

	def nearest_point(self, click):
		"""Return the index of the control point under the pixel click position, if any"""
		if not self.points or click is None:
			return None
		pts_px = np.array([self.vispy_canvas.spectrum_to_px((t, f))[0:2] for t, f in self.points])
		d = np.sum((pts_px - np.asarray(click)[0:2]) ** 2, axis=1)
		ind = int(d.argmin())
		if d[ind] <= self.GRAB_RADIUS ** 2:
			return ind
		return None

	def sorted_points(self):
		"""The control points sorted by time, duplicates in time averaged out"""
		times = np.array([p[0] for p in self.points])
		freqs = np.array([p[1] for p in self.points])
		order = np.argsort(times)
		times, freqs = times[order], freqs[order]
		unique_times, inverse = np.unique(times, return_inverse=True)
		unique_freqs = np.array([freqs[inverse == i].mean() for i in range(len(unique_times))])
		return unique_times, unique_freqs

	def sample_curve(self, max_samples=None):
		"""Sample the interpolated curve through the control points at FFT frame resolution"""
		times, freqs = self.sorted_points()
		canvas = self.vispy_canvas
		frame_0 = max(0, canvas.time_2_frame(times[0]))
		frame_1 = max(frame_0 + 1, canvas.time_2_frame(times[-1]))
		num_samples = max(2, frame_1 - frame_0)
		if max_samples is not None:
			num_samples = min(num_samples, max_samples)
		out_times = np.linspace(canvas.frame_2_time(frame_0), canvas.frame_2_time(frame_1), num_samples)
		if len(times) > 1:
			# interpolate in log frequency, so the curve is smooth in speed
			spline = interpolate.CubicSpline(times, np.log2(freqs), bc_type="natural")
			out_freqs = np.power(2.0, spline(out_times))
		else:
			out_freqs = np.full(len(out_times), freqs[0])
		return out_times, out_freqs

	def rebuild(self):
		times, freqs = self.sample_curve(max_samples=self.MAX_PREVIEW)
		# update the speed curve preview
		speed_data = np.stack((times, np.log2(freqs)), axis=-1)
		self.visuals[0].set_data(pos=speed_data, color=self.color_def)
		# update the spectral preview
		spec_data = np.stack((times, freqs, np.ones(len(times), dtype=np.float32) * -2), axis=-1)
		self.visuals[1].set_data(pos=spec_data, color=self.color_def)
		# update the control point handles
		if self.points:
			pts = np.array([(t, f, -2.0) for t, f in self.points], dtype=np.float32)
		else:
			pts = np.empty((0, 3))
		self.visuals[2].set_data(pos=pts, face_color=self.color_def)

	def commit_data(self):
		"""The final times and freqs for a TraceLine, sampled like a traced track"""
		return self.sample_curve()

class Cursor(BaseMarker):
	"""Cursor for position of audio playback"""

	def __init__(self, vispy_canvas):

		color_def = (1, 1, 1, 1)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_def)

		self.time = 0.0
		self.visuals.append(scene.visuals.InfiniteLine(pos=self.time, color=color_def, **line_settings))
		self.visuals.append(scene.visuals.Line(pos=self.data, color=color_def, **line_settings))
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[1].transform = vispy_canvas.spectra[0].mel_transform

	@property
	def data(self):
		data = np.array([
			[self.time, self.vispy_canvas.sr/2, -2.0],
			[self.time, 0.0, -2.0],
		])
		return data

	def set_time(self, t):
		self.time = t
		self.visuals[0].set_data(pos=self.time)
		self.visuals[1].set_data(pos=self.data)


class SelectionBox(BaseMarker):
	"""Visualizes the mouse actions"""

	def __init__(self, vispy_canvas):

		color_def = (1, 1, 1, 0.5)
		BaseMarker.__init__(self, vispy_canvas, color_def, color_def)

		self.parents = (self.vispy_canvas.spec_view.scene, )
		self.time = 0.0
		self.width = 1.0
		self.height = 1.0
		self.default_pos = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
		poly = scene.Line(pos=self.default_pos, color=color_def, connect='strip')
		self.visuals.append(poly)
		# the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.visuals[0].transform = vispy_canvas.spectra[0].mel_transform

	def update_coords(self, verts=None):
		if verts is None or len(verts) == 0:
			self.visuals[0].set_data(pos=self.default_pos)
		else:
			self.visuals[0].set_data(pos=verts)

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

	def __init__(self, vispy_canvas, a, b, surrounding=None, gain=None):
		BaseMarker.__init__(self, vispy_canvas, self.color_def, self.color_sel)
		self.a = a
		self.b = b
		self.corr = 0
		self.d = 0
		self.width = abs(a[0] - b[0])
		self.t = (a[0] + b[0]) / 2
		self.f = (a[1] + b[1]) / 2
		self.height = abs(a[1] - b[1])
		self.speed_center = np.array((self.t, self.d))
		self.spec_center = np.array((self.t, self.f))
		if surrounding is None:
			self.surrounding = self.vispy_canvas.props.dropout_widget.surrounding
		else:
			self.surrounding = surrounding
		if gain is None:
			self.gain = 0.0
		else:
			self.gain = gain
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

	def toggle(self):
		"""Toggle this line's selection state"""
		super().toggle()
		self.vispy_canvas.props.dropout_widget.surrounding = self.surrounding

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
		return self.a[0], self.a[1], self.b[0], self.b[1], self.surrounding, self.gain

	@classmethod
	def from_cfg(cls, canvas, a0, a1, b0, b1, surrounding, gain):
		return cls(canvas, (a0, a1), (b0, b1), surrounding, gain)


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
		self.lags_raw = np.asarray(lags)
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

	def update_reject(self, overlap, reject):
		# copy untouched data
		self.lags[:] = self.lags_raw
		# reject if correlation is too weak
		self.lags[np.abs(self.corrs) < reject] = np.NaN
		# lerp rejected lags
		interp_nans(self.lags)
		# filter outliers
		self.lags = scipy.ndimage.median_filter(self.lags, size=make_odd(overlap), footprint=None, output=None,
												mode='nearest', cval=0.0, origin=0, )
		self.d = np.mean(self.lags)
		self.lags_data = np.stack((self.times, self.lags), axis=-1)
		self.visuals[0].set_data(pos=self.lags_data)

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

	def auto_phase_offsets(self, regs):
		"""Automatically choose integer cycle offsets for the regressions."""
		if not regs:
			return

		pi2 = 2 * np.pi

		# Ensure chronological order
		regs.sort(key=lambda reg: reg.t_center)

		t = np.array([reg.t_center for reg in regs], dtype=float)
		omega = np.array([reg.omega for reg in regs], dtype=float)

		# Phase at each center before applying its cycle offset
		raw_phase = omega * t + np.array(
			[reg.phase % pi2 for reg in regs]
		)

		# Equivalent phases within one 2*pi cycle
		wrapped_phase = np.mod(raw_phase, pi2)

		# Unwrapped phase, starting from the first regression's offset
		phi = np.empty(len(regs), dtype=float)
		phi[0] = raw_phase[0] + regs[0].offset * pi2

		for i in range(1, len(regs)):
			dt = t[i] - t[i - 1]

			if dt <= 0:
				raise ValueError(
					"Regression centers must have distinct times."
				)

			# Predict phase advance by integrating the interpolated frequency
			expected_advance = (
					0.5 * (omega[i - 1] + omega[i]) * dt
			)
			predicted_phase = phi[i - 1] + expected_advance

			# Choose the equivalent phase closest to the prediction
			cycles = np.rint(
				(predicted_phase - wrapped_phase[i]) / pi2
			)
			phi[i] = wrapped_phase[i] + cycles * pi2

			# Convert the selected phase into your existing offset convention
			regs[i].offset = int(
				np.rint((phi[i] - raw_phase[i]) / pi2)
			)

			# Match the phase convention used by the existing update() method
			phi[i] = raw_phase[i] + regs[i].offset * pi2

	def update(self):
		if self.vispy_canvas.regs:

			# here we want to interpolate smoothly between the regressed sines around their centers
			# https://stackoverflow.com/questions/11199509/sine-wave-that-slowly-ramps-up-frequency-from-f1-to-f2-for-a-given-time
			# https://stackoverflow.com/questions/19771328/sine-wave-that-exponentialy-changes-between-frequencies-f1-and-f2-at-given-time

			times = self.get_times()

			# Sort regressions chronologically
			self.vispy_canvas.markers.sort(
				key=lambda reg: reg.t_center
			)

			# Automatically determine the cycle offsets
			self.auto_phase_offsets(self.vispy_canvas.regs)

			pi2 = 2 * np.pi
			t_centers = []
			amp_centers = []
			phi_centers = []

			for i, reg in enumerate(self.vispy_canvas.regs):
				if i == 0:
					phi_centers.append(
						reg.omega * times[0]
						+ reg.phase % pi2
						+ reg.offset * pi2
					)
					t_centers.append(times[0])
					amp_centers.append(reg.amplitude)

				phi_centers.append(
					reg.omega * reg.t_center
					+ reg.phase % pi2
					+ reg.offset * pi2
				)
				t_centers.append(reg.t_center)
				amp_centers.append(reg.amplitude)

				if i == len(self.vispy_canvas.regs) - 1:
					phi_centers.append(
						reg.omega * times[-1]
						+ reg.phase % pi2
						+ reg.offset * pi2
					)
					t_centers.append(times[-1])
					amp_centers.append(reg.amplitude)

			sine_curve = np.sin(
				np.interp(times, t_centers, phi_centers)
			)
			amplitudes_sampled = np.interp(
				times, t_centers, amp_centers
			)

			self.data = np.stack(
				(times, 1.5 * amplitudes_sampled * sine_curve),
				axis=-1,
			)
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
		# avoid crashing
		dur = abs(dur)
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
