# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# vispy: gallery 2

import numpy as np
from vispy.visuals.transforms import (arg_to_array, BaseTransform)
from vispy.visuals.axis import (AxisVisual, Ticker)
from vispy.visuals.text import TextVisual
from vispy.visuals.line import LineVisual
from vispy.visuals import CompoundVisual
from vispy.scene import PanZoomCamera, AxisWidget#, BaseCamera
from vispy.scene.widgets import Widget
#from vispy.visuals.transforms import STTransform
from vispy.geometry import Rect
from util import units

class PanZoomCameraExt(PanZoomCamera):
	""" Camera implementing 2D pan/zoom mouse interaction.

	For this camera, the ``scale_factor`` indicates the zoom level, and
	the ``center`` indicates the center position of the view.

	By default, this camera inverts the y axis of the scene. This usually
	results in the scene +y axis pointing upward because widgets (including
	ViewBox) have their +y axis pointing downward.

	Parameters
	----------
	rect : Rect
		A Rect object or 4-element tuple that specifies the rectangular
		area to show.
	aspect : float | None
		The aspect ratio (i.e. scaling) between x and y dimension of
		the scene. E.g. to show a square image as square, the aspect
		should be 1. If None (default) the x and y dimensions are scaled
		independently.
	**kwargs : dict
		Keyword arguments to pass to `BaseCamera`.

	Notes
	-----
	Interaction:

		* LMB: pan the view
		* RMB or scroll: zooms the view

	"""

	def viewbox_mouse_event(self, event):
		"""
		The SubScene received a mouse event; update transform
		accordingly.

		Parameters
		----------
		event : instance of Event
			The event.
		"""
		
		if event.handled or not self.interactive:
			return

		if event.type == 'mouse_wheel':
			center = self._scene_transform.imap(event.pos)
			#stretch Y
			if "Shift" in event.mouse_event.modifiers:
				self.zoom((1, (1 + self.zoom_factor) ** (-event.delta[1] * 30)), center)
			#standard zoom
			elif "Control" in event.mouse_event.modifiers:
				self.zoom((1 + self.zoom_factor) ** (-event.delta[1] * 30), center)
			#stretch X
			else:
				self.zoom(((1 + self.zoom_factor) ** (-event.delta[1] * 30), 1), center)
			event.handled = True

		elif event.type == 'mouse_move':
			if event.press_event is None:
				return

			modifiers = event.mouse_event.modifiers
			p1 = event.mouse_event.press_event.pos
			p2 = event.mouse_event.pos

			if 1 in event.buttons and not modifiers:
				# Translate
				p1 = np.array(event.last_event.pos)[:2]
				p2 = np.array(event.pos)[:2]
				p1s = self._transform.imap(p1)
				p2s = self._transform.imap(p2)
				self.pan(p1s-p2s)
				event.handled = True
			# elif 2 in event.buttons and not modifiers:
				# # Zoom
				# p1c = np.array(event.last_event.pos)[:2]
				# p2c = np.array(event.pos)[:2]
				# scale = ((1 + self.zoom_factor) **
						 # ((p1c-p2c) * np.array([1, -1])))
				# center = self._transform.imap(event.press_event.pos[:2])
				# self.zoom(scale, center)
				# event.handled = True
			else:
				event.handled = False
		elif event.type == 'mouse_press':
			# accept the event if it is button 1 or 2.
			# This is required in order to receive future events
			event.handled = event.button in [1, 2]
		else:
			event.handled = False
		
	def _set_scene_transform(self, tr):
		""" Called by subclasses to configure the viewbox scene transform.
		"""
		# todo: check whether transform has changed, connect to
		# transform.changed event
		pre_tr = self.pre_transform
		if pre_tr is None:
			self._scene_transform = tr
		else:
			self._transform_cache.roll()
			self._scene_transform = self._transform_cache.get([pre_tr, tr])

		# Mark the transform dynamic so that it will not be collapsed with
		# others 
		self._scene_transform.dynamic = True
		
		# Update scene
		self._viewbox.scene.transform = self._scene_transform
		self._viewbox.update()

		#Apply same state to linked cameras, but prevent that camera
		#to return the favor
		for cam in self._linked_cameras:
			if cam is self._linked_cameras_no_update:
				continue
			try:
				cam._linked_cameras_no_update = self
				r = cam.rect
				r.left = self.rect.left
				r.right = self.rect.right
				cam.rect = r
				cam.view_changed()
			finally:
				cam._linked_cameras_no_update = None
				
class MelTransform(BaseTransform):
	""" Transform perfoming melodic scale transform on the Y axis.
	"""

	glsl_map = """
		vec4 sineTransform(vec4 pos) {
			return vec4(pos.x, log(pos.y / 700 + 1) * 1127, pos.z, 1);
		}"""

	glsl_imap = """
		vec4 sineTransform(vec4 pos) {
			return vec4(pos.x, (exp(pos.y / 1127) -1) * 700, pos.z, 1);
		}"""
	
	Linear = False
	Orthogonal = True
	NonScaling = False
	Isometric = False

	def __init__(self, ):
		super(MelTransform, self).__init__()

	@arg_to_array
	def map(self, coords):
		ret = np.empty(coords.shape, coords.dtype)
		for i in range(min(ret.shape[-1], 3)):
			if i == 1:
				ret[..., i] = np.log(coords[..., i] / 700 + 1) * 1127
			else:
				ret[..., i] = coords[..., i]
		return ret

	@arg_to_array
	def imap(self, coords):
		ret = np.empty(coords.shape, coords.dtype)
		for i in range(min(ret.shape[-1], 3)):
			if i == 1:
				ret[..., i] = (np.exp(coords[..., i] / 1127) - 1) * 700
			else:
				ret[..., i] = coords[..., i]
		return ret

	def shader_map(self):
		fn = super(MelTransform, self).shader_map()
		return fn

	def shader_imap(self):
		fn = super(MelTransform, self).shader_imap()
		return fn

	def __repr__(self):
		return "<MelTransform>"


class ExtAxisWidget(AxisWidget):
	"""Widget containing an axis

	Parameters
	----------
	orientation : str
		Orientation of the axis, 'left' or 'bottom'.
	**kwargs : dict
		Keyword arguments to pass to AxisVisual.
	"""

	def __init__(self, orientation='left', **kwargs):
		if 'tick_direction' not in kwargs:
			tickdir = {'left': (-1, 0), 'right': (1, 0), 'bottom': (0, 1),
					   'top': (0, -1)}[orientation]
			kwargs['tick_direction'] = tickdir

		self.axis = ExtAxisVisual(**kwargs)
		self.orientation = orientation
		self._linked_view = None
		Widget.__init__(self)
		self.add_subvisual(self.axis)


class ExtAxisVisual(AxisVisual):
	def __init__(self, pos=None, domain=(0., 1.), tick_direction=(-1., 0.),
				 scale_type="linear", axis_color=(1, 1, 1),
				 tick_color=(0.7, 0.7, 0.7), text_color='w',
				 minor_tick_length=5, major_tick_length=10,
				 tick_width=2, tick_label_margin=5, tick_font_size=8,
				 axis_width=3,	axis_label=None,
				 axis_label_margin=35, axis_font_size=10,
				 font_size=None, anchors=None):

		if scale_type not in ('linear', 'logarithmic'):
			raise NotImplementedError('only linear scaling is currently '
									  'supported')

		if font_size is not None:
			tick_font_size = font_size
			axis_font_size = font_size

		self._pos = None
		self._domain = None

		# If True, then axis stops at the first / last major tick.
		# If False, then axis extends to edge of *pos*
		# (private until we come up with a better name for this)
		self._stop_at_major = (False, False)

		self.ticker = ExtTicker(self, anchors=anchors)
		self.tick_direction = np.array(tick_direction, float)
		self.scale_type = scale_type

		self.minor_tick_length = minor_tick_length	# px
		self.major_tick_length = major_tick_length	# px
		self.tick_label_margin = tick_label_margin	# px
		self.axis_label_margin = axis_label_margin	# px

		self._axis_label_text = axis_label

		self._need_update = True

		self._line = LineVisual(method='gl', width=axis_width, antialias=True,
								color=axis_color)
		self._ticks = LineVisual(method='gl', width=tick_width,
								 connect='segments', antialias=True,
								 color=tick_color)

		self._text = TextVisual(font_size=tick_font_size, color=text_color)
		self._axis_label = TextVisual(font_size=axis_font_size,
									  color=text_color)
		CompoundVisual.__init__(self, [self._line, self._text, self._ticks,
									   self._axis_label])
		if pos is not None:
			self.pos = pos
		self.domain = domain

	

class ExtTicker(Ticker):

	def _get_tick_frac_labels(self):
		"""Get the major ticks, minor ticks, and major labels"""
		minor_num = 4  # number of minor ticks per major division
		if (self.axis.scale_type == 'linear'):
			domain = self.axis.domain
			if domain[1] < domain[0]:
				flip = True
				domain = domain[::-1]
			else:
				flip = False
			offset = domain[0]
			scale = domain[1] - domain[0]

			transforms = self.axis.transforms
			length = self.axis.pos[1] - self.axis.pos[0]  # in logical coords
			n_inches = np.sqrt(np.sum(length ** 2)) / transforms.dpi

			# major = np.linspace(domain[0], domain[1], num=11)
			# major = MaxNLocator(10).tick_values(*domain)
			major = _get_ticks_talbot(domain[0], domain[1], n_inches, 2)

			# labels = ['%g' % x for x in major]
			labels = [units.t_2_m_s_ms(x) % x for x in major]
			majstep = major[1] - major[0]
			minor = []
			minstep = majstep / (minor_num + 1)
			minstart = 0 if self.axis._stop_at_major[0] else -1
			minstop = -1 if self.axis._stop_at_major[1] else 0
			for i in range(minstart, len(major) + minstop):
				maj = major[0] + i * majstep
				minor.extend(np.linspace(maj + minstep,
										 maj + majstep - minstep,
										 minor_num))
			major_frac = (major - offset) / scale
			minor_frac = (np.array(minor) - offset) / scale
			major_frac = major_frac[::-1] if flip else major_frac
			use_mask = (major_frac > -0.0001) & (major_frac < 1.0001)
			major_frac = major_frac[use_mask]
			labels = [l for li, l in enumerate(labels) if use_mask[li]]
			minor_frac = minor_frac[(minor_frac > -0.0001) &
									(minor_frac < 1.0001)]
		elif self.axis.scale_type == 'logarithmic':
			#domain = np.log2(self.axis.domain)
			domain = self.axis.domain
			if domain[1] < domain[0]:
				flip = True
				domain = domain[::-1]
			else:
				flip = False
			offset = domain[0]
			scale = domain[1] - domain[0]

			transforms = self.axis.transforms
			length = self.axis.pos[1] - self.axis.pos[0]  # in logical coords
			n_inches = np.sqrt(np.sum(length ** 2)) / transforms.dpi

			# major = np.linspace(domain[0], domain[1], num=11)
			# major = MaxNLocator(10).tick_values(*domain)
			major = _get_ticks_talbot(domain[0], domain[1], n_inches, 2)
			#print(major)
			#np.exp(x / 1127) - 1) * 700
			#labels = ['%g' % np.exp(x / 1127) - 1 * 700 for x in major]
			labels = [str(int((np.exp(x / 1127) - 1) * 700)) for x in major]
			majstep = major[1] - major[0]
			minor = []
			minstep = majstep / (minor_num + 1)
			minstart = 0 if self.axis._stop_at_major[0] else -1
			minstop = -1 if self.axis._stop_at_major[1] else 0
			for i in range(minstart, len(major) + minstop):
				maj = major[0] + i * majstep
				minor.extend(np.linspace(maj + minstep,
										 maj + majstep - minstep,
										 minor_num))
			major_frac = (major - offset) / scale
			minor_frac = (np.array(minor) - offset) / scale
			major_frac = major_frac[::-1] if flip else major_frac
			use_mask = (major_frac > -0.0001) & (major_frac < 1.0001)
			major_frac = major_frac[use_mask]
			labels = [l for li, l in enumerate(labels) if use_mask[li]]
			minor_frac = minor_frac[(minor_frac > -0.0001) &
									(minor_frac < 1.0001)]
		elif self.axis.scale_type == 'power':
			return NotImplementedError
		return major_frac, minor_frac, labels


def _coverage(dmin, dmax, lmin, lmax):
	return 1 - 0.5 * ((dmax - lmax) ** 2 +
					  (dmin - lmin) ** 2) / (0.1 * (dmax - dmin)) ** 2


def _coverage_max(dmin, dmax, span):
	range_ = dmax - dmin
	if span <= range_:
		return 1.
	else:
		half = (span - range_) / 2.0
		return 1 - half ** 2 / (0.1 * range_) ** 2


def _density(k, m, dmin, dmax, lmin, lmax):
	r = (k-1.0) / (lmax-lmin)
	rt = (m-1.0) / (max(lmax, dmax) - min(lmin, dmin))
	return 2 - max(r / rt, rt / r)


def _density_max(k, m):
	return 2 - (k-1.0) / (m-1.0) if k >= m else 1.


def _simplicity(q, Q, j, lmin, lmax, lstep):
	eps = 1e-10
	n = len(Q)
	i = Q.index(q) + 1
	if ((lmin % lstep) < eps or
			(lstep - lmin % lstep) < eps) and lmin <= 0 and lmax >= 0:
		v = 1
	else:
		v = 0
	return (n - i) / (n - 1.0) + v - j


def _simplicity_max(q, Q, j):
	n = len(Q)
	i = Q.index(q) + 1
	return (n - i)/(n - 1.0) + 1. - j
def _get_ticks_talbot(dmin, dmax, n_inches, density=1.):
	# density * size gives target number of intervals,
	# density * size + 1 gives target number of tick marks,
	# the density function converts this back to a density in data units
	# (not inches)
	n_inches = max(n_inches, 2.0)  # Set minimum otherwise code can crash :(
	m = density * n_inches + 1.0
	only_inside = False	 # we cull values outside ourselves
	Q = [1, 5, 2, 2.5, 4, 3]
	w = [0.25, 0.2, 0.5, 0.05]
	best_score = -2.0
	best = None

	j = 1.0
	n_max = 1000
	while j < n_max:
		for q in Q:
			sm = _simplicity_max(q, Q, j)

			if w[0] * sm + w[1] + w[2] + w[3] < best_score:
				j = n_max
				break

			k = 2.0
			while k < n_max:
				dm = _density_max(k, n_inches)

				if w[0] * sm + w[1] + w[2] * dm + w[3] < best_score:
					break

				delta = (dmax-dmin)/(k+1.0)/j/q
				z = np.ceil(np.log10(delta))

				while z < float('infinity'):
					step = j * q * 10 ** z
					cm = _coverage_max(dmin, dmax, step*(k-1.0))

					if (w[0] * sm +
							w[1] * cm +
							w[2] * dm +
							w[3] < best_score):
						break

					min_start = np.floor(dmax/step)*j - (k-1.0)*j
					max_start = np.ceil(dmin/step)*j

					if min_start > max_start:
						z = z+1
						break

					for start in range(int(min_start), int(max_start)+1):
						lmin = start * (step/j)
						lmax = lmin + step*(k-1.0)
						lstep = step

						s = _simplicity(q, Q, j, lmin, lmax, lstep)
						c = _coverage(dmin, dmax, lmin, lmax)
						d = _density(k, m, dmin, dmax, lmin, lmax)
						leg = 1.  # _legibility(lmin, lmax, lstep)

						score = w[0] * s + w[1] * c + w[2] * d + w[3] * leg

						if (score > best_score and
								(not only_inside or (lmin >= dmin and
													 lmax <= dmax))):
							best_score = score
							best = (lmin, lmax, lstep, q, k)
					z += 1
				k += 1
			if k == n_max:
				raise RuntimeError('could not converge on ticks')
		j += 1
	if j == n_max:
		raise RuntimeError('could not converge on ticks')

	if best is None:
		raise RuntimeError('could not converge on ticks')
	return np.arange(best[4]) * best[2] + best[0]

