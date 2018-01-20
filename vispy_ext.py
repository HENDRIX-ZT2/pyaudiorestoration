# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# vispy: gallery 2

import numpy as np
from vispy.visuals.transforms import (arg_to_array, BaseTransform)
from vispy.scene import PanZoomCamera#, BaseCamera
#from vispy.visuals.transforms import STTransform
from vispy.geometry import Rect

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

