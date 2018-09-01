# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


import numpy as np
from vispy import scene, gloo, visuals, color
from vispy.geometry import Rect
#custom modules
import vispy_ext

#log10(x) = log(x) / log(10) = (1 / log(10)) * log(x)
norm_luminance = """
float norm_luminance(vec2 pos) {
	if( pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1 ) {
		return -1.0f;
	}
	vec2 uv = vec2(pos.x, pos.y);
	return (texture2D($texture, uv).r - $vmin)/($vmax - $vmin);
}
"""

simple_cmap = """
vec4 simple_cmap(float x) {
	return vec4(0, 0, x, 1);
}
"""

r_cmap = """
vec4 simple_cmap(float x) {
	return vec4(x, 0, 0, 1);
}
"""

g_cmap = """
vec4 simple_cmap(float x) {
	return vec4(0, x, 0, 1);
}
"""

class Spectrum():
	"""
	The visualization of the whole spectrogram.
	"""
	def __init__(self, parent, overlay=False):
		self.overlay = overlay
		self.pieces = []
		self.parent = parent
		self.MAX_TEXTURE_SIZE = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)
		self.mel_transform = vispy_ext.MelTransform()
		self.empty = np.zeros((1,1), dtype="float32")
		self.delta = 0
	
	def update_data(self, imdata, hop, sr):
		num_bins, num_ffts = imdata.shape
		#determine how many pieces we will create
		num_pieces_new = ((num_ffts-1)//self.MAX_TEXTURE_SIZE +1) * ((num_bins-1)//self.MAX_TEXTURE_SIZE+1)
		num_pieces_old = len(self.pieces)
		#we have too many pieces and need to discard some
		for i in reversed(range(num_pieces_new, num_pieces_old)):
			self.pieces[i].parent = None
			self.pieces.pop(i)
			# print("removing",i)
		#add new pieces
		for i in range(num_pieces_old, num_pieces_new):
			self.pieces.append(SpectrumPiece(self.empty, self.parent.scene, self.overlay))
			# print("adding",i)

		#spectra may only be of a certain size, so split them
		for i, (x, y) in enumerate([(x, y) for x in range(0, num_ffts, self.MAX_TEXTURE_SIZE) for y in range(0, num_bins, self.MAX_TEXTURE_SIZE)]):
			imdata_piece = imdata[y:y+self.MAX_TEXTURE_SIZE, x:x+self.MAX_TEXTURE_SIZE]
			num_piece_bins, num_piece_ffts = imdata_piece.shape
			
			#to get correct heights for subsequent pieces, the start y has to be added in mel space
			height_Hz_in = int(num_piece_bins/num_bins*sr/2)
			ystart_Hz = (y/num_bins*sr/2)
			height_Hz_corrected = to_Hz(to_mel(height_Hz_in + ystart_Hz) - to_mel(ystart_Hz))
			x_start = x * hop / sr + self.delta
			x_len = num_piece_ffts*hop/sr
			
			#do the dB conversion here because the tracers don't like it
			self.pieces[i].tex.set_data(20 * np.log10(imdata_piece))
			self.pieces[i].set_size((x_len, height_Hz_corrected))
			#add this piece's offset with STT
			self.pieces[i].transform = visuals.transforms.STTransform( translate=(x_start, to_mel(ystart_Hz))) * self.mel_transform
			self.pieces[i].bb.left = x_start
			self.pieces[i].bb.right = x_start+x_len
			self.pieces[i].update()
	
	def set_clims(self, vmin, vmax):
		for image in self.pieces:
			image.set_clims(vmin, vmax)
			
	def set_cmap(self, colormap):
		for image in self.pieces:
			image.set_cmap(colormap)
			
	def update_frustum(self, event):
		for image in self.pieces:
			#X frustum
			if image.bb.left > self.parent.camera.rect.right or image.bb.right < self.parent.camera.rect.left:
				image.hide()
			else:
				image.show()
				
	def translate(self, d):
		#move all pieces in X direction by d
		self.delta+=d
		for image in self.pieces:
			t = image.transform.transforms[0].translate
			t[0]+=d
			image.transform.transforms[0].translate = t

class SpectrumPiece(scene.Image):
	"""
	The visualization of one part of the whole spectrogram.
	"""
	def __init__(self, texdata, parent, overlay=False):
		self._parent2 = parent
		self.overlay = overlay
		#just set a dummy value
		self._shape = (10.0, 22500)
		self.tex = gloo.Texture2D(texdata, format='luminance', internalformat='r32f', interpolation="linear")
		self.get_data = visuals.shaders.Function(norm_luminance)
		self.get_data['vmin'] = -80
		self.get_data['vmax'] = -40
		self.get_data['texture'] = self.tex
		
		self.bb = Rect((0, 0, 1, 1))

		scene.Image.__init__(self, method='subdivide', grid=(1000,1), parent=parent)
		
		#set in the main program
		self.shared_program.frag['get_data'] = self.get_data
		
		#needs no external color map
		if self.overlay == "r":
			self.set_gl_state('additive')
			self.shared_program.frag['color_transform'] = visuals.shaders.Function(r_cmap)
		elif self.overlay == "g":
			self.set_gl_state('additive')
			self.shared_program.frag['color_transform'] = visuals.shaders.Function(g_cmap)
		
	def set_size(self, size):
		#not sure if update is needed
		self._shape = size
		self.update()
	
	def set_cmap(self, colormap):
		if not self.overlay:
			# update is needed
			self.shared_program.frag['color_transform'] = visuals.shaders.Function(color.get_colormap(colormap).glsl_map)
			self.update()
		
	def set_clims(self, vmin, vmax):
		self.get_data['vmin'] = vmin
		self.get_data['vmax'] = vmax
		
	@property
	def size(self):
		return self._shape

	def _prepare_draw(self, view):
		if self._need_vertex_update:
			self._build_vertex_data()

		if view._need_method_update:
			self._update_method(view)
			
	def hide(self):
		self.parent = None
		
	def show(self):
		self.parent = self._parent2

		
def to_mel(val):
	### just to set the image size correctly	
	return np.log(val / 700 + 1) * 1127

def to_Hz(val):
	### just to set the image size correctly	
	return (np.exp(val / 1127) - 1) * 700
	