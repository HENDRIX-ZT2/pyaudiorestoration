import logging
import os
import numpy as np
from vispy import scene, gloo, visuals
from vispy.color import BaseColormap
from vispy.geometry import Rect

from util import vispy_ext, qt_threads, units, colormaps

from util.undo import DeleteAction


class FlatRed(BaseColormap):
	glsl_map = """
	vec4 simple_cmap(float x) {
		return vec4(x, 0, 0, 1);
	}
	"""


class FlatGreen(BaseColormap):
	glsl_map = """
	vec4 simple_cmap(float x) {
		return vec4(0, x, 0, 1);
	}
	"""


add_cmaps = {"add_r": FlatRed(), "add_g": FlatGreen()}


class Spectrum:
	"""The visualization of the whole spectrogram."""

	def __init__(self, view, overlay=False):
		self.overlay = overlay
		self.pieces = []
		self.view = view
		self.MAX_TEXTURE_SIZE = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)
		self.mel_transform = vispy_ext.MelTransform()
		self.offset = 0
		self.num_ffts = 0
		# which channel should be used to render this spectrum from?
		self.signal = None
		self.audio_path = None
		# set a reasonable default
		self.sr = 44100
		self.duration = 0.0
		self.selected_channel = 0
		self.fft_storage = {}
		self.key = None

	def get_related_keys(self):
		"""Return the keys for different hops but otherwise same settings"""
		more_dense = None
		more_sparse = None
		# go over all keys and see if there is a bigger one
		for key in self.fft_storage:
			if key[0:2] == self.key[0:2]:
				if key[2] > self.key[2]:
					more_sparse = key
				elif key[2] < self.key[2]:
					# only save key if none had been set or the new key is closer to the desired key
					if not more_dense or more_dense[2] < key[2]:
						more_dense = key
		return more_dense, more_sparse

	def change_file(self, audio_path):
		# remove all ffts of the old file from storage
		self.fft_storage.clear()
		self.audio_path = audio_path
		self.duration = len(self.signal) / self.sr

	def update_data(self, imdata, hop):
		num_bins, self.num_ffts = imdata.shape
		# determine how many pieces we will create
		num_pieces_new = ((self.num_ffts - 1) // self.MAX_TEXTURE_SIZE + 1) * (
				(num_bins - 1) // self.MAX_TEXTURE_SIZE + 1)
		num_pieces_old = len(self.pieces)
		# reusing pieces no longer works for some reason
		# we have too many pieces and need to discard some
		for i in reversed(range(num_pieces_new, num_pieces_old)):
			self.pieces[i].parent = None
			self.pieces.pop(i)
		# add new pieces
		for i in range(num_pieces_old, num_pieces_new):
			self.pieces.append(SpectrumPiece(self.view.scene))
		# spectra may only be of a certain size, so split them
		for i, (x, y) in enumerate([(x, y) for x in range(0, self.num_ffts, self.MAX_TEXTURE_SIZE) for y in
									range(0, num_bins, self.MAX_TEXTURE_SIZE)]):
			logging.debug(f"Piece {i} at x={x}, y={y} sr={self.sr}")
			imdata_piece = imdata[y:y + self.MAX_TEXTURE_SIZE, x:x + self.MAX_TEXTURE_SIZE]
			num_piece_bins, num_piece_ffts = imdata_piece.shape

			# to get correct heights for subsequent pieces, the start y has to be added in mel space
			height_Hz_in = num_piece_bins / num_bins * self.f_max
			ystart_Hz = y / num_bins * self.f_max
			height_Hz_corrected = units.to_Hz(units.to_mel(height_Hz_in + ystart_Hz) - units.to_mel(ystart_Hz))
			x_start = x * hop / self.sr + self.offset
			x_len = num_piece_ffts * hop / self.sr

			# do the dB conversion here because the tracers don't like it
			piece = self.pieces[i]
			piece.set_data(units.to_dB(imdata_piece))
			piece.size = (x_len, height_Hz_corrected)
			# add this piece's offset with STT
			piece.transform = visuals.transforms.STTransform(
				translate=(x_start, units.to_mel(ystart_Hz))) * self.mel_transform
			piece.bb.left = x_start
			piece.bb.right = x_start + x_len
			piece.update()
			piece.show()

	def set_clims(self, vmin, vmax):
		for image in self.pieces:
			image.clim = (vmin, vmax)

	def set_cmap(self, colormap):
		for image in self.pieces:
			if self.overlay:
				image.cmap = self.overlay
			else:
				image.cmap = colormap

	def update_frustum(self, event):
		for image in self.pieces:
			# X frustum
			if image.bb.left > self.view.camera.rect.right or image.bb.right < self.view.camera.rect.left:
				image.hide()
			else:
				image.show()

	def set_offset(self, offset):
		d = offset - self.offset
		self.translate(d)

	def translate(self, d):
		# move all pieces in X direction by d
		self.offset += d
		for image in self.pieces:
			t = image.transform.transforms[0].translate
			image.bb.left += d
			image.bb.right += d
			t[0] += d
			image.transform.transforms[0].translate = t

	@property
	def f_max(self):
		return self.sr / 2

	def get_signal(self, t0, t1):
		"""Get a signal from t0 to t1, padded as needed"""
		ref_t0 = int(t0*self.sr)
		ref_t1 = int(t1*self.sr)
		ref_pad_l = 0
		ref_pad_r = 0
		# pad if needed
		if ref_t0 < 0:
			ref_pad_l = abs(ref_t0)
			ref_t0 = 0
		if ref_t1 > len(self.signal):
			ref_pad_r = ref_t1 - len(self.signal)
		sig = self.signal[ref_t0:ref_t1, self.selected_channel]
		return np.pad(sig, (ref_pad_l, ref_pad_r), "constant", constant_values=0)


class SpectrumPiece(scene.Image):
	"""The visualization of one part of the whole spectrogram."""

	def __init__(self, vispy_scene):
		self._parent2 = vispy_scene
		# # just set a dummy value
		self._shape = (10.0, 22500)
		self.bb = Rect((0, 0, 1, 1))
		scene.Image.__init__(self, parent=vispy_scene, interpolation="linear", method='subdivide', grid=(1000, 1))
		self.cmap = "izo"

	@property
	def cmap(self):
		"""Get the colormap object applied to luminance (single band) data."""
		return self._cmap

	@cmap.setter
	def cmap(self, cmap_name):
		if cmap_name.startswith("add_"):
			self.set_gl_state('additive')
			self._cmap = add_cmaps.get(cmap_name)
		else:
			self._cmap = colormaps.cmaps.get(cmap_name)
		self._need_colortransform_update = True
		self.update()

	@property
	def size(self):
		return self._shape

	@size.setter
	def size(self, v):
		self._shape = v
		self._need_vertex_update = True
		self._need_method_update = True

	def hide(self):
		self.parent = None

	def show(self):
		self.parent = self._parent2


class SpectrumCanvas(scene.SceneCanvas):
	"""This class wraps the vispy canvas and controls all the visualization, as well as the interaction with it."""

	def __init__(self, spectra_colors=("add_r", "add_g"), y_axis='Src. Lag', bgcolor="#353535"):
		label_color = "grey"

		# some default dummy values
		self.props = None
		self.vmin = -120
		self.vmax = 0
		self.cmap = "izo"
		self.fft_size = 1024
		self.hop = 256

		self.fourier_thread = qt_threads.FourierThread()
		self.fourier_thread.finished.connect(self.retrieve_fft)

		scene.SceneCanvas.__init__(self, keys="interactive", size=(1024, 512), bgcolor=bgcolor)

		self.unfreeze()

		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0

		# speed chart
		self.speed_yaxis = scene.AxisWidget(
			orientation='left', axis_label=y_axis, axis_font_size=8,
			axis_label_margin=35, tick_label_margin=5, axis_color=label_color)
		self.speed_yaxis.width_max = 55

		# spectrum
		self.spec_yaxis = vispy_ext.ExtAxisWidget(
			orientation='left', axis_label='Hz', axis_font_size=8,
			axis_label_margin=35, tick_label_margin=5, scale_type="logarithmic", axis_color=label_color)
		self.spec_yaxis.width_max = 55

		self.spec_xaxis = vispy_ext.ExtAxisWidget(
			orientation='bottom', axis_label='m:s:ms', axis_font_size=8,
			axis_label_margin=30, tick_label_margin=20, axis_color=label_color)
		self.spec_xaxis.height_max = 55

		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 10

		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 55

		# create the color bar display
		self.colorbar_display = vispy_ext.ColorBarWidgetExt(label="dB", clim=(self.vmin, self.vmax),
															cmap="viridis", orientation="right", border_width=1,
															label_color="white", axis_ratio=0.01, padding=(0, 0))
		self.colorbar_display.label.font_size = 8
		self.colorbar_display.ticks[0].font_size = 8
		self.colorbar_display.ticks[1].font_size = 8
		self.colorbar_display.cmap = "izo"

		grid.add_widget(self.speed_yaxis, row=1, col=0)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		grid.add_widget(self.colorbar_display, row=2, col=2)

		self.speed_view = grid.add_view(row=1, col=1, border_color=label_color)
		self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, -5, 10, 10), )
		self.speed_view.height_min = 150
		# self.speed_view.stretch = (1, 1)
		self.spec_view = grid.add_view(row=2, col=1, border_color=label_color)
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		self.spec_view.height_min = 550
		self.spec_view.stretch = (2, 2)
		# link them, but use custom logic to only link the x view
		self.spec_view.camera.link(self.speed_view.camera)
		self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)

		self.spectra = [Spectrum(self.spec_view, overlay=color) for color in spectra_colors]
		self.markers = []

		# nb. this is a vispy.util.event.EventEmitter object
		# can this be linked somewhere to the camera? base_camera connects a few events, too
		for spe in self.spectra:
			self.spec_view.transforms.changed.connect(spe.update_frustum)

		self.freeze()

	@property
	def sr(self):
		"""Use the maximum to avoid downsampling when sr differs per source"""
		return max(spec.sr for spec in self.spectra)

	@property
	def duration(self):
		"""Use the maximum to get total when duration differs per source"""
		durations = [spectrum.duration for spectrum in self.spectra if spectrum.signal is not None]
		if durations:
			return max(durations)
		else:
			return 1.0

	@property
	def f_max(self):
		return self.sr / 2

	# todo - get rid of these?
	@property
	def filenames(self):
		return [spectrum.audio_path for spectrum in self.spectra]

	@property
	def signals(self):
		return [spectrum.signal for spectrum in self.spectra]

	def clear_fft_storage(self):
		logging.info("Clearing FFT storage")
		for spectrum in self.spectra:
			for key in tuple(spectrum.fft_storage.keys()):
				if key != spectrum.key:
					logging.info(f"deleting {key}")
					del spectrum.fft_storage[key]

	def compute_spectra(self):
		# maybe move more into the thread
		if self.fourier_thread.jobs:
			logging.warning("Fourier job is still running, wait!")
			return

		for spectrum in self.spectra:
			if spectrum.audio_path:
				spectrum.key = (self.fft_size, spectrum.selected_channel, self.hop)
				# first try to get FFT from current storage and continue directly
				if spectrum.key in spectrum.fft_storage:
					# this happens when only loading from storage is required
					self.update_spectrum(spectrum)
				# check for alternate hops
				else:
					more_dense, more_sparse = spectrum.get_related_keys()
					# prefer reduction via strides
					if more_dense:
						logging.debug(f"reducing resolution via stride from {more_dense[2]} to {spectrum.key[2]}")
						step = spectrum.key[2] // more_dense[2]
						spectrum.fft_storage[spectrum.key] = np.array(spectrum.fft_storage[more_dense][:, ::step])
						self.update_spectrum(spectrum)
					# TODO: implement gap filling, will need changes to stft function
					# then fill missing gaps
					# elif more_sparse:
					# 	print("increasing resolution by filling gaps",self.fft_size)
					else:
						logging.info(f"storing new fft {spectrum.audio_path, spectrum.key}")
						# append to the fourier job list
						self.fourier_thread.jobs.append((
							spectrum.signal[:, spectrum.selected_channel], self.fft_size, self.hop, "blackmanharris", spectrum.key, spectrum.audio_path))
		# perform all fourier jobs
		if self.fourier_thread.jobs:
			self.fourier_thread.start()
		# continue when fourier_thread emits a "finished" signal, connected to retrieve_fft()

	def retrieve_fft(self, ):
		logging.info("Retrieving FFT from processing thread")
		for spec in self.spectra:
			# check if there's a result to fetch
			# when there are more spectra, this is not sure
			if (spec.audio_path, spec.key) in self.fourier_thread.result:
				spec.fft_storage[spec.key] = self.fourier_thread.result[spec.audio_path, spec.key]
				self.update_spectrum(spec)
		self.fourier_thread.result.clear()

	def update_spectrum(self, spec):
		try:
			spec.update_data(spec.fft_storage[spec.key], self.hop)
		except:
			logging.exception(f"Updating spectrum failed")
		spec.set_clims(self.vmin, self.vmax)
		spec.set_cmap(self.cmap)

	# don't bother with audio until it is fixed
	# self.props.audio_widget.set_data(signal[:,channel], spec.sr)

	def reset_view(self):
		# (re)set the spec_view
		self.speed_view.camera.rect = (0, -1, self.duration, 2)
		self.spec_view.camera.rect = (0, 0, self.duration, units.to_mel(self.f_max))

	# fast stuff that does not require rebuilding everything
	def set_colormap(self, cmap):
		"""cmap is a string at this point"""
		self.cmap = cmap
		for spe in self.spectra:
			spe.set_cmap(cmap)
		self.colorbar_display.cmap = cmap

	def set_clims(self, vmin, vmax):
		self.vmin = vmin
		self.vmax = vmax
		for spe in self.spectra:
			spe.set_clims(vmin, vmax)
		self.colorbar_display.clim = (vmin, vmax)

	def on_mouse_wheel(self, event):
		# coords of the click on the vispy canvas
		click = np.array([event.pos[0], event.pos[1], 0, 1])

		# colorbar scroll
		if self.click_on_widget(click, self.colorbar_display):
			y_pos = self.colorbar_display._colorbar.transform.imap(click)[1]
			d = int(event.delta[1])
			# now split Y in three parts
			lower = self.colorbar_display.size[1] / 3
			upper = lower * 2
			if y_pos < lower:
				self.vmax += d
			elif lower < y_pos < upper:
				self.vmin += d
				self.vmax -= d
			elif upper < y_pos:
				self.vmin += d
			self.set_clims(self.vmin, self.vmax)

		# spec & speed X axis scroll
		if self.click_on_widget(click, self.spec_xaxis):
			# the center of zoom should be assigned a new x coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (scene_space[0], self.spec_view.camera.center[1])
			self.spec_view.camera.zoom(((1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30), 1), c)

		# spec Y axis scroll
		if self.click_on_widget(click, self.spec_yaxis):
			# the center of zoom should be assigned a new y coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (self.spec_view.camera.center[0], scene_space[1])
			self.spec_view.camera.zoom((1, (1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)

		# speed Y axis scroll
		if self.click_on_widget(click, self.speed_yaxis):
			# the center of zoom should be assigned a new y coordinate
			grid_space = self.speed_view.transform.imap(click)
			scene_space = self.speed_view.scene.transform.imap(grid_space)
			c = (self.speed_view.camera.center[0], scene_space[1])
			self.speed_view.camera.zoom((1, (1 + self.speed_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)

	def on_mouse_move(self, event):
		# update the inspector label
		click = self.px_to_spectrum(event.pos)
		self.props.inspector_widget.update_text(click, self.spectra[0].sr)

	def _get_closest(self, items, a, b):
		assert len(items) == len(a)
		# actually, we don't need the euclidean distance here, just a relative distance metric
		# so we can avoid the sqrt and just take the squared distance
		ind = np.sum((a - b) ** 2, axis=1).argmin()
		return items[ind]

	def get_closest(self, items, click):
		"""returns the sample (if any exists) whose center is closest to pt"""
		visible_items = [item for item in items if item.visuals[0].parent]
		if len(visible_items) and click is not None:
			# convert the samples to screen space!
			c = self.px_to_spectrum(click)
			if c is not None:
				px_centers = np.array([self.spectrum_to_px(sample.spec_center)[0:2] for sample in visible_items])
				return self._get_closest(visible_items, px_centers, click[0:2])
			c = self.px_to_speed(click)
			if c is not None:
				px_centers = np.array([self.speed_to_px(sample.speed_center)[0:2] for sample in visible_items])
				return self._get_closest(visible_items, px_centers, click[0:2])

	def click_on_widget(self, click, wid):
		grid_space = wid.transform.imap(click)
		dim = wid.size
		return (0 < grid_space[0] < dim[0]) and (0 < grid_space[1] < dim[1])

	def px_to_speed(self, click):
		# in the grid on the canvas
		grid_space = self.speed_view.transform.imap(click)
		# is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.speed_view):
			return self.speed_view.scene.transform.imap(grid_space)

	def px_to_spectrum(self, click):
		# in the grid on the canvas
		grid_space = self.spec_view.transform.imap(click)
		# is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.spec_view):
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			return self.spectra[0].mel_transform.imap(scene_space)

	def speed_to_px(self, pt):
		"""converts a point from octave space to screen space"""
		scene_space = self.speed_view.scene.transform.map(pt)
		return self.speed_view.transform.map(scene_space)

	def spectrum_to_px(self, pt):
		"""converts a point from Hz space to screen space"""
		melspace = self.spectra[0].mel_transform.map(pt)
		scene_space = self.spec_view.scene.transform.map(melspace)
		return self.spec_view.transform.map(scene_space)

	def delete_traces(self, delete_all=False):
		deltraces = []
		for trace in reversed(self.markers):
			if delete_all or trace.selected:
				deltraces.append(trace)
		self.props.undo_stack.push(DeleteAction(deltraces))

	def select_all(self):
		for trace in self.markers:
			trace.select()

	def deselect_all(self):
		for trace in self.markers:
			trace.deselect()

	def invert_selection(self):
		for trace in self.markers:
			trace.toggle()

	@property
	def selected_markers(self):
		return [marker for marker in self.markers if marker.selected]
