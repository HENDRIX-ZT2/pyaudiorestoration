import os
import numpy as np
import soundfile as sf
from vispy import scene, gloo, visuals, color
from vispy.geometry import Rect
#custom modules
from util import vispy_ext, fourier

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
		#add new pieces
		for i in range(num_pieces_old, num_pieces_new):
			self.pieces.append(SpectrumPiece(self.empty, self.parent.scene, self.overlay))
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
			self.pieces[i].show()
	
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
			image.bb.left+=d
			image.bb.right+=d
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


class SpectrumCanvas(scene.SceneCanvas):
	"""This class wraps the vispy canvas and controls all the visualization, as well as the interaction with it."""

	def __init__(self, spectra_colors=("r","g"), y_axis='Src. Lag', bgcolor="#353535"):
		
		#some default dummy values
		self.filenames = ()
		self.props = None
		self.vmin = -80
		self.vmax = -40
		self.num_cores = os.cpu_count()
		self.fft_size = 1024
		self.hop = 256
		self.sr = 44100
		self.num_ffts = 0
		
		scene.SceneCanvas.__init__(self, keys="interactive", size=(1024, 512), bgcolor=bgcolor)
		
		self.unfreeze()
		
		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0
		
		#speed chart
		self.speed_yaxis = scene.AxisWidget(orientation='left', axis_label=y_axis, axis_font_size=8, axis_label_margin=35, tick_label_margin=5)
		self.speed_yaxis.width_max = 55
		
		#spectrum
		self.spec_yaxis = vispy_ext.ExtAxisWidget(orientation='left', axis_label='Hz', axis_font_size=8, axis_label_margin=35, tick_label_margin=5, scale_type="logarithmic")
		self.spec_yaxis.width_max = 55
		
		self.spec_xaxis = scene.AxisWidget(orientation='bottom', axis_label='sec', axis_font_size=8, axis_label_margin=35, tick_label_margin=5)
		self.spec_xaxis.height_max = 55

		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 10
		
		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 70
		
		#create the color bar display
		self.colorbar_display = scene.ColorBarWidget(label="Gain [dB]", clim=(self.vmin, self.vmax), cmap="viridis", orientation="right", border_width=1, label_color="white")
		self.colorbar_display.label.font_size = 8
		self.colorbar_display.ticks[0].font_size = 8
		self.colorbar_display.ticks[1].font_size = 8
		
		grid.add_widget(self.speed_yaxis, row=1, col=0)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		colorbar_column = grid.add_widget(self.colorbar_display, row=2, col=2)
		
		self.speed_view = grid.add_view(row=1, col=1, border_color='white')
		self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, -5, 10, 10), )
		self.speed_view.height_min = 150
		self.spec_view = grid.add_view(row=2, col=1, border_color='white')
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		#link them, but use custom logic to only link the x view
		self.spec_view.camera.link(self.speed_view.camera)
		self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)
		
		self.spectra = [Spectrum(self.spec_view, overlay=color) for color in spectra_colors]
		self.init_fft_storages()
		#nb. this is a vispy.util.event.EventEmitter object
		#can this be linked somewhere to the camera? base_camera connects a few events, too
		for spe in self.spectra:
			self.spec_view.transforms.changed.connect(spe.update_frustum)
		
		self.freeze()
		
	def init_fft_storages(self,):
		self.fft_storages = [{} for x in self.spectra]
	
	def compute_spectra(self, files, fft_size, fft_overlap, channels=None):
		if channels is None:
			channels = [0 for file in files]
		for filename, spec, fft_storage, channel in zip(files, self.spectra, self.fft_storages, channels):
			#set this for the tracers etc.
			self.fft_size = fft_size
			self.hop = fft_size // fft_overlap
			if filename:
				soundob = sf.SoundFile(filename)
					
				self.sr = soundob.samplerate
				k = (self.fft_size, self.hop, channel)
				if k not in fft_storage:
					print("storing new fft",self.fft_size)
					signal = soundob.read(always_2d=True, dtype='float32')[:,channel]
					#now store this for retrieval later
					fft_storage[k] = fourier.stft(signal, self.fft_size, self.hop, "hann", self.num_cores)
				
				#retrieve the FFT data
				imdata = fft_storage[k]
				self.num_ffts = max(self.num_ffts, imdata.shape[1])
				spec.update_data(imdata, self.hop, self.sr)
		
		#has the file changed?
		if self.filenames != files:
			print("file has changed!")
			self.filenames = files
			#(re)set the spec_view
			self.speed_view.camera.rect = (0, -1, self.num_ffts * self.hop / self.sr, 2)
			self.spec_view.camera.rect = (0, 0, self.num_ffts * self.hop / self.sr, to_mel(self.sr//2))
			if filename:
				self.props.audio_widget.set_data(signal, self.sr)
		self.set_clims(self.vmin, self.vmax)
			
	#fast stuff that does not require rebuilding everything
	def set_colormap(self, cmap):
		for spe in self.spectra:
			spe.set_cmap(cmap)
		self.colorbar_display.cmap = cmap
		
	#fast stuff that does not require rebuilding everything
	def set_clims(self, vmin, vmax):
		for spe in self.spectra:
			spe.set_clims(vmin, vmax)
		self.colorbar_display.clim = (vmin, vmax)
	
	def on_mouse_wheel(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		
		#colorbar scroll
		if self.click_on_widget(click, self.colorbar_display):
			y_pos = self.colorbar_display._colorbar.transform.imap(click)[1]
			d = int(event.delta[1])
			#now split Y in three parts
			lower = self.colorbar_display.size[1]/3
			upper = lower*2
			if y_pos < lower:
				self.vmax += d
			elif lower < y_pos < upper:
				self.vmin += d
				self.vmax -= d
			elif upper < y_pos:
				self.vmin += d
			self.set_clims(self.vmin, self.vmax)
				
		#spec & speed X axis scroll
		if self.click_on_widget(click, self.spec_xaxis):
			#the center of zoom should be assigned a new x coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (scene_space[0], self.spec_view.camera.center[1])
			self.spec_view.camera.zoom(((1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30), 1), c)

		#spec Y axis scroll
		if self.click_on_widget(click, self.spec_yaxis):
			#the center of zoom should be assigned a new y coordinate
			grid_space = self.spec_view.transform.imap(click)
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			c = (self.spec_view.camera.center[0], scene_space[1])
			self.spec_view.camera.zoom((1, (1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)
		
		#speed Y axis scroll
		if self.click_on_widget(click, self.speed_yaxis):
			#the center of zoom should be assigned a new y coordinate
			grid_space = self.speed_view.transform.imap(click)
			scene_space = self.speed_view.scene.transform.imap(grid_space)
			c = (self.speed_view.camera.center[0], scene_space[1])
			self.speed_view.camera.zoom((1, (1 + self.speed_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)

	def on_mouse_move(self, event):
		#update the inspector label
		click = self.click_spec_conversion(event.pos)
		self.props.inspector_widget.update_text(click, self.sr)
				
	def get_closest(self, items, click,):
		if click is not None:
			c = self.click_spec_conversion(click)
			#convert the samples to screen space!
			if c is not None:
				A = np.array([self.pt_spec_conversion(sample.spec_center)[0:2] for sample in items])
			#check in speed view
			else:
				c = self.click_speed_conversion(click)
				if c is not None:
					A = np.array([self.pt_spec_conversion(sample.speed_center)[0:2] for sample in items])
			#returns the sample (if any exists) whose center is closest to pt
			if c is not None and len(A):
				#actually, we don't need the euclidean distance here, just a relative distance metric, so we can avoid the sqrt and just take the squared distance
				ind = np.sum((A-click[0:2])**2, axis = 1).argmin()
				return items[ind]
	
	def click_on_widget(self, click, wid):
		grid_space = wid.transform.imap(click)
		dim = wid.size
		return (0 < grid_space[0] < dim[0]) and (0 < grid_space[1] < dim[1])
	
	def click_speed_conversion(self, click):
		#in the grid on the canvas
		grid_space = self.speed_view.transform.imap(click)
		#is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.speed_view):
			return self.speed_view.scene.transform.imap(grid_space)
				
	def click_spec_conversion(self, click):
		#in the grid on the canvas
		grid_space = self.spec_view.transform.imap(click)
		#is the mouse over the spectrum spec_view area?
		if self.click_on_widget(click, self.spec_view):
			scene_space = self.spec_view.scene.transform.imap(grid_space)
			return self.spectra[0].mel_transform.imap(scene_space)
			
	def pt_spec_conversion(self, pt):
		#converts a point from Hz space to screen space
		melspace = self.spectra[0].mel_transform.map(pt)
		scene_space = self.spec_view.scene.transform.map(melspace)
		return self.spec_view.transform.map(scene_space)
	

		
def to_mel(val):
	### just to set the image size correctly	
	return np.log(val / 700 + 1) * 1127

def to_Hz(val):
	### just to set the image size correctly	
	return (np.exp(val / 1127) - 1) * 700
	