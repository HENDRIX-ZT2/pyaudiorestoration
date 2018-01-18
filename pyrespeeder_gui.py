# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


import sys
import numpy as np
from vispy import scene, gloo, visuals, color, util
import soundfile as sf
from PyQt5 import QtGui, QtCore, QtWidgets

#custom modules
import vispy_ext
import fourier
import resampling
import wow_detection

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

class SpectrumPiece(scene.Image):
	"""
	The visualization of one part of the whole spectrogram.
	"""
	def __init__(self, texdata, parent):
		#flip the shape, then it is much more straightforward for the rest
		self._shape = (texdata.shape[1], texdata.shape[0])
		self.get_data = visuals.shaders.Function(norm_luminance)
		self.get_data['vmin'] = -80
		self.get_data['vmax'] = -40
		self.get_data['texture'] = gloo.Texture2D(texdata, format='luminance', internalformat='r32f', interpolation="linear")

		scene.Image.__init__(self, method='subdivide', grid=(1000,1), parent=parent)
		
		#maybe to overlay multiple channels? but only makes sense with single colors
		#self.set_gl_state('additive')#, cull_face=False)
		
		#set in the main program
		self.shared_program.frag['get_data'] = self.get_data
		
		#alternative: needs no external color map
		#self.shared_program.frag['color_transform'] = visuals.shaders.Function(simple_cmap)
		#self.shared_program.frag['color_transform'] = visuals.shaders.Function(color.get_colormap(colormap).glsl_map)
		
	def set_size(self, size):
		#not sure if update is needed
		self._shape = size
		self.update()
	
	def set_cmap(self, colormap):
		#update is needed
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

def to_mel(val):
	### just to set the image size correctly	
	return np.log(val / 700 + 1) * 1127

class TaskThread(QtCore.QThread):
	notifyProgress = QtCore.pyqtSignal(int)
	settings = "foo"
	def run(self):
		name, speed_curve, resampling_mode, frequency_prec, use_channels, dither, target_freq = self.settings
		resampling.run(name, speed_curve= speed_curve, resampling_mode = resampling_mode, frequency_prec=frequency_prec, use_channels=use_channels, dither=dither, target_freq=target_freq, prog_sig=self)
			
class ObjectWidget(QtWidgets.QWidget):
	"""
	Widget for editing OBJECT parameters
	"""
	settings_hard_changed = QtCore.pyqtSignal(name='objectChanged')
	settings_soft_changed = QtCore.pyqtSignal(name='objectChanged2')

	def __init__(self, parent=None):
		super(ObjectWidget, self).__init__(parent)
		
		self.parent = parent
		
		self.filename = ""
		
		self.open_b = QtWidgets.QPushButton('Open Audio', self)
		self.open_b.clicked.connect(self.open_audio)
		
		self.save_b = QtWidgets.QPushButton('Save Traces', self)
		self.save_b.clicked.connect(self.save_traces)
		
		fft_l = QtWidgets.QLabel("FFT Size")
		self.fft_c = QtWidgets.QComboBox(self)
		self.fft_c.addItems(list(("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768")))
		self.fft_c.currentIndexChanged.connect(self.update_param_hard)
		
		overlap_l = QtWidgets.QLabel("FFT Overlap")
		self.overlap_c = QtWidgets.QComboBox(self)
		self.overlap_c.addItems(list(("1", "2", "4", "8")))
		self.overlap_c.currentIndexChanged.connect(self.update_param_hard)
		
		cmap_l = QtWidgets.QLabel("Colors")
		self.cmap_c = QtWidgets.QComboBox(self)
		self.cmap_c.addItems(sorted(color.colormap.get_colormaps().keys()))
		self.cmap_c.currentIndexChanged.connect(self.update_param_soft)

		self.resample_b = QtWidgets.QPushButton("Resample")
		self.resample_b.clicked.connect(self.run_resample)
		
		mode_l = QtWidgets.QLabel("Resampling Mode")
		self.mode_c = QtWidgets.QComboBox(self)
		self.mode_c.addItems(("Linear", "Expansion", "Sinc", "Windowed Sinc", "Blocks"))
		self.mode_c.currentIndexChanged.connect(self.update_resampling_presets)
		
		trace_l = QtWidgets.QLabel("Tracing Mode")
		self.trace_c = QtWidgets.QComboBox(self)
		self.trace_c.addItems(("Center of Gravity","Pitchtrack",))
		#self.trace_c.currentIndexChanged.connect(self.update_param_soft)
		
		
		prec_l = QtWidgets.QLabel("Precision")
		self.prec_s = QtWidgets.QDoubleSpinBox()
		self.prec_s.setRange(0.0001, 100.0)
		self.prec_s.setSingleStep(0.1)
		self.prec_s.setValue(0.01)
		
		self.progressBar = QtWidgets.QProgressBar(self)
		self.progressBar.setRange(0,100)
		
		
		gbox = QtWidgets.QGridLayout()
		
		#column 01
		gbox.addWidget(self.open_b, 0, 0)
		gbox.addWidget(self.save_b, 0, 1)
		
		gbox.addWidget(fft_l, 1, 0)
		gbox.addWidget(self.fft_c, 1, 1)
		
		gbox.addWidget(overlap_l, 2, 0)
		gbox.addWidget(self.overlap_c, 2, 1)
		
		gbox.addWidget(cmap_l, 3, 0)
		gbox.addWidget(self.cmap_c, 3, 1)
		
		#column 23
		gbox.addWidget(trace_l, 0, 2)
		gbox.addWidget(self.trace_c, 0, 3)
		
		#column 45
		gbox.addWidget(self.resample_b, 0, 5)
		
		gbox.addWidget(prec_l, 1, 4)
		gbox.addWidget(self.prec_s, 1, 5)
		
		gbox.addWidget(mode_l, 2, 4)
		gbox.addWidget(self.mode_c, 2, 5)
		
		self.myLongTask = TaskThread()
		self.myLongTask.notifyProgress.connect(self.onProgress)
		
		gbox.addWidget(self.progressBar, 3, 4, 1, 2 )

		vbox = QtWidgets.QVBoxLayout()
		vbox.addLayout(gbox)
		vbox.addStretch(1.0)

		self.setLayout(vbox)
	
	def onProgress(self, i):
		self.progressBar.setValue(i)
		
	def open_audio(self):
		f = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Audio files (*.flac *.wav)")
		if f:
			#pyqt5 returns a tuple
			self.filename = f[0]
			self.settings_hard_changed.emit()
			data = resampling.read_trace(self.filename)
			for offset, times, freqs in data:
				TraceLine(self.parent.canvas, times, freqs, offset=offset)
	
	def save_traces(self):
		#get the data from the traces and save it
		data = []
		for line in self.parent.canvas.lines:
			data.append( (line.offset, line.times, line.freqs) )
		print("Saved",len(data),"traces")
		if data:
			resampling.write_trace(self.filename, data)
	
	def run_resample(self):
		mode = self.mode_c.currentText()
		prec = self.prec_s.value()
		#make a copy to prevent unexpected side effects
		speed_curve = np.array(self.parent.canvas.master_speed.data)
		print("Resampling",self.filename, mode, prec)
		self.myLongTask.settings = (self.filename, speed_curve, mode, prec, [0,], "Diffused", None)
		self.myLongTask.start()
		#resampling.run(self.filename, speed_curve = speed_curve, resampling_mode=mode, frequency_prec=prec, use_channels=[0,], dither="Diffused", target_freq=None)
			
	def update_resampling_presets(self, option):
		mode = self.mode_c.currentText()
		if mode == "Expansion":
			self.prec_s.setValue(5.0)
		elif mode == "Blocks":
			self.prec_s.setValue(0.01)
		elif mode == "Sinc":
			self.prec_s.setValue(0.05)
		elif mode == "Windowed Sinc":
			self.prec_s.setValue(0.05)
		elif mode == "Linear":
			self.prec_s.setValue(0.01)
		
	def update_param_hard(self, option):
		self.settings_hard_changed.emit()
		
	def update_param_soft(self, option):
		self.settings_soft_changed.emit()


class MainWindow(QtWidgets.QMainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)

		self.resize(700, 500)
		self.setWindowTitle('pyrespeeder 2.0')

		splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

		self.canvas = Canvas()
		self.canvas.create_native()
		self.canvas.native.setParent(self)

		self.props = ObjectWidget(parent=self)
		splitter.addWidget(self.props)
		splitter.addWidget(self.canvas.native)

		self.setCentralWidget(splitter)
		self.props.settings_hard_changed.connect(self.update_settings_hard)
		self.props.settings_soft_changed.connect(self.update_settings_soft)

	def update_settings_hard(self):
		self.canvas.set_data_hard(self.props.filename,
								 fft_size = int(self.props.fft_c.currentText()),
								 fft_overlap = int(self.props.overlap_c.currentText()))
		#also force a soft update here
		self.update_settings_soft()
		
	def update_settings_soft(self):
		self.canvas.set_data_soft(cmap=self.props.cmap_c.currentText())	

class MasterSpeedLine:
	"""Stores and displays the average, ie. master speed curve."""
	def __init__(self, vispy_canvas):
		
		self.vispy_canvas = vispy_canvas
		
		#create the speed curve visualization
		self.data = np.ones((2, 2), dtype=np.float32)
		self.data[:, 0] = (0, 999)
		self.data[:, 1] = (1, 1)
		self.line_speed = scene.Line(pos=self.data, color=(1, 0, 0, 1), method='gl')
		self.line_speed.parent = vispy_canvas.speed_view.scene
		
	def update(self):
		num = self.vispy_canvas.num_ffts
		#get the times at which the average should be sampled
		times = np.linspace(0, num * self.vispy_canvas.hop / self.vispy_canvas.sr, num=num)
		#create the array for sampling
		out = np.ones((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
		#lerp and sample all lines, use NAN for missing parts
		for i, line in enumerate(self.vispy_canvas.lines):
			line_sampled = np.interp(times, line.times, line.speed, left = np.nan, right = np.nan)
			out[:, i] = line_sampled
		#take the mean and ignore nans
		mean_with_nans = np.nanmean(out, axis=1)
		mean_with_nans[np.isnan(mean_with_nans)]=1
		#set the output data
		self.data = np.ones((len(times), 2), dtype=np.float32)
		self.data[:, 0] = times
		self.data[:, 1] = mean_with_nans
		self.line_speed.set_data(pos=self.data)

class TraceLine:
	"""Stores and visualizes a trace fragment, including its speed offset."""
	def __init__(self, vispy_canvas, times, freqs, offset=0.0):
		
		self.vispy_canvas = vispy_canvas
		self.vispy_canvas.lines.append(self)
		
		self.offset = offset
		
		self.times = np.asarray(times)
		self.freqs = np.asarray(freqs)
		self.speed = freqs / np.mean(freqs) + offset
		
		self.center = np.array( (np.mean(self.times), np.mean(self.speed)) )
		print(self.center)
		
		data = np.ones((len(times), 3), dtype=np.float32)*-2
		data[:, 0] = times
		data[:, 1] = freqs
		
		#create the spectral visualization
		self.line_spec = scene.Line(pos=data, color=(1, 1, 1, 1), method='gl')
		#the data is in Hz, so to visualize correctly, it has to be mel'ed
		self.line_spec.transform =	vispy_ext.MelTransform()
		self.line_spec.parent = vispy_canvas.spec_view.scene
		
		#create the speed curve visualization
		self.speed_data = np.ones((len(times), 2), dtype=np.float32)
		self.speed_data[:, 0] = self.times
		self.speed_data[:, 1] = self.speed
		self.line_speed = scene.Line(pos=self.speed_data, color=(1, 1, 1, .5), method='gl')
		self.line_speed.parent = vispy_canvas.speed_view.scene
		self.vispy_canvas.master_speed.update()
		
	def set_offset(self, offset):
		print("offset",offset)
		self.offset += offset
		self.center[1] += offset
		self.speed += offset
		print("new center",self.center)
		self.speed_data[:, 1] = self.speed
		self.line_speed.set_data(pos = self.speed_data)
		self.vispy_canvas.master_speed.update()
		
	def remove(self):
		self.line_speed.parent = None
		self.line_spec.parent = None
		#note: this has to search the list
		self.vispy_canvas.lines.remove(self)
		self.vispy_canvas.master_speed.update()
		
	
class Canvas(scene.SceneCanvas):

	def __init__(self):
		
		#some default dummy values
		self.filename = "None"
		cm = "fire"
		self.vmin = -80
		self.vmax = -40
		
		self.last_click = None
		self.fft_size = 1024
		self.hop = 256
		self.sr = 44100
		self.num_ffts = 0
		
		self.MAX_TEXTURE_SIZE = None
		
		scene.SceneCanvas.__init__(self, keys="interactive", size=(1024, 512), bgcolor="grey")
		
		self.unfreeze()
		
		grid = self.central_widget.add_grid(margin=10)
		grid.spacing = 0

		#self.header = scene.Label(self.filename, color='white')
		#self.header.height_max = 30
		
		#speed chart
		self.speed_yaxis = scene.AxisWidget(orientation='left',
								 axis_label='%',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.speed_yaxis.width_max = 55
		# self.speed_xaxis = scene.AxisWidget(orientation='bottom',
								 # axis_label='sec',
								 # axis_font_size=8,
								 # axis_label_margin=35,
								 # tick_label_margin=5)
		# self.speed_xaxis.height_max = 55
								 
		
		#spectrum
		self.spec_yaxis = scene.AxisWidget(orientation='left',
								 axis_label='Hz',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.spec_yaxis.width_max = 55
		self.spec_xaxis = scene.AxisWidget(orientation='bottom',
								 axis_label='sec',
								 axis_font_size=8,
								 axis_label_margin=35,
								 tick_label_margin=5)
		self.spec_xaxis.height_max = 55

		top_padding = grid.add_widget(row=0)
		top_padding.height_max = 20
		
		right_padding = grid.add_widget(row=1, col=2, row_span=1)
		right_padding.width_max = 70

		#create the color bar display
		self.colorbar_display = scene.ColorBarWidget(label="Gain [dB]", clim=(self.vmin, self.vmax),
										   cmap=cm, orientation="right",
											border_width=1)
		self.colorbar_display.label.font_size = 10
		self.colorbar_display.label.color = "white"

		self.colorbar_display.ticks[0].font_size = 8
		self.colorbar_display.ticks[1].font_size = 8
		self.colorbar_display.ticks[0].color = "white"
		self.colorbar_display.ticks[1].color = "white"
		
		#grid.add_widget(self.header, row=0, col=0, col_span=2)
		grid.add_widget(self.speed_yaxis, row=1, col=0)
		#grid.add_widget(self.speed_xaxis, row=1, col=1)
		grid.add_widget(self.spec_yaxis, row=2, col=0)
		grid.add_widget(self.spec_xaxis, row=3, col=1)
		self.speed_view = grid.add_view(row=1, col=1, border_color='white')
		self.spec_view = grid.add_view(row=2, col=1, border_color='white')
		grid.add_widget(self.colorbar_display, row=2, col=2)
		self.spec_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0, 10, 10), )
		self.speed_view.camera = vispy_ext.PanZoomCameraExt(rect=(0, 0.95, 10, 1.05), )
		


						
		#TODO: make sure they are bound when started, not just after scrolling
		#self.speed_xaxis.link_view(self.speed_view)
		self.speed_yaxis.link_view(self.speed_view)
		self.spec_xaxis.link_view(self.spec_view)
		self.spec_yaxis.link_view(self.spec_view)
		
		
		self.images = []
		self.lines = []
		self.fft_storage = {}
		
		self.master_speed = MasterSpeedLine(self)
		
		self.freeze()
		
		
	#fast stuff that does not require rebuilding everything
	def set_data_soft(self, cmap="fire"):
		for image in self.images:
			image.set_cmap(cmap)
		self.colorbar_display.cmap = cmap
	
		
	def set_data_hard(self, filename, fft_size = 256, fft_overlap = 1):
		if filename:
			
			#only set this one once, but can't set it in _init_
			if not self.MAX_TEXTURE_SIZE:
				self.MAX_TEXTURE_SIZE = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)
				print("self.MAX_TEXTURE_SIZE", self.MAX_TEXTURE_SIZE)
			
			#has the file changed?
			if self.filename != filename:
				#clear the fft storage!
				self.fft_storage = {}
				
				print("removing old lines")
				for i in reversed(range(0, len(self.lines))):
					#self.lines[i].parent = None
					#self.lines.pop(i)
					self.lines[i].remove()
					
			soundob = sf.SoundFile(filename)
			sr = soundob.samplerate
			
			hop = int(fft_size/fft_overlap)
			
			#set this for the tracers etc.
			self.fft_size = fft_size
			self.hop = hop
			self.sr = sr
			
			print("fft_size,hop",fft_size,hop)
			
			#store the FFTs for fast shuffling around
			#TODO: analyze the RAM consumption of this
			#TODO: perform FFT at minimal hop and get shorter hops via stride operation ::x
			k = (fft_size, hop)
			if k not in self.fft_storage:
				print("storing new fft")
				signal = soundob.read(always_2d=True)[:,0]
				#this will automatically zero-pad the last fft
				imdata = fourier.stft(signal, fft_size, hop, "hann")
				#imdata = util.fourier.stft(signal, fft_size, hop, sr, "hann")
				#get the magnitude spectrum
				imdata = np.abs(imdata)
				#change to dB scale later, for the tracers
				#imdata = 20 * np.log10(imdata)
				#clamping the data to 0,1 range happens in the vertex shader
				
				#now store this for retrieval later
				self.fft_storage[k] = imdata.astype('float32')
			#retrieve the FFT data
			imdata = self.fft_storage[k]
			num_bins, num_ffts = imdata.shape
			self.num_ffts = num_ffts

			#has the file changed?
			if self.filename != filename:
				print("file has changed!")
				self.filename = filename
				#self.header.text = filename
				
				#(re)set the spec_view
				#only the camera dimension is mel'ed, as the image gets it from its transform
				self.speed_view.camera.rect = (0, 0, num_ffts * hop / sr, 2)
				self.spec_view.camera.rect = (0, 0, num_ffts * hop / sr, to_mel(sr//2))
				#link them, but use custom logic to only link the x view
				self.spec_view.camera.link(self.speed_view.camera)
				
			#TODO: is this as efficient as possible?
			print("removing old images")
			for i in reversed(range(0, len(self.images))):
				self.images[i].parent = None
				self.images.pop(i)
				
			#spectra may only be of a certain size, so split them
			#to support 2**17 FFT sizes, it would also need to split along the Y axis
			for i in range(0, num_ffts, self.MAX_TEXTURE_SIZE):
				imdata_piece = imdata[:,i:i+self.MAX_TEXTURE_SIZE]
				
				#do the dB conversion here because the tracers don't like it
				self.images.append(SpectrumPiece(20 * np.log10(imdata_piece), parent=self.spec_view.scene))
				
				self.images[-1].set_clims(self.vmin, self.vmax)
				self.images[-1].set_size((imdata_piece.shape[1]*hop/sr, sr//2))
				#add this piece's offset with STT
				self.images[-1].transform = visuals.transforms.STTransform( translate=(i * hop / sr, 0)) * vispy_ext.MelTransform()
			
		
	def on_mouse_wheel(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		
		#colorbar scroll
		if self.click_on_widget(click, self.colorbar_display):
			grid_space = self.colorbar_display._colorbar.transform.imap(click)
			dim = self.colorbar_display.size
			d = event.delta
			self.vmin, self.vmax = self.colorbar_display.clim
			
			#now split Y in three parts
			a = dim[1]/3
			b = a*2
			if grid_space[1] < a:
				self.vmax += d[1]
			elif a < grid_space[1] < b:
				self.vmin += d[1]
				self.vmax -= d[1]
			elif b < grid_space[1]:
				self.vmin += d[1]

			self.colorbar_display.clim = (int(self.vmin), int(self.vmax))
			for image in self.images:
				image.set_clims(self.vmin, self.vmax)
				
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
			#TODO: do we need MEL space here, but that also needs proper axis ticks first
			c = (self.spec_view.camera.center[0], scene_space[1])
			self.spec_view.camera.zoom((1, (1 + self.spec_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)
		
		#speed Y axis scroll
		if self.click_on_widget(click, self.speed_yaxis):
			#the center of zoom should be assigned a new y coordinate
			grid_space = self.speed_view.transform.imap(click)
			scene_space = self.speed_view.scene.transform.imap(grid_space)
			c = (self.speed_view.camera.center[0], scene_space[1])
			self.speed_view.camera.zoom((1, (1 + self.speed_view.camera.zoom_factor) ** (-event.delta[1] * 30)), c)

	def on_mouse_press(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		if "Control" in event.modifiers:
			self.last_click = click
		else:
			self.last_click = None
		if "Alt" in event.modifiers:
			#in speed view?
			c = self.click_speed_conversion(click)
			if c is not None:
				closest_line = self.get_closest_line( (c[0], c[1]) )
				if closest_line: closest_line.remove()
		
	def on_mouse_release(self, event):
		#coords of the click on the vispy canvas
		click = np.array([event.pos[0],event.pos[1],0,1])
		if self.last_click is not None:
			a = self.click_spec_conversion(self.last_click)
			b = self.click_spec_conversion(click)
			#are they in spec_view?
			if a is not None and b is not None:
				t0, t1 = sorted((a[0], b[0]))
				f0, f1 = sorted((a[1], b[1]))
				
				fft_key = (self.fft_size, self.hop)
				times, freqs = wow_detection.trace_cog(self.fft_storage[fft_key], fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
				
				if freqs and np.nan not in freqs:
					TraceLine(self, times, freqs)
				return
			
			#or in speed view?
			a = self.click_speed_conversion(self.last_click)
			b = self.click_speed_conversion(click)
			if a is not None and b is not None:
				diff = b[1]-a[1]
				closest_line = self.get_closest_line( (a[0], a[1]) )
				if closest_line: closest_line.set_offset(diff)
				
				
	def get_closest_line(self, pt):
		#returns the line (if any exists) whose center (including any offsets) is closest to pt
		if self.lines:
			A = np.array([line.center for line in self.lines])
			ind = np.array([np.linalg.norm(x+y) for (x,y) in A-pt]).argmin()
			return self.lines[ind]
	
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
			#careful, we need a spectrum already
			if self.images[0]:
				#in fact the simple Y mel transform would be enough in any case
				#but this would also support other transforms
				return self.images[0].transform.imap(scene_space)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
	appQt = QtWidgets.QApplication(sys.argv)
	win = MainWindow()
	win.show()
	appQt.exec_()
