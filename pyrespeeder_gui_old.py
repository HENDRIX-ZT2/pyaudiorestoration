from tkinter import BooleanVar,StringVar,IntVar,Tk,ttk,Menu,filedialog,Checkbutton,Text,messagebox
import soundfile as sf
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

#custom modules
import resampling
import wow_detection

# try:
    # approot = os.path.dirname(os.path.abspath(__file__))
# except NameError:  # We are the main py2exe script, not a module
    # import sys
    # approot = os.path.dirname(os.path.abspath(sys.argv[0]))
	
class TraceBoxes(object):
	def __init__(self, ax, file_name, fft_size, hop, sr, D):
		self.ax = ax 
		self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
		self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
		self.rects = []
		self.lines = []
		self.file_name = file_name
		self.fft_size = fft_size
		self.hop = hop
		self.sr = sr
		self.spec = D
		

	def on_press(self, event):
		if event.key == 'delete':
			print("Removing")
			self.x0 = None
			self.y0 = None
			if self.rects:
				x = event.xdata
				y = event.ydata
				for i, rect in enumerate(self.rects):
					bb = rect.get_bbox()
					#print(bb)
					#is the click in this bb?
					if (bb.x0 < x < bb.x1 or bb.x0 > x > bb.x1) and (bb.y0 < y < bb.y1 or bb.y0 > y > bb.y1):
						#delete the rectangle
						self.rects[i].remove()
						#remove it from the list
						self.rects.pop(i)
						
						print(self.ax.lines)
						del self.ax.lines[i]
						#delete the trace
						#self.ax.lines.remove(0)
						#remove it from the list
						#self.lines.pop(i)
						self.ax.figure.canvas.draw()
						break
			
		elif event.key == 'control':
			print('Adding')
			self.x0 = event.xdata
			self.y0 = event.ydata
		
		#do nothing
		else:
			self.x0 = None
			self.y0 = None

	def on_release(self, event):
		#only add one if we are not deleting one
		if self.x0 and self.y0:
			#and only if we are in the plot
			if event.xdata and event.ydata:
				self.x1 = event.xdata
				self.y1 = event.ydata
				
				t0, t1 = sorted((self.x0, self.x1))
				f0, f1 = sorted((self.y0, self.y1))
				
				#todo: debug and report errors
				try:
					times, freqs = wow_detection.trace_cog(self.spec, fft_size = self.fft_size, hop = self.hop, sr = self.sr, fL = f0, fU = f1, t0 = t0, t1 = t1)
					self.lines.append(self.ax.plot(times, freqs))
					#create the artist: (x,y), w, h
					self.rects.append( Rectangle((self.x0, self.y0), self.x1 - self.x0, self.y1 - self.y0, alpha=.2) )
					#store it in the plot
					self.ax.add_patch(self.rects[-1])
					self.ax.figure.canvas.draw()	
				except:
					print("Error in Center of Gravity trace!")
							
	
class EditableLine:
	def __init__(self, line):
		self.line = line
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

	def redraw(self):
		self.line.figure.draw_artist(self.line)
		self.line.figure.canvas.update()
		#self.line.figure.canvas.flush_events()
	
	def __call__(self, event):
		#print('click', event)
		if event.inaxes!=self.line.axes: return
		if event.key == 'delete':
			self.xs = []
			self.ys = []
			self.line.set_data(self.xs, self.ys)
			#self.line.figure.canvas.draw()
			self.redraw()
		elif event.key == 'alt':
			#find the nearest point's index, and delete it from the stack
			if self.xs:
				pt = (event.xdata, event.ydata)
				A = np.array([(x,y) for x, y in zip(self.xs, self.ys)])
				
				ind = np.array([np.linalg.norm(x+y) for (x,y) in A-pt]).argmin()
				self.xs.pop(ind)
				self.ys.pop(ind)
				self.line.set_data(self.xs, self.ys)
				#self.line.figure.canvas.draw()
				self.redraw()
		elif event.key == 'control':
		
			#sort the data and add it inbetween
			
			self.xs.append(event.xdata)
			self.ys.append(event.ydata)
			
			x = np.array(self.xs)
			y = np.array(self.ys)
			inds = x.argsort()
			self.xs = list(x[inds])
			self.ys = list(y[inds])

			self.line.set_data(self.xs, self.ys)
			#self.line.figure.canvas.draw()
			self.redraw()
			

def draw_spec(filename, fft_size = 8192, fft_overlap = 0.5, use_channel=0, trace_mode="Manual"):
	print("Drawing spectrum")
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True)
	channels = soundob.channels
	sr = soundob.samplerate

	fig = plt.figure()
	hop = int(fft_overlap*fft_size)
	
	# #working mel scaled spectrum - but the accuracy is not perfect
	# S = librosa.feature.melspectrogram(signal[:,use_channel], n_fft=fft_size, hop_length=hop, n_mels = int(fft_size/4))
	# D = librosa.logamplitude(S, ref_power=np.max)
	# librosa.display.specshow(D, sr=sr, x_axis="time", y_axis='mel', hop_length=hop)#, shading="gouraud")
	
	D = librosa.amplitude_to_db(librosa.stft(signal[:,use_channel], n_fft=fft_size, hop_length=hop), ref=np.max)
	librosa.display.specshow(D, sr=sr, x_axis="time", y_axis='log', hop_length=hop)#, shading="gouraud")
	plt.colorbar(format='%+2.0f dB')
	plt.title('Log-frequency power spectrogram')

	ax = fig.add_subplot(111)
	ax.set_title('CTRL+Click: add marker, ALT+Click: delete closest marker, DEL+Click: delete all markers. Close window to save.')

	times =[]
	freqs =[]
	
	#can we load existing speed data?
	speedfilename = filename.rsplit('.', 1)[0]+".speed"
	if os.path.isfile(speedfilename):
		data = resampling.read_trace(filename)
		if trace_mode == "Manual":
			offset, times, freqs = resampling.flatten_trace(data)
			line, = ax.plot(times, freqs)
			linebuilder = EditableLine(line)
	if trace_mode == "COG":
		test = TraceBoxes(ax, filename, fft_size, hop, sr, D)
	
	plt.show()
	
	#get the data from the traces and save it
	data = []
	for line in ax.lines:
		data.append( (0, line.get_xdata(), line.get_ydata()) )
	if data:
		resampling.write_trace(filename, data)
	

class Application:
		
	def dither_changed(self, dither):
		#print("dither?",dither)
		if self.var_mode.get() == "Expansion":
			if dither == "None":
				#self.var_freq_prec.set(str(float(self.var_freq_prec.get())/10))
				self.var_freq_prec.set("0.5")
			else:
				#self.var_freq_prec.set(str(float(self.var_freq_prec.get())*10))
				self.var_freq_prec.set("5.0")
			
	def mode_changed(self, mode):
		#print("new mode",mode)
		if mode == "Expansion":
			self.var_dither.set("Diffused")
			self.var_freq_prec.set("5.0")
			self.var_temp_prec.set("?")
		elif mode == "Blocks":
			self.var_dither.set("None")
			self.var_freq_prec.set("0.01")
			self.var_temp_prec.set("0.226s")
		elif mode == "Sinc":
			self.var_dither.set("None")
			self.var_freq_prec.set("0.05")
			self.var_temp_prec.set("?")
		elif mode == "Windowed Sinc":
			self.var_dither.set("None")
			self.var_freq_prec.set("0.05")
			self.var_temp_prec.set("?")
		
	def open_file(self):
		file_path = filedialog.askopenfilename(filetypes = [("Audio Files", (".wav", ".flac")), ('All Files', '.*'), ('WAV Files', '.wav'), ('FLAC Files', '.flac')], defaultextension=".wav", initialdir="", parent=self.parent, title="Open Audio File" )
		if file_path:
			#read the file
			soundob = sf.SoundFile(file_path)
			self.var_file.set(file_path)
			self.show_channels(soundob.channels)
			
	def run_spectrum(self):
		if self.var_file.get():
			#only use the first channel of those that are selected
			use_channel = [k for k, v in self.vars_channels.items() if v.get()][0]
			
			fft_size = int(self.var_fft_size.get())
			fft_overlap = float(self.var_fft_overlap.get())
			print(use_channel,fft_size,fft_overlap)
			draw_spec(self.var_file.get(), fft_size = fft_size, fft_overlap = fft_overlap, use_channel=use_channel, trace_mode="Manual")
			
	def run_cog_trace(self):
		if self.var_file.get():
			# #only use the first channel of those that are selected
			# use_channel = [k for k, v in self.vars_channels.items() if v.get()][0]
			
			# fft_size = int(self.var_fft_size.get())
			# fft_overlap = float(self.var_fft_overlap.get())
			# #fL = float(self.var_freq_lower.get())
			# #fU = float(self.var_freq_upper.get())
			# print(use_channel,fft_size,fft_overlap,fL,fU)
			# trace_cog(self.var_file.get(), fL=fL, fU=fU, use_channel=use_channel)#, fft_size = fft_size, fft_overlap = fft_overlap)
			use_channel = [k for k, v in self.vars_channels.items() if v.get()][0]
			
			fft_size = int(self.var_fft_size.get())
			fft_overlap = float(self.var_fft_overlap.get())
			print(use_channel,fft_size,fft_overlap)
			draw_spec(self.var_file.get(), fft_size = fft_size, fft_overlap = fft_overlap, use_channel=use_channel, trace_mode="COG")
			
	def run_resample(self):
		if self.var_file.get():
			use_channels = [k for k, v in self.vars_channels.items() if v.get()]
			mode = self.var_mode.get()
			freq_prec = float(self.var_freq_prec.get())
			dither = self.var_dither.get()
			try: target_freq = float(self.var_target_freq.get())
			except: target_freq = None
			print(use_channels,mode,freq_prec,dither, target_freq)
			starttime = time.clock()
			resampling.run(self.var_file.get(), resampling_mode=mode, frequency_prec=freq_prec, use_channels=use_channels, dither= dither, target_freq=target_freq)
			print('Finished Respeeding in %.2f seconds' %(time.clock()-starttime))
	
	def show_channels(self, num_channels):
		self.vars_channels={}
		try: self.b_properties.destroy()
		except: pass
		
		self.b_properties = ttk.LabelFrame(self.parent, text="Channels")
		self.b_properties.grid(row=0, column=3, rowspan=10, sticky='new')
		
		#all active by default?
		active = [0,]
		
		for i in range(0, num_channels):
			self.vars_channels[i] = BooleanVar()
			ttk.Checkbutton(self.b_properties, text=str(i), variable=self.vars_channels[i]).grid(sticky='nw', column=0, row=i)
			if i in active:
				self.vars_channels[i].set(True)
			else:
				self.vars_channels[i].set(False)
	
	def __init__(self, parent):
		#basic UI setup
		self.parent = parent
		#self.parent.geometry('640x480+100+100')
		self.parent.option_add("*Font", "Calibri 9")
		self.parent.title("pyrespeeder")
		
		ttk.Button(self.parent, text="Open File", command=self.open_file).grid(row=0, column=0, sticky='nsew')
		
		self.var_file = StringVar()
		ttk.Entry(self.parent, textvariable=self.var_file).grid(row=0, column=1, sticky='nsew')
		
		self.var_fft_size = StringVar()
		ttk.Label(self.parent, text="FFT Size").grid(row=1, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_fft_size).grid(row=1, column=1, sticky='nsew')
		
		self.var_fft_overlap = StringVar()
		ttk.Label(self.parent, text="FFT Overlap").grid(row=2, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_fft_overlap).grid(row=2, column=1, sticky='nsew')
		
		self.var_mode = StringVar()
		modes = ("Expansion", "Blocks", "Sinc", "Windowed Sinc")
		ttk.Label(self.parent, text="Resampling Mode").grid(row=3, column=0, sticky='nsew')
		ttk.OptionMenu(self.parent, self.var_mode, modes[0], *modes, command=self.mode_changed).grid(row=3, column=1, sticky='nsew')
		
		self.var_freq_prec = StringVar()
		ttk.Label(self.parent, text="Frequency Precision").grid(row=4, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_freq_prec).grid(row=4, column=1, sticky='nsew')
		self.var_temp_prec = StringVar()
		ttk.Label(self.parent, text="Temporal Precision").grid(row=5, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_temp_prec).grid(row=5, column=1, sticky='nsew')
		
		self.var_target_freq = StringVar()
		ttk.Label(self.parent, text="Target Freq. (Opt.)").grid(row=6, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_target_freq).grid(row=6, column=1, sticky='nsew')
		
		self.var_dither = StringVar()
		dithers = ("Diffused", "Random", "None")
		ttk.Label(self.parent, text="Expansion Dithering").grid(row=7, column=0, sticky='nsew')
		ttk.OptionMenu(self.parent, self.var_dither, dithers[0], *dithers, command=self.dither_changed).grid(row=7, column=1, sticky='nsew')
				
				
				
				
		ttk.Button(self.parent, text="Manual Trace", command=self.run_spectrum).grid(row=8, column=0, sticky='nsew')
		ttk.Button(self.parent, text="Center of Gravity", command=self.run_cog_trace).grid(row=8, column=1, sticky='nsew')
		
		# self.var_freq_lower = StringVar()
		# ttk.Label(self.parent, text="Lower End (Hz)").grid(row=9, column=0, sticky='nsew')
		# ttk.Entry(self.parent, textvariable=self.var_freq_lower).grid(row=9, column=1, sticky='nsew')
		# self.var_freq_upper = StringVar()
		# ttk.Label(self.parent, text="Upper End (Hz)").grid(row=10, column=0, sticky='nsew')
		# ttk.Entry(self.parent, textvariable=self.var_freq_upper).grid(row=10, column=1, sticky='nsew')
		
		ttk.Button(self.parent, text="Resample!", command=self.run_resample).grid(row=11, column=0, columnspan=2, sticky='nsew')
		
		self.var_fft_size.set("4096")
		self.var_fft_overlap.set("0.125")
		self.var_mode.set(modes[0])
		self.mode_changed(modes[0])
		self.var_dither.set(dithers[0])
		self.dither_changed(dithers[0])
		
		#self.var_freq_lower.set("2260")
		#self.var_freq_upper.set("2320")
		
		
		
	
if __name__ == '__main__':
	app_root = Tk()
	app = Application(app_root)
	app_root.mainloop()