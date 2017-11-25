from tkinter import BooleanVar,StringVar,IntVar,Tk,ttk,Menu,filedialog,Checkbutton,Text,messagebox
import soundfile as sf
import os
import sys
import resampy
from numpy import asarray, arange, interp, repeat, rint, int64, resize, loadtxt, median, mean, random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class LineBuilder:
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
			

def draw_spec(filename, fft_size = 8192, fft_overlap = 0.5, use_channel=0):
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True)
	channels = soundob.channels
	sr = soundob.samplerate

	fig = plt.figure()
	hop = int(fft_overlap*fft_size)
	D = librosa.amplitude_to_db(librosa.stft(signal[:,use_channel], n_fft=fft_size, hop_length=hop), ref=np.max)
	librosa.display.specshow(D, sr=sr, x_axis="time", y_axis='log', hop_length=hop)#, shading="gouraud")
	plt.colorbar(format='%+2.0f dB')
	plt.title('Log-frequency power spectrogram')

	#fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('CTRL+Click: add marker, ALT+Click: delete closest marker, DEL+Click: delete all markers. Close window to save.')
	line, = ax.plot([], [])  # empty line
	linebuilder = LineBuilder(line)
	
	plt.show()
	
	if not linebuilder.xs:
		print("No speed to write!")
		return
	#write the data to the speed file
	print("Writing speed data")
	speedfilename = filename.rsplit('.', 1)[0]+".speed"
	outstr = "\n".join([str(t)+" "+str(f) for t, f in zip(linebuilder.xs, linebuilder.ys)])
	text_file = open(speedfilename, "w")
	text_file.write(outstr)
	text_file.close()
	

def work_resample(filename, resampling_mode = "Blocks", frequency_prec=0.01, target_freq = None, use_channels = [0,], dither=True):

	#read the file
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True)
	channels = soundob.channels
	sr = soundob.samplerate
	#always_2d=True would make some things easier?
	
	print('Analyzing ' + filename + '...')
	print('Shape:', signal.shape)
	
	#return
	block_size = int(100 / frequency_prec)
	resampling_factor = 100 / frequency_prec
	
	#here we can use larger blocks without losing frequency precision (?), this speeds it up but be careful with the memory limit
	#end value is fixed -> hence the more frequency precision, the smaller the block gets
	if resampling_mode == "Feaster":
		block_size = int(1000000 / resampling_factor)
	
	#user/debugging info
	print(resampling_mode)
	if resampling_mode == "Blocks":
		print("frequency precision:",frequency_prec,"%")
		print("temporal precision (different speed every X sec)",block_size/sr)
	if resampling_mode == "Feaster":
		print("frequency precision:",frequency_prec,"%")
		print("expansion_factor",resampling_factor)
		print("raw block_size:",block_size)
		print("expanded block_size:",block_size*resampling_factor)
	
	
	overlap = 0
	blocks = []
	for n in range(0, len(signal), block_size):
		blocks.append((n, block_size+n+overlap))
	print("Num Blocks",len(blocks))
	
	#return

	#these are measured frequencies in Hz and their times (converted to samples on the fly)
	speedfilename = filename.rsplit('.', 1)[0]+".speed"
	if not os.path.isfile(speedfilename):
		print("Speed file does not exist!")
		return
	speeds = loadtxt(speedfilename)
	if not len(speeds):
		print("No speed data in file!")
		return
	
	times = speeds[:,0]*sr
	freqs = speeds[:,1]

	#this is the frequency all measured frequencies should have in the end
	#if not supplied, use mean or median of freqs
	if not target_freq:
		target_freq	= median(freqs)

	#divide them to get the speed ratio for sample expansion
	speeds = freqs/target_freq
	
	#lerp the samples to the length of the signal
	#!this does not handle extrapolation!
	speed_samples_r = interp(arange(0, len(signal)), times, speeds, left=speeds[0], right=speeds[-1])
	
	#resample on mono channels and export each separately as repeat does not like more dimensions
	for channel in use_channels:
		print('Processing channel ',channel)
		outfilename = filename[:-4]+str(channel)+'.wav'
		with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
			for block_start, block_end in blocks:
				#print("Writing",block_start,"to",block_end)
				
				#create a mono array for the current block and channel
				#only multichannel files are ndimensional, so mono needs its own special case
				signal_block = signal[block_start:block_end, channel]
				
				#apply the different methods
				if resampling_mode == "Feaster":
					
					#multiply each sample by the expansion factor
					signal_block_s = resampling_factor * speed_samples_r[block_start:block_end]
					
					#add -.5 to +.5 random noise before quantization to minimizes the global error at the cost of more local error
					if dither:
						signal_block_s = random.rand(len(signal_block_s))-.5 + signal_block_s
						
					#the factor has to be rounded and converted to int64 for repeat to accept it
					speed_samples_i = rint(signal_block_s).astype(int64)
					#repeat each sample by the resampling_factor * the speed factor for that sample
					upsampled = repeat(signal_block, speed_samples_i)
					
					#now divide it by the resampling_factor to return to the original median speed
				elif resampling_mode == "Blocks":
					#take a block of the signal
					#tradeoff between frequency and time precision -> good for wow, unusable for flutter
					upsampled = signal_block
					
					#here we do not use the specified resampling factor but instead make our own from the average speed of the current block
					resampling_factor = 1/ mean(speed_samples_r[block_start:block_end])
				
				#this is fast and does not cut off high freqs
				resampled = resampy.resample(upsampled, resampling_factor, 1, filter='sinc_window', num_zeros=4)
				
				#fast attenuation of clicks, not perfect but better than nothing
				resampled[0] = signal_block[0]
				resampled[1] = signal_block[1]
				resampled[-2] = signal_block[-2]
				resampled[-1] = signal_block[-1]
				
				#resample write the output block to the audio file
				outfile.write(resampled)
	print("Done!\n")	
	#TODO:
	#cover up block cuts, with overlap
	#for blocks: adaptive resolution according to speed derivation. more derivation = shorter blocks

#Feaster:
#the more frequency precision (smaller percentage), the longer it takes
#temporal precision is -theoretically- unaffected
#but practically, input blocks become shorter for more precision, as output size should remain stable to avoid overflow

# #Blocks
# #classic tradeoff: the more frequency precision, the faster
# #at the cost of less temporal precision
# resampling_mode = "Blocks"
# #resampling_mode = "Feaster"

# #filename = "C:/Users/arnfi/Desktop/10. Lady Madonna.flac"
# filename = "C:/Users/arnfi/Desktop/test.wav"
# resample(filename, resampling_mode = resampling_mode, maxchannels = 1, target_freq = None)

try:
    approot = os.path.dirname(os.path.abspath(__file__))
except NameError:  # We are the main py2exe script, not a module
    import sys
    approot = os.path.dirname(os.path.abspath(sys.argv[0]))

class Application:
		
	def dither_changed(self):
		dither = self.var_dither.get()
		#print("dither?",dither)
		if self.var_mode.get() == "Feaster":
			if dither:
				self.var_freq_prec.set(str(float(self.var_freq_prec.get())*10))
			else:
				self.var_freq_prec.set(str(float(self.var_freq_prec.get())/10))
			
	def mode_changed(self, mode):
		#print("new mode",mode)
		if mode == "Feaster":
			self.var_dither.set(True)
			self.var_freq_prec.set("5.0")
			self.var_temp_prec.set("?")
		elif mode == "Blocks":
			self.var_freq_prec.set("0.01")
			self.var_temp_prec.set("0.226s")
		
	def open_file(self):
		file_path = filedialog.askopenfilename(filetypes = [('WAV', '.wav'), ('FLAC', '.flac')], defaultextension=".wav", initialdir="", parent=self.parent, title="Open Audio File" )
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
			draw_spec(self.var_file.get(), fft_size = fft_size, fft_overlap = fft_overlap, use_channel=use_channel)
			
	def run_resample(self):
		if self.var_file.get():
			use_channels = [k for k, v in self.vars_channels.items() if v.get()]
			mode = self.var_mode.get()
			freq_prec = float(self.var_freq_prec.get())
			dither = self.var_dither.get()
			print(use_channels,mode,freq_prec,dither)
			work_resample(self.var_file.get(), resampling_mode=mode, frequency_prec=freq_prec, use_channels=use_channels, dither= dither)
	
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
		modes = ("Feaster", "Blocks")
		ttk.Label(self.parent, text="Resampling Mode").grid(row=3, column=0, sticky='nsew')
		ttk.OptionMenu(self.parent, self.var_mode, modes[0], *modes, command=self.mode_changed).grid(row=3, column=1, sticky='nsew')
		
		self.var_freq_prec = StringVar()
		ttk.Label(self.parent, text="Frequency Precision").grid(row=4, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_freq_prec).grid(row=4, column=1, sticky='nsew')
		self.var_temp_prec = StringVar()
		ttk.Label(self.parent, text="Temporal Precision").grid(row=5, column=0, sticky='nsew')
		ttk.Entry(self.parent, textvariable=self.var_temp_prec).grid(row=5, column=1, sticky='nsew')
		
		self.var_dither = BooleanVar()
		ttk.Checkbutton(self.parent, text="Use Dither", variable=self.var_dither, command=self.dither_changed).grid(sticky='nw', column=0, row=6)
				
		ttk.Button(self.parent, text="Run Spectrum", command=self.run_spectrum).grid(row=7, column=0, sticky='nsew')
		ttk.Button(self.parent, text="Run Resample", command=self.run_resample).grid(row=7, column=1, sticky='nsew')
		
		self.var_fft_size.set("4096")
		self.var_fft_overlap.set("0.125")
		self.var_mode.set(modes[0])
		self.mode_changed(modes[0])
		self.var_dither.set(True)
		
		
		
	
if __name__ == '__main__':
	app_root = Tk()
	app = Application(app_root)
	app_root.mainloop()