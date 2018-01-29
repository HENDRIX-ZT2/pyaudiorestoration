import numpy as np
import resampy
import soundfile as sf
import os

def sinc_interp_windowed(y_in, x_in, x_out, write_after=10000, NT = 100):
	"""
	base by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
	Interpolate the signal to the new points using a hann-windowed sinc kernel
	>input:
	x_in			time points x is defined on
	y_in		 	input signal column vector or matrix, with a signal in each row
	x_out			points to evaluate the new signal on
	write_after		yield a block after X output samples
	NT				samples to take before and after the central sample
	>output:
	y				 the interpolated signal at points x_out
	"""
	y = np.zeros(write_after+1)
	offset = 0
	win_func = np.hanning(2*NT)
	in_len = len(x_in)
	out_len = len(x_out)
	y = np.zeros(write_after)
	period_from = x_in[1]-x_in[0]
	while offset < out_len:
		#print("piece at", offset)
		outind = 0
		for i in range(offset, offset+write_after):
			#now we are at the end and we can yield the rest of the piece
			if i == out_len-1:
				#print("end",i,outind)
				#print(len(y[0:outind]))
				yield y[0:outind]
				offset = out_len
				break
			p = x_out[i]
			#map the output to the input
			#error diffusion here makes no significant difference
			ind = int(round(p))
			lower = max(0, ind-NT)
			upper = min(ind+NT, in_len)
			length = upper - lower
			
			#fc is the cutoff frequency expressed as a fraction of the nyquist freq
			#we need anti-aliasing when sampling rate is bigger than before, ie. exceeding the nyquist frequency
			#could skip this calculation and get it from the speed curve instead?
			period_to = x_out[i+1]-p
			fc = min(period_from/period_to, 1)
			
			#use Hann window to reduce the prominent sinc ringing of a rectangular window
			#(http://www-cs.engr.ccny.cuny.edu/~wolberg/pub/crc04.pdf, p. 11ff)
			#http://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
			#claims that only hamming and blackman are worth using? my experiments look best with hann window
			si = np.sinc ((x_in[lower:upper] - p) * fc) * fc
			y[outind] = np.sum(si * y_in[lower:upper] * win_func[0:length])
		
			outind+=1
		#we may have to exit here to avoid sending another block
		if offset == out_len:
			break
		offset+=write_after
		yield y

def write_speed(filename, speed_curve):
	#only for testing
	speedfilename = filename.rsplit('.', 1)[0]+".npy"
	np.save(speedfilename, speed_curve, allow_pickle=True, fix_imports=True)

def read_speed(filename):
	#only for testing
	speedfilename = filename.rsplit('.', 1)[0]+".npy"
	return np.load(speedfilename)
	
def write_trace(filename, data):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	data:	 a list of (times, frequencies) lists
	"""
	
	#write the data to the speed file
	print("Writing trace data")
	speedfilename = filename.rsplit('.', 1)[0]+".speed"
	outstr = ""
	for offset, times, frequencies in data:
		outstr+= "? "+str(offset)+"\n"+"\n".join([str(t)+" "+str(f) for t, f in zip(times, frequencies)])+"\n"
	text_file = open(speedfilename, "w")
	text_file.write(outstr)
	text_file.close()

def read_trace(filename):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	returns:
	data:	 a list of (times, frequencies) lists
	"""
	
	#write the data to the speed file
	print("Reading speed data")
	speedfilename = filename.rsplit('.', 1)[0]+".speed"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for l in text_file:
				#just for completeness
				if l:
					if "?" in l:
						offset = float(l.split(" ")[1])
						data.append((offset, [], []))
					else:
						s = l.split(" ")
						data[-1][1].append(float(s[0]))
						data[-1][2].append(float(s[1]))
	return data
	
def write_regs(filename, data):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	data:	 a list of sine parameters
	"""
	
	#write the data to the speed file
	print("Writing regression data")
	speedfilename = filename.rsplit('.', 1)[0]+".sin"
	outstr = "\n".join([" ".join([str(v) for v in values]) for values in data])
	text_file = open(speedfilename, "w")
	text_file.write(outstr)
	text_file.close()
	
def read_regs(filename):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	returns:
	data:	 a list of sine parameters
	"""
	
	#write the data to the speed file
	print("Reading regression data")
	speedfilename = filename.rsplit('.', 1)[0]+".sin"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for l in text_file:
				#just for completeness
				if l:
					data.append( [float(v) for v in l.split(" ")])
	return data

def run(filename, speed_curve=None, resampling_mode = "Blocks", frequency_prec=0.01, use_channels = [0,], dither="Random", patch_ends=False, prog_sig=None):

	if prog_sig:
		prog_sig.notifyProgress.emit(0)
		
	#read the file
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True)
	channels = soundob.channels
	sr = soundob.samplerate
	
	print('Analyzing ' + filename + '...')
	#print('Shape:', signal.shape)
	
	block_size = int(100 / frequency_prec)
	resampling_factor = 100 / frequency_prec
	
	#here we can use larger blocks without losing frequency precision (?), this speeds it up but be careful with the memory limit
	#end value is fixed -> hence the more frequency precision, the smaller the block gets
	if resampling_mode == "Expansion":
		block_size = int(1000000 / resampling_factor)
	
	#user/debugging info
	print(resampling_mode)
	if resampling_mode == "Blocks":
		print("frequency precision:",frequency_prec,"%")
		print("temporal precision (different speed every X sec)",block_size/sr)
	if resampling_mode == "Expansion":
		print("frequency precision:",frequency_prec,"%")
		print("expansion_factor",resampling_factor)
		print("raw block_size:",block_size)
		print("expanded block_size:",block_size*resampling_factor)
	
	times = speed_curve[:,0]
	#note: this expects a a linscale speed curve centered around 1 (= no speed change)
	speeds = speed_curve[:,1]
		
	if resampling_mode in ("Sinc", "Linear"):
	
		in_len = len(signal)
		out_len = int(in_len*np.mean(speeds))
		period = times[1]-times[0]
		print("Input length:",in_len,"samples")
		print("Output length:",out_len,"samples")
		print("Curve period:",period,"sec = ",period*sr,"samples")
		print("Be ware that a densely sampled speed curve causes inaccuracies with the current implementation")
		#most likely, the issue is in the cumsum, because the error increases towards the end
		#if the periods are long, say 4096, a 5min file does not show issues, and is -very- close to dithered expansion quality
		#if the periods are short, 64 samples, you get heavy issues at the ends > you have more periods and the error accumulates
		#dithering in the sinc interpolator does NOT improve it
		samples_out = np.cumsum(period*sr*speeds)
		samples_in = times*sr
		samples_in2 = np.arange(0, in_len)
		#here we create an array that maps every single sample of the output to a sub-sample time on the input
		samples_out2 = np.interp(np.arange(0, out_len), samples_out, samples_in)
		NT = 100
		write_after=100000
		#blocks are integers
		num_blocks = np.ceil( out_len/write_after )
		
		# #recode again?
		# #1) Firstly, each element f[b] of the vector f is associated to the central sample of its respective block;
		# #f_centers = times*sr
		# #2)  Secondly, the number of samples which will be reconstructed between each pair of consecutive central samples of f[b] and f[b + 1] is calculated, so the transition between these central samples be smooth.
		# f_sr = sr*speeds
		# for i in range(len(f_sr))
		# num_samples_per_second = (f_sr[i]+f_sr[i+1])/2*period
		# print(num_samples_per_second)
		# #how many samples in each segment?
		# return
	else:
		overlap = 0
		blocks = []
		for n in range(0, len(signal), block_size):
			blocks.append((n, block_size+n+overlap))
		num_blocks = len(blocks)
		#convert times to samples
		#lerp the samples to the length of the signal
		#!this does not handle extrapolation!
		speed_samples_r = np.interp(np.arange(0, len(signal)), times*sr, speeds)
			
		#do all processing of the speed samples right here to speed up multi channel files - the speed curve can be reused as is!
		if resampling_mode == "Expansion":
			#multiply each speed sample by the expansion factor
			speed_samples_r *= resampling_factor
			if dither == "Random":
				#add -.5 to +.5 random noise before quantization to minimizes the global error at the cost of more local error
				speed_samples_final = np.rint(random.rand(len(speed_samples_r))-.5 + speed_samples_r).astype(np.int64)
			elif dither == "Diffused":
				# the error of each sample is passed onto the next sample
				speed_samples_final = np.ones(len(speed_samples_r), dtype = np.int64)
				err = 0
				for i in range( len(speed_samples_r) ):
					inerr = speed_samples_r[i] + err
					out = round(inerr)
					err =  inerr-out
					speed_samples_final[i] = out
			else:
				#do not dither, just round
				speed_samples_final = np.rint(speed_samples_r).astype(np.int64)
				
		else:
			#do not dither, do not round, just copy
			speed_samples_final = speed_samples_r
	
	print("Num Blocks",num_blocks)
	#there are no half blocks...
	prog_fac = 100 / num_blocks*len(use_channels)
	progress = 0
	#resample on mono channels and export each separately as repeat does not like more dimensions
	for channel in use_channels:
		print('Processing channel ',channel)
		outfilename = filename.rsplit('.', 1)[0]+str(channel)+'.wav'
		with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
			if resampling_mode in ("Sinc",):
				for block in sinc_interp_windowed(signal[:,channel], samples_in2, samples_out2, write_after=write_after, NT=NT):
					if prog_sig:
						progress += prog_fac
						prog_sig.notifyProgress.emit(progress)
					print("progress",progress)
					print("len(block)",len(block))
					outfile.write( block )
			elif resampling_mode in ("Linear",):
				block = np.interp(samples_out2, samples_in2, signal[:,channel])
				outfile.write( block )
				#we don't split here (yet)
				if prog_sig:
					progress += (100 / len(use_channels))
					prog_sig.notifyProgress.emit(progress)
			else:
				for block_start, block_end in blocks:
					if prog_sig:
						progress += prog_fac
						prog_sig.notifyProgress.emit(progress)
					#print("Writing",block_start,"to",block_end)
					
					#create a mono array for the current block and channel
					signal_block = signal[block_start:block_end, channel]
					speed_block = speed_samples_final[block_start:block_end]
					
					#apply the different methods
					if resampling_mode == "Expansion":
						
						#repeat each sample by the resampling_factor * the speed factor for that sample
						upsampled = np.repeat(signal_block, speed_block)
						#now divide it by the resampling_factor to return to the original median speed
						#this is fast and does not cut off high freqs
						resampled = resampy.resample(upsampled, resampling_factor, 1, filter='sinc_window', num_zeros=4)
						
					elif resampling_mode == "Blocks":
						#take a block of the signal
						#tradeoff between frequency and time precision -> good for wow, unusable for flutter
						upsampled = signal_block
						
						#here we do not use the specified resampling factor but instead make our own from the average speed of the current block
						resampling_factor = 1/ np.mean(speed_block)
						#this is fast and does not cut off high freqs
						resampled = resampy.resample(upsampled, resampling_factor, 1, filter='sinc_window', num_zeros=4)
					
					elif resampling_mode in ("Sinc", "Windowed Sinc", "Linear"):
						resampled = formula_interpolation(signal_block, speed_block, mode=resampling_mode)
						
					if patch_ends:
						resampled[0] = signal_block[0]
						resampled[-1] = signal_block[-1]
					#fast attenuation of clicks, not perfect but better than nothing
					
					#resample write the output block to the audio file
					outfile.write(resampled)
	if prog_sig:
		prog_sig.notifyProgress.emit(100)
	print("Done!\n")	
	#TODO:
	#cover up block cuts, with overlap
	#for blocks: adaptive resolution according to speed derivation. more derivation = shorter blocks

#Expansion:
#the more frequency precision (smaller percentage), the longer it takes
#temporal precision is -theoretically- unaffected
#but practically, input blocks become shorter for more precision, as output size should remain stable to avoid overflow

# #Blocks
# #classic tradeoff: the more frequency precision, the faster
# #at the cost of less temporal precision

	