import numpy as np
import resampy
import soundfile as sf
import os

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
		write_after=400000
		NT = 50
		
		in_len = len(signal[:,0])
		samples_in2 = np.arange(0, in_len)
			
		offsets_speeds = []
		offset = 0
		err = 0
		temp_offset = 0
		temp_pos = []
		for i in range(0, len(speeds)-1):
			#save a new block for interpolation
			period = (times[i+1]-times[i]) * sr
			if len(temp_pos)* period > write_after:
				offsets_speeds.append( ( offset, np.concatenate(temp_pos) ) )
				offset += temp_offset
				temp_offset = 0
				temp_pos = []
			#we want to know how many samples there are in this section, so get the period (should be the same for all sections)
			mean_speed = ( (speeds[i]+speeds[i+1])/ 2 )
			#the desired number of samples in this block - this is clearly correct
			n = period*mean_speed
			
			inerr = n + err
			n = round(inerr)
			err =  inerr-n
			
			linsp = np.linspace(0, 1, n)
			block_speeds = np.interp(linsp, (0, 1),(speeds[i], speeds[i+1])  )
			positions = np.cumsum(1/block_speeds)
			
			# this is more accurate for long pieces, but a bit slower and gives clicks for short pieces
			# samples_in = np.arange(0, period)
			# block_speeds = np.interp(samples_in, (0, period),(speeds[i], speeds[i+1])  )
			# positions = np.interp(np.arange(0, n), np.cumsum(block_speeds), samples_in)
			
			temp_pos.append( positions +  temp_offset)
			temp_offset+=positions[-1]
			#temp_offset+=period
		# the end
		if temp_pos: offsets_speeds.append( (offset, np.concatenate(temp_pos) ) )
		num_blocks = len(offsets_speeds)
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
				speed_samples_final = np.rint(np.random.rand(len(speed_samples_r))-.5 + speed_samples_r).astype(np.int64)
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
				win_func = np.hanning(2*NT)
				for offset, positions in offsets_speeds:
					block_len = int(len(positions))
					block = np.zeros( block_len)
					for i in range( block_len):
						#now we are at the end and we can yield the rest of the piece
						p = positions[i]
						#map the output to the input
						#error diffusion here makes no significant difference
						ind = int(round(p))+int(offset)
						#print(ind)
						lower = max(0, ind-NT)
						upper = min(ind+NT, in_len)
						length = upper - lower
						
						#fc is the cutoff frequency expressed as a fraction of the nyquist freq
						#we need anti-aliasing when sampling rate is bigger than before, ie. exceeding the nyquist frequency
						#could skip this calculation and get it from the speed curve instead?
						if i+1 != block_len:
							period_to = positions[i+1]-p
						fc = min(1/period_to, 1)
						
						#use a Hann window to reduce the prominent sinc ringing of a rectangular window
						#(http://www-cs.engr.ccny.cuny.edu/~wolberg/pub/crc04.pdf, p. 11ff)
						#http://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
						#claims that only hamming and blackman are worth using? my experiments look best with hann window
						
						si = np.sinc ((samples_in2[lower:upper]-int(offset) - p) * fc) * fc
						block[i] = np.sum(si * signal[lower:upper,channel] * win_func[0:length])
					
					outfile.write( block )
					if prog_sig:
						progress += prog_fac
						prog_sig.notifyProgress.emit(progress)
					print("progress",progress)
					print("len(block)",len(block))
			elif resampling_mode in ("Linear",):
				for offset, positions in offsets_speeds:
					block = np.interp(positions, samples_in2-int(offset), signal[:,channel])
					outfile.write( block )
					print("progress",progress)
					print("len(block)",len(block))
					if prog_sig:
						progress += prog_fac
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
						#now divide it by the resampling_factor to return to the original median speed
						#this is fast and does not cut off high freqs
						resampled = resampy.resample(np.repeat(signal_block, speed_block), resampling_factor, 1, filter='sinc_window', num_zeros=4)
						
					elif resampling_mode == "Blocks":
						#take a block of the signal
						#tradeoff between frequency and time precision -> good for wow, unusable for flutter
						#here we do not use the specified resampling factor but instead make our own from the average speed of the current block
						#this is fast and does not cut off high freqs
						resampled = resampy.resample(signal_block, 1/ np.mean(speed_block), 1, filter='sinc_window', num_zeros=4)
					
					#resample write the output block to the audio file
					outfile.write(resampled)
	if prog_sig:
		prog_sig.notifyProgress.emit(100)
	print("Done!\n")