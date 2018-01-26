import numpy as np
import resampy
import soundfile as sf
import os



# def sinc_interp_windowed_functional_non_generator(y_in, x_in, x_out, ):
	# """
	# base by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
	# Interpolate the signal to the new points using a sinc kernel
	# input:
	# x_in		time points x is defined on
	# y_in		 input signal column vector or matrix, with a signal in each row
	# x_out		points to evaluate the new signal on
	# output:
	# y		 the interpolated signal at points x_out
	# """

	# y = np.zeros(len(x_out))
	
	# #has to be fairly high
	# NT = 300
	# for pi, p in enumerate (x_out):
		# #map the output to the input
		# ind = int(round(p))
		# #print(ind)
		# #ok, but we still have a click
		# lower = max(0, ind-NT)
		# upper = ind+NT
		# #this gives ugly breaks
		# #lower = ind-NT
		# #upper = ind+NT
		# si = np.sinc (x_in[lower:upper] - p)
		# y[pi] = np.sum(si * y_in[lower:upper])
	# return y

def sinc_interp_windowed(y_in, x_in, x_out, write_after=10000, NT = 100):
	"""
	base by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
	Interpolate the signal to the new points using a sinc kernel
	input:
	x_in			time points x is defined on
	y_in		 	input signal column vector or matrix, with a signal in each row
	x_out			points to evaluate the new signal on
	write_after		yield a block after X output samples
	NT				samples to take before and after the central sample
	output:
	y				 the interpolated signal at points x_out
	"""
	y = np.zeros(write_after+1)
	offset = 0
	
	#has to be fairly high
	#NT = 100
	# for i, p in enumerate (x_out):
		# #map the output to the input
		# ind = int(round(p))
		# lower = max(0, ind-NT)
		# upper = ind+NT
		
		# si = np.sinc (x_in[lower:upper] - p)
		# y[i- offset*(write_after+1)] = np.sum(si * y_in[lower:upper])
		# if i - offset*write_after  == write_after:
			# offset += 1
			# yield y
	
	y = np.zeros(write_after)
	while offset < len(x_out):
		print("piece at", offset)
		outind = 0
		for i in range(offset, offset+write_after):
			#now we are at the end and we can yield the rest of the piece
			if i == len(x_out)-1:
				print("end",i,outind)
				print(len(y[0:outind]))
				yield y[0:outind]
				offset = len(x_out)
				break
			p = x_out[i]
			#map the output to the input
			ind = int(round(p))
			lower = max(0, ind-NT)
			upper = ind+NT
			
			si = np.sinc (x_in[lower:upper] - p)
			y[outind] = np.sum(si * y_in[lower:upper])
		
			outind+=1
		#we may have to exit here to avoid sending another block
		if offset == len(x_out):
			break
		offset+=write_after
		
		yield y

# def sinc_interp_windowed_gen(y_in, x_in, x_out ):
	# #https://gist.github.com/nadavb/13030ca99336d4a1eb6390ad03d23c00
	# NT = 50	 # Change this number to change the accuracy
	# write_after = 10000
	# result = np.ones(write_after+1)
	# T = x_in[1] - x_in[0]  # Find the period	   
	# fac = T * len(y_in) / len(x_out)
	# offset = 0
	# for i in range(0, len(x_out)):
		# t = i * fac
		# n = [n for n in range(max(int(-NT + t/T), 0), min(int(NT + t/T), len(y_in)))]
		# result[i- offset*write_after] = np.sum( y_in[n] * np.sinc((t - n*T)/T) )
		# if i - offset*write_after == write_after:
			# offset += 1
			# yield result
			# result = np.ones(write_after+1)
	
# def sinc_interp(y_in, x_in, x_out):
	# """
	# base by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
	# Interpolate the signal to the new points using a sinc kernel

	# input:
	# y_in	   input signal vector
	# x_in	  time points y_in is defined on
	# x_out	   points to evaluate the new signal on

	# output:
	# y		the interpolated signal at points x_out
	# """
	# y = np.zeros(len(x_out))
	# for pi, p in enumerate(x_out):
		# y[pi] = np.sum(y_in * np.sinc(x_in - p))
	# return y
  # #https://stackoverflow.com/questions/25181642/how-set-numpy-floating-point-accuracy
# def formula_interpolation(samples, speed, mode="Sinc"):
	# #signal and speed arrays must be mono and have the same length
	# #IDK where this has to be inverted otherwise
	# speed = 1/speed
	# in_length = len(speed)

	# #create a new speed array that is as long as the output is going to be. the magic is in the output x's step length
	# interp_speed = np.interp(np.arange(0, in_length, np.mean(speed)), np.arange(0, in_length), speed)

	# #the times of the input samples, these must be evenly sampled, starting at 1
	# in_positions = np.arange(1, in_length+1)
	# #these are unevenly sampled, but when played back at the correct (original) sampling rate, yield the respeeded result
	# out_positions = np.cumsum(interp_speed )
	# if mode == "Sinc": return sinc_interp(samples, in_positions, out_positions)
	# elif mode == "Windowed Sinc": return sinc_interp_windowed(samples, in_positions, out_positions)
	# elif mode == "Linear": return np.interp(out_positions, in_positions, samples)


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
		NT=100
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

	