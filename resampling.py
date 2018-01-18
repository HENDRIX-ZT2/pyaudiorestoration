import numpy as np
import resampy
import soundfile as sf
import os

def sinc_interp_windowed(y_in, x_in, x_out, ):
	"""
	base by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
	Interpolate the signal to the new points using a sinc kernel

	input:
	x_in		time points x is defined on
	y_in		 input signal column vector or matrix, with a signal in each row
	x_out		points to evaluate the new signal on

	output:
	y		 the interpolated signal at points x_out
	"""

	mn = y_in.shape
	if len(mn) == 2:
		m = mn[0]
		n = mn[1]
	elif len(mn) == 1:
		m = 1
		n = mn[0]
	else:
		raise ValueError ("y_in is greater than 2D")

	nn = len(x_out)

	y = np.zeros((m, nn))
	
	#has to be fairly high
	a = int(max(len(x_in), len(x_out))/10)

	for pi, p in enumerate (x_out):
		#map the output to the input
		pi_in = int(round(pi / len(x_out) * len(x_in)))
		
		lower = max(0, pi_in-a)
		upper = pi_in+a
		si = np.tile(np.sinc (x_in[lower:upper] - p), (m, 1))
		y[:, pi] = np.sum(si * y_in[lower:upper])
	return y.squeeze ()
	
def sinc_interp(y_in, x_in, x_out):
  """
  by Gaute Hope: https://gist.github.com/gauteh/8dea955ddb1ed009b48e
  Interpolate the signal to the new points using a sinc kernel

  input:
  y_in     input signal column vector or matrix, with a signal in each row
  x_in    time points y_in is defined on
  x_out    points to evaluate the new signal on

  output:
  y     the interpolated signal at points x_out
  """

  mn = y_in.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError("y_in is greater than 2D")

  nn = len(x_out)

  y = np.zeros((m, nn))

  for pi, p in enumerate(x_out):
    si = np.tile(np.sinc(x_in - p), (m, 1))
    y[:, pi] = np.sum(si * y_in)

  return y.squeeze()
  
def sinc_interp2(y_in, x_in, x_out):
	"""
	by endolith: https://gist.github.com/endolith/1297227#file-sinc_interp-py
	Interpolates y_in, sampled at "x_in" instants
	Output y is sampled at "x_out" instants
	
	from Matlab:
	http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html		   
	"""
	if len(y_in) != len(x_in):
		raise ValueError('y_in and x_in must be the same length')
	
	# Find the period	 
	T = x_in[1] - x_in[0]
	
	sincM = np.tile(x_out, (len(x_in), 1)) - np.tile(x_in[:, np.newaxis], (1, len(x_out)))
	y = np.dot(y_in, np.sinc(sincM/T))
	return y

def formula_interpolation(samples, speed, mode="Sinc"):
	#signal and speed arrays must be mono and have the same length
	#IDK where this has to be inverted otherwise
	speed = 1/speed
	in_length = len(speed)

	#create a new speed array that is as long as the output is going to be. the magic is in the output x's step length
	interp_speed = np.interp(np.arange(0, in_length, np.mean(speed)), np.arange(0, in_length), speed)

	#the times of the input samples these must be evenly sampled, starting at 1
	in_positions = np.arange(1, in_length+1)
	#these are unevenly sampled, but when played back at the correct (original) sampling rate, yield the respeeded result
	out_positions = np.cumsum(interp_speed )
	if mode == "Sinc": return sinc_interp(samples, in_positions, out_positions)
	elif mode == "Windowed Sinc": return sinc_interp(samples, in_positions, out_positions)
	elif mode == "Linear": return np.interp(out_positions, in_positions, samples)

def flatten_trace(data):
	"""
	change multi-trace data into one flat continous trace
	may have jumps
	TODO: correct jumps by assimilating the median of all parts on log2 scale, then rescale to Hz
	"""
	alltimes = []
	allfreqs = []
	
	for t, f in data:
		alltimes+=t
		allfreqs+=f
	times, freqs = zip(*sorted(zip(alltimes, allfreqs)))
	return np.asarray(times), np.asarray(freqs)
	
def trace_to_speed(data, length=None, steps=None, target_freq=None):
	"""
	input: multi-channel, possibly overlapping, possibly with gaps
	output: 1D speed curve (+ time stamps, but unchanged)
	"""
	
	# num = self.vispy_canvas.num_ffts
	# length = num * self.vispy_canvas.hop / self.vispy_canvas.sr
	# #get the times at which the average should be sampled
	# times = np.linspace(0, length, num=steps)
	# #create the array for sampling
	# out = np.ones((len(times), len(self.vispy_canvas.lines)), dtype=np.float32)
	# #lerp and sample all lines, use NAN for missing parts
	# for i, line in enumerate(self.vispy_canvas.lines):
		# line_sampled = np.interp(times, line.times, line.speed, left = np.nan, right = np.nan)
		# out[:, i] = line_sampled
	# #take the mean and ignore nans
	# mean_with_nans = np.nanmean(out, axis=1)
	# mean_with_nans[np.isnan(mean_with_nans)]=1
	# #set the output data
	# self.master_speed = np.ones((len(times), 2), dtype=np.float32)
	# self.master_speed[:, 0] = times
	# self.master_speed[:, 1] = mean_with_nans
	# self.line_speed.set_data(pos=self.master_speed)
	#idea:
	#-generate continous time sample (arange(0,end)
	#find overlapping regions
	#-interp every freq curve
	# extents = []
	# #the instances on which our curve is sampled
	# #even though the segments may overlap, the points probably do not match up
	# #hence, interpolate
	# alltimes = []
	# for times, frequencies in data:
		# extents.append( (times[0], times[-1]) )
		# alltimes.extend(times)
	# alltimes = sorted(alltimes)
	# for i in range(0, extents):
		# for j in range(0, extents):
			# #just so we are not comparing the same data set with each other
			# if i != i:
				# #t0
				# #i starts in the middle of j
				# extents[j][0] < extents[i][0] < extents[j][1]
				# #i ends in the middle of j
				# extents[j][0] < extents[i][1] < extents[j][1]
				
	times = np.asarray(data[0][0])
	freqs = np.asarray(data[0][1])
	#this is the frequency all measured frequencies should have in the end
	#if not supplied, use mean or median of freqs
	if not target_freq:
		target_freq	= np.median(freqs)

	#divide them to get the speed ratio for sample expansion
	speeds = freqs/target_freq
	
	return times, speeds

def write_trace(filename, data):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	data:    a list of (times, frequencies) lists
	"""
	
	#write the data to the speed file
	print("Writing speed data")
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
	data:    a list of (times, frequencies) lists
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
	
def run(filename, speed_curve=None, resampling_mode = "Blocks", frequency_prec=0.01, target_freq = None, use_channels = [0,], dither="Random", patch_ends=False, prog_sig=None):

	#read the file
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True)
	channels = soundob.channels
	sr = soundob.samplerate
	
	print('Analyzing ' + filename + '...')
	print('Shape:', signal.shape)
	
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
	
	overlap = 0
	blocks = []
	for n in range(0, len(signal), block_size):
		blocks.append((n, block_size+n+overlap))
	print("Num Blocks",len(blocks))
	
	if prog_sig:
		prog_sig.notifyProgress.emit(0)
		prog_fac = 100/ (len(blocks)*len(use_channels))
	
	if speed_curve is not None:
		times = speed_curve[:,0]
		speeds = speed_curve[:,1]
	else:
		#these are measured frequencies in Hz and their times (converted to samples on the fly)
		speedfilename = filename.rsplit('.', 1)[0]+".speed"
		if not os.path.isfile(speedfilename):
			print("Speed file does not exist!")
			return
		#read the trace curves
		data = read_trace(filename)
		if not len(data):
			print("No speed data in file!")
			return
		#convert the fragmentary frequency trace to continous speed curve
		times, speeds = trace_to_speed(data, target_freq=target_freq)
	
	#convert times to samples
	times *= sr
	
	#lerp the samples to the length of the signal
	#!this does not handle extrapolation!
	speed_samples_r = np.interp(np.arange(0, len(signal)), times, speeds, left=speeds[0], right=speeds[-1])
	
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
			
	
	progress = 0
	#resample on mono channels and export each separately as repeat does not like more dimensions
	for channel in use_channels:
		print('Processing channel ',channel)
		outfilename = filename.rsplit('.', 1)[0]+str(channel)+'.wav'
		with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
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
	return
#Expansion:
#the more frequency precision (smaller percentage), the longer it takes
#temporal precision is -theoretically- unaffected
#but practically, input blocks become shorter for more precision, as output size should remain stable to avoid overflow

# #Blocks
# #classic tradeoff: the more frequency precision, the faster
# #at the cost of less temporal precision

	