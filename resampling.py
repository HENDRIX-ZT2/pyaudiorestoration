import numpy as np
from time import time
import soundfile as sf
import os
from numba import jit#, prange

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
	if data:
		print("Saved",len(data),"traces")
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
	if data:
		print("Writing",len(data),"regressions")
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

def write_lag(filename, data):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	data:	 a list of sine parameters
	"""
	
	#write the data to the speed file
	if data:
		print("Writing",len(data),"lag")
		speedfilename = filename.rsplit('.', 1)[0]+".syn"
		outstr = "\n".join([" ".join([str(v) for v in values]) for values in data])
		text_file = open(speedfilename, "w")
		text_file.write(outstr)
		text_file.close()
	
def read_lag(filename):
	"""
	TODO: rewrite into BIN format
	filename: the name of the original audio file
	returns:
	data:	 a list of sine parameters
	"""
	
	#write the data to the speed file
	print("Reading lag data")
	speedfilename = filename.rsplit('.', 1)[0]+".syn"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for l in text_file:
				#just for completeness
				if l:
					data.append( [float(v) for v in l.split(" ")])
	return data
	
 # @guvectorize(['(float64[:,:],float64[:,:],float64[:,:])'],
    # '(n,m),(n,m)->(n,m)',target='parallel')
	
#jit notes: parallel makes it at least as slow as the normal one
# lazy optimization is recommended
@jit(nopython=True, nogil=True, cache=True)
def sinc_core(positions, samples_in2, signal, output, win_func, NT ):
	in_len = len(samples_in2)
	block_len = len(positions)
	#just using prange here does not make it faster
	for i in range( block_len):
		#now we are at the end and we can yield the rest of the piece
		p = positions[i]
		#map the output to the input
		#error diffusion here makes no significant difference
		ind = int(round(p))
		
		#can we end it here already?
		#then cut the output and break out of this loop
		if ind == in_len:
			return output[0:i]
			
		lower = max(0, ind-NT)
		upper = min(ind+NT, in_len)
		# length = upper - lower
		
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
		
		si = np.sinc ((samples_in2[lower:upper] - p) * fc) * fc
		#note that in some end cases, len may be 0 but win_func gets another len so it would break
		#TODO: this is probably obsolete and could get its indices like the other two, because the end is cut off already
		output[i] = np.sum(si * signal[lower:upper] * win_func[0:len(si)])
	return output

def update_progress(prog_sig, progress, prog_fac):
	if prog_sig:
		progress += prog_fac
		prog_sig.notifyProgress.emit(progress)
	return progress
	
@jit(nopython=False, nogil=True, cache=True)
def prepare_linear_or_sinc(sampletimes, speeds):
	# replace periods with fixed dt, but how to deal with the end?
	# dt = sampletimes[1]-sampletimes[0]
	periods = np.diff(sampletimes)
	err = 0
	offset = sampletimes[0]
	
	#just add a little bit of tolerance here
	end_guess = int(np.mean(speeds)*(sampletimes[-1]-sampletimes[0]) *1.01 )
	output = np.empty(end_guess)
	out_ind = 0
	for i in range(0, len(speeds)-1):
		#the desired number of output samples in this block, before dithering
		n = periods[i] * np.mean(speeds[i:i+2])
		
		#without dithering here, we get big gaps at the end of each segment!
		inerr = n + err
		n = int(round(inerr))
		err =  inerr-n
		#this is fast and ready for jit, but not so nicely readable
		block_speeds = np.arange(n)/(n-1)*(speeds[i+1]-speeds[i])+speeds[i]
		#linspace is slower for numpy than interp, but interp is not supported by jit
		# block_speeds = np.interp(np.arange(n), (0, n-1), (speeds[i], speeds[i+1]) )
		# block_speeds = np.linspace(speeds[i], speeds[i+1], n)
		#is this correct, should not rather be + old_positions[-1]  to account for dithering?
		positions = np.cumsum(1/block_speeds) + offset
		offset = positions[-1]
		output[out_ind:out_ind+n] = positions
		out_ind+=n
	#trim to remove the extra tolerance
	return output[:out_ind]

def run(filename, speed_curve=None, resampling_mode = "Linear", sinc_quality=50, use_channels = [0,], prog_sig=None, lag_curve=None):
	print('Resampling ' + filename + '...', resampling_mode, sinc_quality, use_channels)
	if prog_sig: prog_sig.notifyProgress.emit(0)
	start_time = time()
		
	#read the file
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True, dtype='float32')
	sr = soundob.samplerate
	
	samples_in = np.arange( len(signal[:,0]) )
	if speed_curve is not None:
		sampletimes = speed_curve[:,0]*sr
		speeds = speed_curve[:,1]
		samples_out = prepare_linear_or_sinc(sampletimes, speeds)
	elif lag_curve is not None:
		sampletimes = lag_curve[:,0]*sr
		lags = lag_curve[:,1]*sr
		samples_out = np.interp(samples_in, sampletimes, sampletimes-lags)
		
	dur = time() - start_time
	print("Preparation took",dur)
	start_time = time()
	#resample mono channels and export each separately
	for progress, channel in enumerate(use_channels):
		print('Processing channel ',channel)
		outfilename = filename.rsplit('.', 1)[0]+str(channel)+'.wav'
		with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
			if resampling_mode == "Sinc":
				outfile.write( sinc_core(samples_out, samples_in, signal[:,channel], np.empty(len(samples_out), "float32"), np.hanning(2*sinc_quality), sinc_quality ) )
			elif resampling_mode == "Linear":
				outfile.write( np.interp(samples_out, samples_in, signal[:,channel]) )
		if prog_sig: prog_sig.notifyProgress.emit(progress/len(use_channels)*100)
	if prog_sig: prog_sig.notifyProgress.emit(100)
	dur = time() - start_time
	print("Resampling took",dur)
	print("Done!\n")