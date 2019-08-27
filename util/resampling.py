import numpy as np
from time import time
import soundfile as sf
import os
from numba import jit#, prange, guvectorize
import math
import threading

def sinc_wrapper(sample_at, signal, lowpass, NT ):
	# initialize arrays here because we can't do so in nopython mode
	N = np.arange(-NT,NT+1, dtype="float32")
	win_func = np.hanning(2*NT+1).astype("float32")
	output = np.empty(len(sample_at), "float32")
	sinc_core(sample_at, signal, lowpass, output, win_func, N )
	return output
	
def sinc_wrapper_mt(sample_at, signal, lowpass, NT ):
	# manual parallelization with threading module
	# from: http://numba.pydata.org/numba-doc/dev/user/examples.html#multi-threading
	numthreads = os.cpu_count()
	length = len(sample_at)
	N = np.arange(-NT,NT+1, dtype="float32")
	win_func = np.hanning(2*NT+1).astype("float32")
	output = np.empty(length, dtype="float32")
	chunklen = (length + numthreads - 1) // numthreads
	# Create argument tuples for each input chunk
	chunks = [ (sample_at[i * chunklen:(i + 1) * chunklen], signal, lowpass, output[i * chunklen:(i + 1) * chunklen], win_func, N) for i in range(numthreads)]
	# Spawn one thread per chunk
	threads = [threading.Thread(target=sinc_core, args=chunk) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	return output
	
# lazy optimization is recommended
# @jit('float32[:](float64[:], float64[:], float64[:], float32[:], float64[:], int32[:])', nopython=True, nogil=True, parallel=True)
@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def sinc_core(sample_at, signal, lowpass, output, win_func, N ):
	"""
	sample_at: 1D array holding the points at which the old signal should be sampled to get the new signal, is expected to be 0 <= i < len(signal)
	signal: 1D array of the input signal
	lowpass: 1D array of lowpass filter coefficients
	output: 1D array, pre-allocated, len(output) == len(sample_at)
	win_func: windowing used for the sinc to reduce ringing, np.hanning(2*sinc_quality)
	N: fixed range of input for sinc
	"""
	NT = (len(N)-1)//2
	len_in = len(signal)
	len_out = len(sample_at)
	#just using prange here does not make it faster
	for i in range( len_out ):
		# p is the position in the signal where this output sample should be sampled at
		p = sample_at[i]
		# rounding here makes no significant difference; truncating may be enough
		ind = int(round(p))
		
		lower = max(0, ind-NT)
		upper = min(ind+NT, len_in)
		
		#fc is the cutoff frequency expressed as a fraction of the nyquist freq
		#we need anti-aliasing when sampling rate is bigger than before, ie. exceeding the nyquist frequency
		if i+1 != len_out:
			period_to = max(0.000000000001, sample_at[i+1]-p)
			#period to must not be 0, crashes if jit is enabled
		fc = min(1/period_to, 1)
		# fc = lowpass[i]
		# old and new are not identical because their interpolation is different. old is better
		
		# evaluate the sinc function around its center (with slight shift)
		# stretched & scaled by fc parameter as a lowpass / anti-aliasing filter
		shift = p - ind
		si = np.sinc ((N - shift) * fc) * fc
		
		# create output sample by taking the sum of its neighboring input samples weighted by the sinc and window
		sigbit = signal[lower:upper]
		output[i] = np.sum(sigbit * si[0:len(sigbit)] * win_func[0:len(sigbit)])

@jit(forceobj=True)
def speed_to_pos(sampletimes, speeds, num_imput_samples):
	"""
	sampletimes: 1D array of sample numbers at which speeds is sampled; must have even spacing
	speeds: 1D array of speed samples
	num_imput_samples: int
	"""
	
	#TODO: this could actually make use of sampletimes by locking start and end values of each block to these positions
	# but, especially for small dt, that could introduce unwanted speed errors
	dt = sampletimes[1]-sampletimes[0]
	err = 0
	offset = sampletimes[0]
	
	#just add a little bit of tolerance here
	end_guess = int(np.mean(speeds)*(sampletimes[-1]-sampletimes[0]) *1.01 )
	output = np.empty(end_guess, dtype="float32")
	out_ind = 0
	for i in range(0, len(speeds)-1):
		#the desired number of output samples in this block, before dithering
		n = dt * np.mean(speeds[i:i+2])
		
		#without dithering here, we get big gaps at the end of each segment!
		inerr = n + err
		n = int(round(inerr))
		err =  inerr-n
		#this is fast and ready for jit, but not so nicely readable
		block_speeds = np.arange(n)/(n-1)*(speeds[i+1]-speeds[i])+speeds[i]
		#linspace is slower for numpy than interp, but interp is not supported by jit
		# block_speeds = np.interp(np.arange(n), (0, n-1), (speeds[i], speeds[i+1]) )
		# block_speeds = np.linspace(speeds[i], speeds[i+1], n)
		sample_at = np.cumsum(1/block_speeds) + offset
		offset = sample_at[-1]
		output[out_ind:out_ind+n] = sample_at
		# print(sample_at[0],sampletimes[i],sample_at[-1],sampletimes[i+1])
		
		# see if this block reaches the end of the input signal
		if output[out_ind] <= num_imput_samples <= output[out_ind+n-1]:
			# ok the end is somewhere in this block
			# we can fine tune it to find the exact output sample 
			end = out_ind + np.argmin(np.abs(sample_at-num_imput_samples))
			# now trim to get rid of the extra tolerance
			output = output[:end]
			break
		out_ind+=n
	return output


def lag_to_pos(sampletimes, lags, num_imput_samples):
	"""
	sampletimes: 1D array of sample numbers at which lags is sampled
	lags: 1D array of lag samples, ie. shift from actual sample time
	num_imput_samples: int
	"""
	# dt = sampletimes[1]-sampletimes[0]
	end_guess = int((num_imput_samples+lags[-1]) *1.01 )
	output = np.empty(end_guess, dtype="float32")
	speeds_in = np.empty(len(lags), dtype="float32")
	speeds = np.empty(len(lags), dtype="float32")
	for i in range(0, len(lags)-1):
		original = sampletimes[i+1]-sampletimes[i]
		target = original- (lags[i+1]-lags[i])
		# this is the mean speed for this segment
		speeds_in[i] = target/original
	# now split the segment and do a cumsum in each part
	for i in range(0, len(lags)-1):
		speeds[i] = (speeds[i]+speeds[i+1])/2
	print(speeds[i])
		

def run(filenames, signal_data=None, speed_curve=None, resampling_mode = "Linear", sinc_quality=50, use_channels = [0,], prog_sig=None, lag_curve=None):
	if prog_sig: prog_sig.notifyProgress.emit(0)
	if signal_data is None: signal_data = [None for filename in filenames]
	for filename, sig_data in zip(filenames, signal_data):
		start_time = time()
		print('Resampling ' + filename + '...', resampling_mode, sinc_quality, use_channels)
		#read the file
		if sig_data:
			signal, sr = sig_data
		else:
			from util import io_ops
			signal, sr, channels = io_ops.read_file(filename)
		if resampling_mode == "Linear":
			samples_in = np.arange( len(signal) )
		lowpass = 0
		if speed_curve is not None:
			sampletimes = speed_curve[:,0]*sr
			speeds = speed_curve[:,1]
			sample_at = speed_to_pos(sampletimes, speeds, len(signal))
			# the problem is we don't really need the lerped speeds but what happens from the cumsum
			# get the speed for every output sample
			# if resampling_mode == "Sinc":
				# lowpass = np.interp(np.arange( len(sample_at) ), sampletimes, speeds)
		elif lag_curve is not None:
			sampletimes = lag_curve[:,0]*sr
			lags = lag_curve[:,1]*sr
			# lag_to_pos(sampletimes, lags, len(signal))
			sample_at = np.interp(np.arange( len(signal)+lags[-1] ), sampletimes, sampletimes-lags)
			# ensure we have no sub-zero values, saves one max in sinc
			np.clip(sample_at, 0, None, out=sample_at)
			# with lerped speed curve
			# speeds = np.diff(lag_curve[:,1])/np.diff(lag_curve[:,0])+1
			# sampletimes = (lag_curve[:-1,0]+np.diff(lag_curve[:,0])/2)*sr
			# sample_at = speed_to_pos(sampletimes, speeds)
		dur = time() - start_time
		print("Preparation took",dur)
		start_time = time()
		#resample mono channels and export each separately
		for progress, channel in enumerate(use_channels):
			print('Processing channel ',channel)
			outfilename = filename.rsplit('.', 1)[0]+str(channel)+'.wav'
			with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
				if resampling_mode == "Sinc":
					outfile.write( sinc_wrapper_mt(sample_at, signal[:,channel], lowpass, sinc_quality ) )
				elif resampling_mode == "Linear":
					outfile.write( np.interp(sample_at, samples_in, signal[:,channel]) )
			if prog_sig: prog_sig.notifyProgress.emit((progress+1)/len(use_channels)*100)
		if prog_sig: prog_sig.notifyProgress.emit(100)
		dur = time() - start_time
		print("Resampling took",dur)
	print("Done!\n")
	
def timefunc(correct, s, func, *args, **kwargs):
	"""
	Benchmark *func* and print out its runtime.
	"""
	from timeit import repeat
	print(s.ljust(20), end=" ")
	# Make sure the function is compiled before we start the benchmark
	res = func(*args, **kwargs)
	if correct is not None:
		assert np.allclose(res, correct), (res, correct)
	# time it
	print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
										  number=5, repeat=2)) * 1000))
	return res
	
def test_sinc():
	# just a test function
	sr = 44100
	volume = 0.5	 # range [0.0, 1.0]
	duration = 2.0	 # seconds
	f = 440.0		 # sine frequency, Hz
	
	# generate signal
	signal = np.sin(2*np.pi*np.arange(sr*duration)*f/sr, dtype="float32")*volume
	signal += np.sin(2*np.pi*np.arange(sr*duration)*21000/sr)*.1
	# generate speed curve
	sampletimes = (0,len(signal))
	speeds = (0.5, 2)
	sample_at = speed_to_pos(sampletimes, speeds, len(signal))
	lowpass = np.interp(np.arange( len(sample_at) ), sampletimes, speeds)
	np.clip(lowpass, None, 1, out=lowpass)
	
	correct = timefunc(None, "normal", sinc_wrapper, sample_at, signal, lowpass, 50)
	timefunc(correct, "mt", sinc_wrapper_mt, sample_at, signal, lowpass, 50)
	with sf.SoundFile("test_raw.wav", 'w+', sr, 1, subtype='FLOAT') as outfile:
		outfile.write( signal )
	with sf.SoundFile("test_sin.wav", 'w+', sr, 1, subtype='FLOAT') as outfile:
		outfile.write( sinc_wrapper_mt(sample_at, signal, lowpass, 50 ) )


if __name__ == '__main__':
	test_sinc()