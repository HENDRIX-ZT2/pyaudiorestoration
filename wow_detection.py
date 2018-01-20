import numpy as np
import fourier
	
def COG(magnitudes, freqs, NL, NU):
	#adapted from Czyzewski et al. (2007)
	#18
	#calculate the COG with hanned frequency importance
	#error in the printed formula: the divisor also has to contain the hann window
	weighted_magnitudes = np.hanning(NU-NL) * magnitudes[NL:NU]
	return np.sum(weighted_magnitudes * np.log2(freqs[NL:NU])) / np.sum(weighted_magnitudes)

def trace_cog(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None):
	#adapted from Czyzewski et al. (2007)
	#note that the input D must NOT be in dB scale, just abs(FFT) and nothing more
	
	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	#bin indices of the starting band
	N = 1 + fft_size
	#clamp to valid frequency range
	fL = max(1.0, fL)
	fU = min(sr/2, fU)
	#make sure it doesn't escape the frequency limits
	NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
	
	if first_fft_i == last_fft_i:
		print("No point in tracing just one FFT")
		return [],[]
	if NL == NU:
		print("Can not trace one bin only")
		return [],[]
		
	#the frequencies of the bins
	freqs = fourier.fft_freqs(fft_size, sr)

	LSCoct = []
	#LfL = []
	#LfU = []
	times = []
	
	#calculate the first COG
	SCoct0  = COG(D[:, first_fft_i], freqs, NL, NU)
	#16a,b
	#the limits for the first time frame
	#constant for all time frames
	#SCoct[0]: the the first COG
	dfL = SCoct0 - np.log2(fL)
	dfU = np.log2(fU) - SCoct0
	#print(dfL,dfU)
	
	for i in range(first_fft_i, last_fft_i):
		#18
		#calculate the COG with hanned frequency importance
		#error in the printed formula: the divisor also has to contain the hann window
		SCoct = COG(D[:, i], freqs, NL, NU)
		
		#save the data of this frame
		t = i*hop/sr
		times.append(t)
		
		#19
		#Hz = 2^COG
		LSCoct.append(2**SCoct)
		#LfL.append(fL)
		#LfU.append(fU)
		
		#15a,b
		#set the limits for the consecutive frame [i+1]
		#based on those of the first frame
		fL = 2**(SCoct-dfL)
		fU = 2**(SCoct+dfU)
		#bin 0 must not be used in the trace, it gives NaN error
		NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
		NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
		#NL = int(round(fL * N / sr))
		#NU = int(round(fU * N / sr))
	
	return times, LSCoct#, LfL, LfU
	
	
def parabolic(f, x):
	"""Helper function to refine a peak position in an array"""
	xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
	yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
	return (xv, yv)
	

def trace_correlation(D, fft_size = 8192, hop = 256, sr = 44100, t0 = None, t1 = None):

	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	num_freq_samples = fft_size//2
	freqs = fourier.fft_freqs(fft_size, sr)
	num_ffts = last_fft_i-first_fft_i

	#skip the first bin (which is DC offset / 0Hz)
	log_freqs = np.log2(freqs[1:])
	#print("log_freqs",len(log_freqs))

	linspace_freqs = np.linspace(log_freqs[0], log_freqs[-1], num_freq_samples)

	#num_bins, num_ffts = D.shape
	#print("num_bins, last_fft_i",num_bins, last_fft_i )

	times = []
	
	#create a new array to store FFTs resampled on a log2 scale
	i2 = 0
	resampled = np.ones((num_freq_samples, num_ffts),)
	for i in range(first_fft_i, last_fft_i):
		#save the data of this frame
		t = i*hop/sr
		times.append(t)
		
		#skip the first bin (0Hz)
		resampled[:,i2] = np.interp(linspace_freqs, log_freqs, D[1:,i])
		i2 +=1

	#compute the change over each frame
	changes = np.ones(num_ffts-1)
	for i in range(num_ffts-1):
		#correlation against the next sample, output will be of the same length
		#TODO: this could be optimized by doing only, say 10 correlations instead of the full set (one correlation for each input sample)
		res = np.correlate(resampled[:,i], resampled[:,i+1], mode="same")
		#this should maybe hanned before argmax to kill obvious outliers
		i_peak = np.argmax(res)
		#interpolate the most accurate fit
		i_interp = parabolic(res, i_peak)[0]
		changes[i] = (num_freq_samples//2) -i_interp

	#sum up the changes to a speed curve
	speed = np.cumsum(changes)
	#up to this point, we've been dealing with interpolated "indices"
	#now, we are on the log scale
	#on log scale, +1 means double frequency or speed
	speed = speed / num_freq_samples * (log_freqs[-1]-log_freqs[0])

	#convert to scale and from log2 scale
	freqs = np.power(2, (9.9+ speed))
	return times[1:], freqs