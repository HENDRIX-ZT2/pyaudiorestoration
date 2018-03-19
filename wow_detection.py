import numpy as np
import fourier
	
#https://github.com/librosa/librosa/blob/86275a8949fb4aef3fb16aa88b0e24862c24998f/librosa/core/pitch.py#L165
#librosa piptrack

#todo
# M. Lagrange, S. Marchand, and J. B. Rault, “Using linear prediction to enhance the tracking of partials,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP’04) , May 2004, vol. 4, pp. 241–244.

#2003 enhanced partial tracking - good!
	
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
	
def trace_sine(fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None):
	#https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
	meanfreq = (fU+fL)/2
	amp = (fU-fL ) / 2
	log2amp = (np.log2(fU)-np.log2(fL))/2
	print("amp",amp, log2amp)
	
	fft_sr = hop/sr
	times = np.arange(t0, t1, fft_sr)
	period = 1.7 #seconds
	#so this is in log2 scale
	#basic sine curve, amplitude 1
	sine = np.sin(times* 2*np.pi / period )# / 180. )

	#to get to the desired mean frequency, add the log of that freq
	#and then take it power 2
	sine_on_hz = np.power(2, sine * log2amp + np.log2(meanfreq))
	#sine_on_log = sine * amp + meanfreq

	# #this can be the speed curve
	# sine_in = sine*log2amp
	# #this is equivalent
	# sine_back = np.log2(sine_on_hz)
	# sine_back = sine_back-np.mean(sine_back)



	#print("sine")
	#times = []
	#freqs = []
	return times, sine_on_hz#, LfL, LfU

def trace_peak(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None):
	
	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	#bin indices of the starting band
	#N = 1 + fft_size
	N = fft_size
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
		
	freqs = []
	times = []
	
	for i in range(first_fft_i, last_fft_i):
		# * np.hanning(NU-NL) does not really make it better
		# instead adapt the NL, NU borders consecutively
		i_raw = np.argmax( D[NL:NU, i] )+NL
		i_interp = (D[i_raw-1, i] - D[i_raw+1, i]) / (D[i_raw-1, i] - 2 * D[i_raw, i] + D[i_raw+1, i]) / 2 + i_raw
		freq = sr * i_interp / N
		freqs.append(freq)
		#save the data of this frame
		t = i*hop/sr
		times.append(t)
	return times, freqs
	
	
def fit_sin(tt, yy, assumed_freq=None):
	'''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
	#by unsym from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-np
	tt = np.array(tt)
	yy = np.array(yy)
	#new: only use real input, so rfft
	ff = np.fft.rfftfreq(len(tt), (tt[1]-tt[0]))	  # assume uniform spacing
	fft_data = np.fft.rfft(yy)[1:]
	#new: add an assumed frequency as an optional pre-processing step
	if assumed_freq:
		period = tt[1]-tt[0]
		N = len(yy)+1
		#find which bin this corresponds to
		peak_est = int(round(assumed_freq * N * period))
		#print("peak_est", peak_est)
		#window the bins accordingly to maximize the expected peak
		win = np.interp( np.arange(0, len(fft_data)), (0, peak_est, len(fft_data)), (0, 1, 0) )
		#print("win", win)
		#does this affect the phase?
		fft_data *= win
	peak_bin = np.argmax(np.abs(fft_data))+1
	print("peak_bin", peak_bin)
	guess_freq = ff[peak_bin]	# excluding the zero frequency "peak", which is related to offset
	guess_amp = np.std(yy) * 2.**0.5
	guess_offset = np.mean(yy)
	#new: get the phase at the peak
	guess_phase = np.angle(fft_data[peak_bin])
	#print("guess_phase",guess_phase)
	guess = np.array([guess_amp, 2.*np.pi*guess_freq, guess_phase, guess_offset])

	#using cosines does not seem to make it better?
	def sinfunc(t, A, w, p, c):	 return A * np.sin(w*t + p) + c
	import scipy.optimize
	popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
	A, w, p, c = popt
	f = w/(2.*np.pi)
	fitfunc = lambda t: A * np.sin(w*t + p) + c
	return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

	
def trace_sine_reg(speed_curve, t0, t1, rpm = None):
	"""Perform a regression on an area of the master speed curve to yield a sine fit"""
	#the master speed curve
	times = speed_curve[:,0]
	speeds = speed_curve[:,1]
	
	period = times[1]-times[0]
	#which part to sample
	ind_start = int(t0 / period)
	ind_stop = int(t1 / period)
	
	try:
		#33,3 RPM means the period of wow is = 1,8
		#hence its frequency 1/1,8 = 0,55
		assumed_freq = float(rpm) / 60
		print("Source RPM:",rpm)
		print("Assumed Wow Frequency:",assumed_freq)
	except:
		assumed_freq = None

	res = fit_sin(times[ind_start:ind_stop], speeds[ind_start:ind_stop], assumed_freq=assumed_freq)
	
	return res["amp"], res["omega"], res["phase"], res["offset"]
	
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