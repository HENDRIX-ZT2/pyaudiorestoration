import numpy as np
import fourier
import scipy
	
#https://github.com/librosa/librosa/blob/86275a8949fb4aef3fb16aa88b0e24862c24998f/librosa/core/pitch.py#L165
#librosa piptrack

#todo
# M. Lagrange, S. Marchand, and J. B. Rault, “Using linear prediction to enhance the tracking of partials,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP’04) , May 2004, vol. 4, pp. 241–244.

#2003 enhanced partial tracking - good!


#todo: create a hann window in log space?
# or just lerp 0, 1, 0 for NL, i, NU
	
def COG(magnitudes, freqs, NL, NU):
	#adapted from Czyzewski et al. (2007)
	#18
	#calculate the COG with hanned frequency importance
	#error in the printed formula: the divisor also has to contain the hann window
	weighted_magnitudes = np.hanning(NU-NL) * magnitudes[NL:NU]
	return np.sum(weighted_magnitudes * np.log2(freqs[NL:NU])) / np.sum(weighted_magnitudes)

def adapt_band(freqs, num_bins, freq_2_bin, tolerance, adaptation_mode, i):
	"""
	The goal of this function is, given the last frequency peaks, to return suitable boundaries for the next frequency detection
	
	freqs: the input frequency list
	num_bins: the amount of bins, for limiting
	freq_2_bin: the factor a freq has to be multiplied with to get the corresponding bin
	tolerance: tolerance above and below peak, in semitones (1/12th of an octave)
	adaptation_mode: how should the "prediction" happen?
	
	maxrise: tolerance for absolute rise between a frame, in semitones (redundant? - could be a fraction of tolerance), should this be done here or in postpro?
	"""
	logfreq = np.log2( freqs[i] )
	if adaptation_mode == "None":
		#keep the initial limits
		#implement this elsewhere?
		pass
	elif adaptation_mode == "Constant":
		#do nothing to the freq
		pass
	elif adaptation_mode == "Linear":
		# repeat the trend and predict the freq of the next bin to get the boundaries accordingly
		#this should be done in logspace
		if len(freqs) > 1:
			delta = logfreq - np.log2( freqs[i-2] )
			logfreq += delta
	elif adaptation_mode == "Average":
		#take the average of n deltas,
		#and use the preceding freqs as the starting point to avoid outliers?
		logfreqs = np.log2( freqs[max(0,i-3):i+1] )
		deltas = np.diff( logfreqs )
		logfreq = logfreqs[0]
		if len(deltas): logfreq += np.nanmean(deltas)*len(logfreqs)
		#print(deltas,logfreq)
		
	fL = np.power(2, (logfreq-tolerance/12))
	fU = np.power(2, (logfreq+tolerance/12))
	NL = max(1, min(num_bins-3, int(round(fL * freq_2_bin))) )
	NU = min(num_bins-2, max(1, int(round(fU * freq_2_bin))) )
	
	if NU-NL > 5:
		window = np.interp(np.arange(NL, NU), (NL, np.power(2, logfreq)*freq_2_bin, NU-1), (0,1,0))
		#print(len(window),window)
	else:
		window = np.ones(NU-NL)
	return NL, NU, window, logfreq


def trace_cog(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None, adaptation_mode="Linear"):
	#Todo: calculate tolerance from fL, fU

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
	
def get_peak(D, i, NL, NU, window, freq_2_bin):
	fft_data = D[NL:NU, i] * window
	i_raw = np.argmax( fft_data )+NL
	i_interp = (D[i_raw-1, i] - D[i_raw+1, i]) / (D[i_raw-1, i] - 2 * D[i_raw, i] + D[i_raw+1, i]) / 2 + i_raw
	#sometimes the interpolation fails bad, then just use the raw index
	if i_interp < 1:
		i_interp = i_raw
	return i_interp / freq_2_bin, np.mean(20*np.log10(fft_data+.0000001))
	
def trace_peak_static(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None, tolerance = 1, adaptation_mode="Linear", dB_cutoff=-82):
	"""
	tolerance: tolerance above and below, in semitones (1/12th of an octave)
	"""
	
	#print("adaptation_mode",adaptation_mode)
	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
	if t1:
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	#bin indices of the starting band
	#N = 1 + fft_size
	N = fft_size
	if first_fft_i == last_fft_i:
		print("No point in tracing just one FFT")
		return [],[]
	
	#let's say we take the lower as the starting point
	#define the tolerance in semitones
	#on log2 scale, one semitone is 1/12
	#so take our start value, log2 it, +- 1/12
	#and take power of two for both
	freq = (fL+fU)/2
	logfreq = np.log2( freq )
	fL = np.power(2, (logfreq-tolerance/12))
	fU = np.power(2, (logfreq+tolerance/12))
	NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )

	freq_2_bin = N / sr
	#how many octaves may the pitch rise between consecutive FFTs?
	#note that this is of course influenced by overlap and FFT size
	fft_data = D[NL:NU, first_fft_i: last_fft_i]
	i_raws = np.argmax( fft_data, axis=0)+NL
	times = np.arange(first_fft_i, last_fft_i)*hop/sr
	freqs = np.empty(last_fft_i-first_fft_i)
	for out_i, i in enumerate(range(first_fft_i, last_fft_i)):
		i_raw = i_raws[out_i]
		i_interp = ((D[i_raw-1, i] - D[i_raw+1, i]) / (D[i_raw-1, i] - 2 * D[i_raw, i] + D[i_raw+1, i]) / 2 + i_raw)  / freq_2_bin
		if i_interp < 1:
			i_interp = i_raw
		freqs[out_i] = i_interp
	dBs = np.nanmean(20*np.log10(D[NL:NU, first_fft_i: last_fft_i]+.0000001), axis=0)
	freqs[dBs < dB_cutoff] = np.mean(freqs)
	return times, freqs#, dbs
	
def trace_peak2(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None, tolerance = 1, adaptation_mode="Linear", dB_cutoff=-82):
	"""
	tolerance: tolerance above and below, in semitones (1/12th of an octave)
	"""
	
	#print("adaptation_mode",adaptation_mode)
	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
	if t1:
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	#bin indices of the starting band
	#N = 1 + fft_size
	N = fft_size
	#clamp to valid frequency range
	fL = max(1.0, fL)
	fU = min(sr/2, fU)
	# #make sure it doesn't escape the frequency limits
	# NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	# NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
	
	if first_fft_i == last_fft_i:
		print("No point in tracing just one FFT")
		return [],[]
		
	freqs = np.ones(last_fft_i-first_fft_i)
	times = np.ones(last_fft_i-first_fft_i)
	
	#let's say we take the lower as the starting point
	#define the tolerance in semitones
	#on log2 scale, one semitone is 1/12
	#so take our start value, log2 it, +- 1/12
	#and take power of two for both
	freq = (fL+fU)/2
	freq_o = freq
	logfreq = np.log2( freq )
	log_prediction = logfreq
	fL = np.power(2, (logfreq-tolerance/12))
	fU = np.power(2, (logfreq+tolerance/12))
	NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
	window = np.ones(NU-NL)
	gap_len=0
	
	freq_2_bin = N / sr
	dBs = []
	#how many octaves may the pitch rise between consecutive FFTs?
	#note that this is of course influenced by overlap and FFT size
	#maxraise = .2/12
	maxraise = 1/12
	max_delta = 0.04
	stable=True
	dif = 0
	for out_i, i in enumerate(range(first_fft_i, last_fft_i)):
		freq, dB = get_peak(D, i, NL, NU, window, freq_2_bin)
		frametime =  i*hop/sr
		
		#discard this frame if
		#the signal level is too low
		#or the pitch slope is too steep
		if out_i > 1: dif = np.log2(freq) - np.log2( freqs[out_i-1] )
		#is it loud enough?
		if dB < dB_cutoff:
			#no? then mark it as unstable
			if stable:
				print("gap start at ",frametime)
				stable=False
		#ok, there is enough dB
		#but is the pitch stable?
		else:
			if dif > max_delta:
				if stable:
					print("big rise at",frametime,freq)
					stable=False
			elif dif < -max_delta:
				if stable:
					print("big fall at",frametime,freq)
					stable=False
			else:
				# go back to normal mode
				if not stable:
					print("normal at",frametime)
					stable=True
		#then correct the peak
		if not stable:
			freq = freq_o
		
		freqs[out_i] = freq
		times[out_i] = frametime
	
	return times, freqs#, dbs
	
def trace_peak(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None, tolerance = 1, adaptation_mode="Linear", dB_cutoff=-75):
	"""
	tolerance: tolerance above and below, in semitones (1/12th of an octave)
	"""
	
	#how many seconds must pass between 
	MIN_DROPOUT_DIST = 0.05
	
	#print("adaptation_mode",adaptation_mode)
	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
	if t1:
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
	
	#bin indices of the starting band
	#N = 1 + fft_size
	N = fft_size
	#clamp to valid frequency range
	fL = max(1.0, fL)
	fU = min(sr/2, fU)
	# #make sure it doesn't escape the frequency limits
	# NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	# NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
	
	if first_fft_i == last_fft_i:
		print("No point in tracing just one FFT")
		return [],[]
		
	freqs = np.ones(last_fft_i-first_fft_i)
	times = np.ones(last_fft_i-first_fft_i)
	
	#let's say we take the lower as the starting point
	#define the tolerance in semitones
	#on log2 scale, one semitone is 1/12
	#so take our start value, log2 it, +- 1/12
	#and take power of two for both
	freq = (fL+fU)/2
	logfreq = np.log2( freq )
	log_prediction = logfreq
	fL = np.power(2, (logfreq-tolerance/12))
	fU = np.power(2, (logfreq+tolerance/12))
	NL = max(1, min(num_bins-3, int(round(fL * N / sr))) )
	NU = min(num_bins-2, max(1, int(round(fU * N / sr))) )
	NL_o = NL
	NU_o = NU
	window = np.ones(NU-NL)
	gap_len=0
	
	freq_2_bin = N / sr
	#dbs = []
	#how many octaves may the pitch rise between consecutive FFTs?
	#note that this is of course influenced by overlap and FFT size
	#maxraise = .2/12
	maxraise = 1/12
	max_delta = 0.02
	stable=True
	dif = 0
	dB_cutoff = -180
	db_buffer_len = 100
	db_ring = np.empty(db_buffer_len)
	db_ring.fill(np.nan)
	db_i = 0
	last_stable_time = 0
	for out_i, i in enumerate(range(first_fft_i, last_fft_i)):
		try:
			freq, dB = get_peak(D, i, NL, NU, window, freq_2_bin)
		except ValueError:
			print("peak error at",i)
			window = np.ones(NU_o-NL_o)
			freq, dB = get_peak(D, i, NL_o, NU_o, window, freq_2_bin)
		frametime =  i*hop/sr
		
		#print(freq)
		#discard this frame if
		#the signal level is too low
		#or the pitch slope is too steep
		if out_i > 1: dif = np.log2(freq) - np.log2( freqs[out_i-1] )
		#is it loud enough?
		if dB < dB_cutoff:
			#no? then mark it as unstable
			if stable:
				print("gap start at ",frametime)
				stable=False
				last_stable_time = frametime
		#ok, there is enough dB
		#but is the pitch stable?
		else:
			if dif > max_delta:
				if stable:
					print("big rise at",frametime,freq)
					stable=False
					last_stable_time = frametime
			elif dif < -max_delta:
				if stable:
					print("big fall at",frametime,freq)
					stable=False
					last_stable_time = frametime
			else:
				#go back to normal mode
				if not stable:
					time_since_stable = frametime -last_stable_time
					#the duration of the dropout must be longer than X?
					#if time_since_stable > 0.05:
					#	print("normal at",frametime, time_since_stable)
					print("normal at",frametime)
					stable=True
					start_stable = frametime
						
					#now trace back to where it all began
		
		#only add good data to our volume ring buffer
		if stable:
			db_ring[db_i] = dB
			db_i += 1
			if db_i == db_buffer_len:
				db_i = 0
		#then correct the peak
		if not stable:
			freq = np.mean(freqs[max(0,out_i-80):out_i-1])
		
		#a good signal should be at least X frames long
		#start_stable
		
		#if mean is significantly bigger than current dB value, we have a dropout
		#de = np.nanmean(db_ring)-dB
		dB_cutoff = np.nanmean(db_ring) - 15
		#print(dB_cutoff)
		# if de > 15:
			# print("dropout at ",frametime, dB, de)
			
		#todo:
		#store start of last dropout /difference to last dropout / duration of dropout
		#then backtrack until start of dropout when signal is back in reach

		freqs[out_i] = freq
		times[out_i] = frametime
		try:
			NL, NU, window, log_prediction = adapt_band(freqs, num_bins, freq_2_bin, tolerance, adaptation_mode, out_i)
		except ValueError:
			print("ERROR",len(freqs))
			NL = NL_o
			NU = NU_o
			window = np.ones(NU_o-NL_o)
			log_prediction = 0
	return times, freqs#, dbs
	
def trace_correlation(D, fft_size = 8192, hop = 256, sr = 44100, fL = 2260, fU = 2320, t0 = None, t1 = None):

	#start and stop reading the FFT data here, unless...
	first_fft_i = 0
	num_bins, last_fft_i = D.shape
	#we have specified start and stop times, which is the usual case
	if t0:
		#make sure we force start and stop at the ends!
		first_fft_i = max(first_fft_i, int(t0*sr/hop)) 
		last_fft_i = min(last_fft_i, int(t1*sr/hop))
		
	NL = max(1, min(num_bins-3, int(round(fL * fft_size / sr))) )
	NU = min(num_bins-2, max(1, int(round(fU * fft_size / sr))) )
	
	#print(NL,NU)
	num_freq_samples = (NU-NL)*4
	freqs = fourier.fft_freqs(fft_size, sr)
	num_ffts = last_fft_i-first_fft_i

	#skip the first bin (which is DC offset / 0Hz)
	log_freqs = np.log2(freqs[NL:NU])

	linspace_freqs = np.linspace(log_freqs[0], log_freqs[-1], num_freq_samples)

	times = []
	
	#create a new array to store FFTs resampled on a log2 scale
	i2 = 0
	resampled = np.ones((num_freq_samples, num_ffts),)
	for i in range(first_fft_i, last_fft_i):
		#save the data of this frame
		t = i*hop/sr
		times.append(t)
		
		#skip the first bin (0Hz)
		#resampled[:,i2] = np.interp(linspace_freqs, log_freqs, D[NL:NU,i])
		interolator = scipy.interpolate.interp1d(log_freqs, D[NL:NU,i], kind='quadratic')
		resampled[:,i2] = interolator(linspace_freqs)
		i2 +=1

	wind = np.hanning(num_freq_samples)
	#compute the change over each frame
	changes = np.ones(num_ffts-1)
	for i in range(num_ffts-1):
		#correlation against the next sample, output will be of the same length
		#TODO: this could be optimized by doing only, say 10 correlations instead of the full set (one correlation for each input sample)
		res = np.correlate(resampled[:,i]*wind, resampled[:,i+1]*wind, mode="same")
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

	
	log_mean_freq = np.log2((fL+fU)/2)
	#convert to scale and from log2 scale
	freqs = np.power(2, (log_mean_freq+ speed))
	return times[1:], freqs
	
	
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
	
	#return res["amp"], res["omega"], res["phase"], res["offset"]
	return res["amp"], res["omega"], res["phase"], 0
	
def parabolic(f, x):
	"""Helper function to refine a peak position in an array"""
	xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
	yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
	return (xv, yv)
	
