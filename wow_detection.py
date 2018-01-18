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
	