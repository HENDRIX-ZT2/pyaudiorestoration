import soundfile as sf
import numpy as np
from time import time
import os
from util import fourier, resampling, wow_detection

def sec_to_timestamp(t):
	m, s = divmod(t, 60)
	s, ms = divmod(s*1000, 1000)
	h, m = divmod(m, 60)
	return "%d:%02d:%02d:%03d h:m:s:ms" % (h, m, s, ms)
	
def write_speed(filename, speed_curve, piece=None):
	piece_str  = ""
	if piece is not None:
		piece_str = "_"+format(piece, '03d')
	#only for testing
	speedfilename = filename.rsplit('.', 1)[0]+piece_str+".npy"
	np.save(speedfilename, speed_curve, allow_pickle=True, fix_imports=True)

def trace_all(filename, blocksize, overlap, fft_size, fft_overlap, hop, start= 16.7, fL = 900, fU = 1100):
	start_time = time()
	soundob = sf.SoundFile(filename)
	sr = soundob.samplerate
	block_start = 0
	num_cores = os.cpu_count()
	#we read the audio in with overlap
	for i, block in enumerate(soundob.blocks( blocksize=blocksize*hop, overlap=overlap*hop)):
		# if i not in (0, 1):
			# continue
		print("Tracing from",sec_to_timestamp(block_start),"to",sec_to_timestamp(block_start+len(block)/sr))
		imdata = fourier.stft(block, fft_size, hop, "hann", num_cores)
		
		#we can't do the start automatically
		#note: this is already accorded for in trace_peak
		if i == 0:
			t0 = start
		else:
			#we only start at a good FFT, not influenced by cut artifacts
			t0 = fft_size/2/sr
		times, freqs = wow_detection.trace_peak_static(imdata, fft_size, hop, sr, fL = fL, fU = fU, t0 = t0, t1 = None, tolerance = 1, adaptation_mode="Average")
		#now we trim the start and end of each trace to remove end artifacts
		half = overlap//2
		if i == 0:
			speed = np.stack((times[:-half], freqs[:-half]), axis=1)
		else:
			speed = np.stack((times[half:-half], freqs[half:-half]), axis=1)
		speed[:,0]*=sr
		speed[:,1]/=1000
		write_speed(filename, speed, piece=i)
		block_start+= ((blocksize*hop - overlap*hop) / sr)
	dur = time() - start_time
	print("duration",sec_to_timestamp(dur))

def show_all(speedname, hi=1020, lo=948):
	dir = os.path.dirname(speedname)
	name = os.path.basename(speedname).rsplit('.', 1)[0]
	files = [os.path.join(dir,file) for file in os.listdir(dir) if name in file and file.endswith(".npy")]

	mins=[]
	maxs=[]
	
	speedcurves = []
	for file in files:
		speedcurve = np.load(file)
		speedcurves.append(speedcurve)
		
		ma = np.max(speedcurve[:,1])
		mi = np.min(speedcurve[:,1])
		mins.append( mi)
		maxs.append(ma)
		if mi < lo:
			print("too low", file)
		if ma > hi:
			print("too high", file)
	import matplotlib.pyplot as plt
	plt.figure()
	#plt.plot(mins, label="0", alpha=0.5)
	#plt.plot(maxs, label="1", alpha=0.5)
	
	#maybe: set dropout freq to mean(freqs)
	plt.plot(speedcurves[0][:,0], speedcurves[0][:,1], label="1", alpha=0.5)
	plt.xlabel('Speed')
	plt.ylabel('Freg Hz')
	plt.legend(frameon=True, framealpha=0.75)
	plt.show()
	
	
def resample_all(speedname, filename, blocksize, overlap, hop, resampling_mode = "Linear"):
	dir = os.path.dirname(speedname)
	name = os.path.basename(speedname).rsplit('.', 1)[0]
	speed_files = [os.path.join(dir,file) for file in os.listdir(dir) if name in file and file.endswith(".npy")]
	batch_res(filename, blocksize, overlap, speed_files, resampling_mode)
	
def batch_res(filename, blocksize, overlap, speed_curve_names, resampling_mode):
	print('Resampling ' + filename + '...',resampling_mode)
	start_time = time()
	#read the file
	soundob = sf.SoundFile(filename)
	in_len = 0
	outfilename = filename.rsplit('.', 1)[0]+'_cloned.w64'
	with sf.SoundFile(outfilename, 'w', soundob.samplerate, 1, subtype='FLOAT') as outfile:
		for i, in_block in enumerate(soundob.blocks( blocksize=blocksize*hop, overlap=overlap*hop)):
			try:
				speed_curve = np.load(speed_curve_names[i])
			except:
				print("Resampling aborted! No more speed curves, can not resample block",i)
				break
			print("Block",i)
			#only update if it changes
			if len(in_block) != in_len:
				in_len = len(in_block)
				samples_in2 = np.arange( in_len )
			offsets_speeds = resampling.prepare_linear_or_sinc(speed_curve[:,0], speed_curve[:,1])
			#these must be called as generators...
			if resampling_mode in ("Sinc",):
				for i in resampling.sinc_kernel(outfile, offsets_speeds, in_block, samples_in2, NT = 50):
					pass
			elif resampling_mode in ("Linear",):
				for i in resampling.linear_kernel(outfile, offsets_speeds, in_block, samples_in2):
					pass

	dur = time() - start_time
	print("duration",sec_to_timestamp(dur))

#settings...
# #at 8kHz
# fft_size=512
#at 44kHz
# fft_size=4096
# fft_overlap=16
fft_size=2048
fft_overlap=16
hop=fft_size//fft_overlap
overlap=10
blocksize=100000
#speedname = "C:/Users/arnfi/Desktop/nasa/A11_T876_HR1L_CH1.wav"
speedname = "C:/Users/arnfi/Desktop/nasa/A11_T869_HR1U_CH1.wav"
filename = "C:/Users/arnfi/Desktop/nasa/A11_T869_HR1U_CH7.wav"
#speedname = "C:/Users/arnfi/Desktop/nasa/test.wav"
#filename = "C:/Users/arnfi/Desktop/nasa/A11_T876_HR1L_CH2.wav"
trace_all(speedname, blocksize, overlap, fft_size, fft_overlap, hop, start=0, fL = 900, fU = 1100)
# show_all(speedname, hi=1020, lo=948)
# resample_all(speedname, filename, blocksize, overlap, hop, resampling_mode = "Linear")