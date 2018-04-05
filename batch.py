import soundfile as sf
import numpy as np
from time import time
import os
import fourier
import wow_detection
import resampling

def write_speed(filename, speed_curve, piece=None):
	piece_str  = ""
	if piece is not None:
		piece_str = "_"+format(piece, '03d')
	#only for testing
	speedfilename = filename.rsplit('.', 1)[0]+piece_str+".npy"
	np.save(speedfilename, speed_curve, allow_pickle=True, fix_imports=True)

def trace_all(filename, blocksize, overlap, fft_size, fft_overlap, hop, start= 16.7):
	start_time = time()
	#how many frames do we want?
	half = overlap//2
	quart = overlap//4
	#read in chunks for FFT
	soundob = sf.SoundFile(filename)
	sr = soundob.samplerate
	block_start = 0
	for i, block in enumerate(soundob.blocks( blocksize=blocksize*hop, overlap=overlap*hop)):
		# if i not in (0, 1):
			# continue
		print("Block from",block_start,"to",block_start+len(block)/sr)
		imdata = np.abs(fourier.stft(block, fft_size, hop, "hann"))
		
		#we can't do the start automatically
		#note: this is already accorded for in trace_peak
		if i == 0:
			t0 = start
			lag = 0
		else:
			#we only start at a good FFT, not influenced by cut artifacts
			t0 = fft_size/2/sr
			lag = fft_size//2 //hop
		print("start at",t0)
		#times, freqs = wow_detection.trace_peak2(imdata, fft_size, hop, sr, fL = 900, fU = 1100, t0 = t0, t1 = None, tolerance = 1, adaptation_mode="Average")
		times, freqs = wow_detection.trace_peak_static(imdata, fft_size, hop, sr, fL = 900, fU = 1100, t0 = t0, t1 = None, tolerance = 1, adaptation_mode="Average")
		if i == 0:
			times = times[:-half]
			freqs = freqs[:-half]
		else:
			times = times[half-lag:-half]
			freqs = freqs[half-lag:-half]
		
		# import matplotlib.pyplot as plt
		# plt.figure()
		# #plt.plot(mins, label="0", alpha=0.5)
		# #plt.plot(maxs, label="1", alpha=0.5)
		
		# #maybe: set dropout freq to mean(freqs)
		# plt.plot(times, freqs , label="1", alpha=0.5)
		# plt.xlabel('Speed')
		# plt.ylabel('Freg Hz')
		# plt.legend(frameon=True, framealpha=0.75)
		# plt.show()
		speed = np.stack((times, freqs), axis=1)
		write_speed(filename, speed, piece=i)
		block_start+= ((blocksize*hop - overlap*hop) / sr)
	dur = time() - start_time
	print("duration",dur)

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
	print('Analyzing ' + filename + '...')
	start_time = time()
	print(resampling_mode)
	#read the file
	soundob = sf.SoundFile(filename)
	sr = soundob.samplerate
	in_len = 0
	outfilename = filename.rsplit('.', 1)[0]+'_cloned.wav'
	with sf.SoundFile(outfilename, 'w', sr, 1, subtype='FLOAT') as outfile:
		for i, in_block in enumerate(soundob.blocks( blocksize=blocksize*hop, overlap=overlap*hop)):
			# if i not in (0, 1,2,3):
				# continue
			print(i, len(in_block))
			speed_curve = np.load(speed_curve_names[i])
			times = speed_curve[:,0]
			print("times:",times[0],times[len(times)-1])
			#note: this expects a a linscale speed curve centered around 1 (= no speed change)
			speeds = speed_curve[:,1]/1000
			
			#only update if it changes
			if len(in_block) != in_len:
				in_len = len(in_block)
				samples_in2 = np.arange( in_len )
			offsets_speeds = resampling.prepare_linear_or_sinc(times*sr, speeds)
			#these must be called as generators...
			if resampling_mode in ("Sinc",):
				for i in resampling.sinc_kernel(outfile, offsets_speeds, in_block, samples_in2, NT = 50):
					pass
			elif resampling_mode in ("Linear",):
				for i in resampling.linear_kernel(outfile, offsets_speeds, in_block, samples_in2):
					pass

	dur = time() - start_time
	print("duration",dur)

#settings...
fft_size=512
fft_overlap=16
hop=512//16
overlap=100
blocksize=100000
#speedname = "C:/Users/arnfi/Desktop/nasa/A11_T876_HR1L_CH1.wav"
speedname = "C:/Users/arnfi/Desktop/nasa/A11_T648_HR1U_CH1.wav"
filename = "C:/Users/arnfi/Desktop/nasa/A11_T648_HR1U_CH1.wav"
#speedname = "C:/Users/arnfi/Desktop/nasa/test.wav"
#filename = "C:/Users/arnfi/Desktop/nasa/A11_T876_HR1L_CH2.wav"
#trace_all(speedname, blocksize, overlap, fft_size, fft_overlap, hop, start=0)
#show_all(speedname, hi=1020, lo=948)
resample_all(speedname, filename, blocksize, overlap, hop, resampling_mode = "Linear")