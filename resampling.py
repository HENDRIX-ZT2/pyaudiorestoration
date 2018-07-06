import numpy as np
from time import time
import soundfile as sf
import os
from numba import jit

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

#jit notes: parallel makes it at least as slow as the normal one
# lazy optimization is recommended
# @jit(float32[:](float32[:], int64, int64, float64[:], float64[:], int64, int64, float32[:], int32[:]), nopython=True, cache=True, parallel=True)
@jit(nopython=True, nogil=True, cache=True)
def sinc_core(block, offset, positions, win_func, NT, in_len, signal, samples_in2):
	block_len = len(positions)
	#this is probably not required!
	offset = int(offset)
	for i in range( block_len):
		#now we are at the end and we can yield the rest of the piece
		p = positions[i]
		#map the output to the input
		#error diffusion here makes no significant difference
		ind = int(round(p))+offset
		
		#can we end it here already?
		#then cut the block and break out of this loop
		if ind == in_len:
			return block[0:i]
			
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
		
		si = np.sinc ((samples_in2[lower:upper]-offset - p) * fc) * fc
		#note that in some end cases, len may be 0 but win_func gets another len so it would break
		#TODO: this is probably obsolete and could get its indices like the other two, because the end is cut off already
		block[i] = np.sum(si * signal[lower:upper] * win_func[0:len(si)])
	return block

def sinc_kernel(outfile, offsets_speeds, signal, samples_in2, NT = 50):
	"""
	outfile: an open sound file object in w+ mode
	offsets_speeds: list of (offset, positions) tuples
	signal: mono, 1D input samples
	samples_in2: input positions
	NT (optional): more NT means better quality (less overtones); tradeoff between quality and speed
	"""
	in_len = len(samples_in2)
	win_func = np.hanning(2*NT)
	for offset, positions in offsets_speeds:
		#note that currently, numba does not allow np array creation in nopython mode, so we have to pass the empty output array...
		block = sinc_core(np.empty(len(positions), "float32"), offset, positions, win_func, NT, in_len, signal, samples_in2)
		outfile.write( block )
		yield 1
					
def linear_kernel(outfile, offsets_speeds, signal, samples_in2, prog_sig=None):
	"""
	outfile: an open sound file object in w+ mode
	offsets_speeds: list of (offset, positions) tuples
	signal: mono, 1D input samples
	samples_in2: input positions
	prog_sig (optional): reference to the progress bar for the gui
	"""
	for offset, positions in offsets_speeds:
		block = np.interp(positions, samples_in2-int(offset), signal)
		outfile.write( block )
		#print("len(block)",len(block))
		yield 1

def update_progress(prog_sig, progress, prog_fac):
	if prog_sig:
		progress += prog_fac
		prog_sig.notifyProgress.emit(progress)
	return progress
	
def prepare_linear_or_sinc(sampletimes, speeds):
	write_after=400000
	
	periods = np.diff(sampletimes)
		
	offsets_speeds = []
	#the first time stamp is our offset value
	#normal pyrespeeder speed curves start at 0, but that is not universal
	offset = int(sampletimes[0])
	err = 0
	temp_offset = 0
	temp_pos = []
	for i in range(0, len(speeds)-1):
		#save a new block for interpolation
		if len(temp_pos)* periods[i] > write_after:
			offsets_speeds.append( ( offset, np.concatenate(temp_pos) ) )
			offset += temp_offset
			temp_offset = 0
			temp_pos = []
		#we want to know how many samples there are in this section, so get the period (should be the same for all sections)
		mean_speed = ( (speeds[i]+speeds[i+1])/ 2 )
		#the desired number of samples in this block - this is clearly correct
		n = periods[i]*mean_speed
		
		#without dithering here, we get big gaps at the end of each segment!
		inerr = n + err
		n = round(inerr)
		err =  inerr-n
		
		block_speeds = np.interp(np.arange(n), (0, n-1), (speeds[i], speeds[i+1]) )
		positions = np.cumsum(1/block_speeds)
		
		# this is more accurate for long pieces, but a bit slower and gives clicks for short pieces
		# samples_in = np.arange(0, period)
		# block_speeds = np.interp(samples_in, (0, period),(speeds[i], speeds[i+1])  )
		# positions = np.interp(np.arange(0, n), np.cumsum(block_speeds), samples_in)
		
		temp_pos.append( positions +  temp_offset)
		if len(positions):
			temp_offset+=positions[-1]
		else:
			print("ERROR: no positions in segment - something was wrong with the speed curve (negative speed?)")
		#temp_offset+=period
	# the end
	if temp_pos: offsets_speeds.append( (offset, np.concatenate(temp_pos) ) )
	return offsets_speeds

def run(filename, speed_curve=None, resampling_mode = "Linear", sinc_quality=50, use_channels = [0,], prog_sig=None):

	print('Resampling ' + filename + '...', resampling_mode, sinc_quality, use_channels)
	if prog_sig: prog_sig.notifyProgress.emit(0)
		
	#read the file
	soundob = sf.SoundFile(filename)
	signal = soundob.read(always_2d=True, dtype='float32')
	sr = soundob.samplerate
	
	sampletimes = speed_curve[:,0]*sr
	#note: this expects a a linscale speed curve centered around 1 (= no speed change)
	speeds = speed_curve[:,1]
		
	samples_in2 = np.arange( len(signal[:,0]) )
	start_time = time()
	offsets_speeds = prepare_linear_or_sinc(sampletimes, speeds)
	dur = time() - start_time
	print("Preparation took",dur)
	num_blocks = len(offsets_speeds)
	
	print("Num Blocks",num_blocks)
	start_time = time()
	prog_fac = 100 / num_blocks / len(use_channels)
	progress = 0
	#resample on mono channels and export each separately as repeat does not like more dimensions
	for channel in use_channels:
		print('Processing channel ',channel)
		outfilename = filename.rsplit('.', 1)[0]+str(channel)+'.wav'
		with sf.SoundFile(outfilename, 'w+', sr, 1, subtype='FLOAT') as outfile:
			if resampling_mode == "Sinc":
				for i in sinc_kernel(outfile, offsets_speeds, signal[:,channel], samples_in2, NT = sinc_quality):
					progress = update_progress(prog_sig, progress, prog_fac)
			elif resampling_mode == "Linear":
				for i in linear_kernel(outfile, offsets_speeds, signal[:,channel], samples_in2):
					progress = update_progress(prog_sig, progress, prog_fac)
	if prog_sig: prog_sig.notifyProgress.emit(100)
	dur = time() - start_time
	print("Resampling took",dur)
	print("Done!\n")