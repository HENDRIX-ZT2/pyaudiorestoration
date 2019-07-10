import soundfile as sf
import os

def read_file(file_path):
	print("Reading",file_path)
	soundob = sf.SoundFile(file_path)
	return soundob.read(always_2d=True), soundob.samplerate, soundob.channels

def write_file(file_path, signal, sr, channels, suffix="_out"):
	# write the final signal
	with sf.SoundFile( os.path.splitext(file_path)[0]+suffix+".wav", 'w+', sr, channels, subtype='FLOAT') as outfile:
		outfile.write(signal)
	print("Finished",file_path)
	
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