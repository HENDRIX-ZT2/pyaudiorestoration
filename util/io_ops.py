import logging

import soundfile as sf
import os


def read_file(audio_path):
	logging.info(f"Reading {audio_path}")
	soundob = sf.SoundFile(audio_path)
	signal = soundob.read(always_2d=True, dtype="float32")
	if len(signal) == 0:
		# multichannel flacs give errors due to a bug in libsoundfile, more info below
		# pysoundfile should soon be updated with a fixed dll
		# https://github.com/kenowr/read_flac
		raise AttributeError(f"Reading {audio_path} failed, possible libsoundfile bug")
	return signal, soundob.samplerate, soundob.channels


def write_file(audio_path, signal, sr, channels, suffix="_out"):
	# write the final signal
	with sf.SoundFile(f"{os.path.splitext(audio_path)[0]}{suffix}.wav", 'w+', sr, channels, subtype='FLOAT') as outfile:
		outfile.write(signal)
	logging.info(f"Wrote {audio_path}")


def read_trace(filename):
	"""
	filename: the name of the original audio file
	returns:
	data:	 a list of (times, frequencies) lists
	"""

	# write the data to the speed file
	print("Reading speed data")
	speedfilename = filename.rsplit('.', 1)[0] + ".speed"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for l in text_file:
				# just for completeness
				if l:
					if "?" in l:
						offset = float(l.split(" ")[1])
						data.append((offset, [], []))
					else:
						s = l.split(" ")
						data[-1][1].append(float(s[0]))
						data[-1][2].append(float(s[1]))
	return data


def read_regs(filename):
	"""
	filename: the name of the original audio file
	returns:
	data:	 a list of sine parameters
	"""

	# write the data to the speed file
	print("Reading regression data")
	speedfilename = filename.rsplit('.', 1)[0] + ".sin"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for l in text_file:
				# just for completeness
				if l:
					data.append([float(v) for v in l.split(" ")])
	return data


def read_lag(filename):
	# write the data to the speed file
	print("Reading lag data")
	speedfilename = filename.rsplit('.', 1)[0] + ".syn"
	data = []
	if os.path.isfile(speedfilename):
		with open(speedfilename, "r") as text_file:
			for line in text_file:
				if line:
					data.append([float(v) for v in line.split(" ")])
	return data
