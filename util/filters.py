from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""Performs a low, high or bandpass filter if low & highcut are in range"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	low_in_range = 0 < low < 1
	high_in_range = 0 < high < 1
	if low_in_range and high_in_range:
		b, a = butter(order, [low, high], btype='band')
	elif low_in_range and not high_in_range:
		b, a = butter(order, low, btype='high')
	elif not low_in_range and high_in_range:
		b, a = butter(order, high, btype='low')
	else:
		return data
	return filtfilt(b, a, data)
	