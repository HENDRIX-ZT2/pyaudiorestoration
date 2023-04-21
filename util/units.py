import numpy as np


def sec_to_timestamp(t):
	m, s = divmod(t, 60)
	s, ms = divmod(s * 1000, 1000)
	h, m = divmod(m, 60)
	# return f"{h:.0f}:{m:.0f}:{s:.0f}:{ms:.0f} h:m:s:ms"
	return "%d:%02d:%02d:%03d h:m:s:ms" % (h, m, s, ms)


def t_2_m_s_ms(t):
	m, s = divmod(t, 60)
	s, ms = divmod(s * 1000, 1000)
	return "%02d:%02d:%03d" % (m, s, ms)


def to_dB(a):
	return 20 * np.log10(a)


def to_fac(a):
	return np.power(10, a / 20)


def normalize(_d, copy=False):
	# d is a (n x dimension) np array
	d = _d if not copy else np.copy(_d)
	m = np.max(np.abs(_d))
	# d -= np.min(d, axis=0)
	d /= m
	# print(m)
	return d


def to_mel(val):
	return np.log(val / 700 + 1) * 1127


def to_Hz(val):
	return (np.exp(val / 1127) - 1) * 700


A4 = 440
C0 = A4 * np.power(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch(freq):
	if freq > 0:
		h = round(12 * np.log2(freq / C0))
		octave = int(h // 12)
		n = int(h % 12)
		if -1 < octave < 10:
			return note_names[n] + str(octave)
	return "-"
