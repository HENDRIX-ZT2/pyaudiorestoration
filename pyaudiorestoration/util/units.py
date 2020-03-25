import numpy as np

def sec_to_timestamp(t):
	m, s = divmod(t, 60)
	s, ms = divmod(s*1000, 1000)
	h, m = divmod(m, 60)
	return "%d:%02d:%02d:%03d h:m:s:ms" % (h, m, s, ms)
	
def t_2_m_s_ms(t):
	m, s = divmod(t, 60)
	s, ms = divmod(s*1000, 1000)
	return "%02d:%02d:%03d" % (m, s, ms)
	
def to_dB(a):
	return 20 * np.log10(a)
	
def to_fac(a):
	return np.power(10, a/20)

def normalize(_d, copy=False):
	# d is a (n x dimension) np array
	d = _d if not copy else np.copy(_d)
	m = np.max(np.abs(_d))
	# d -= np.min(d, axis=0)
	d /= m
	print(m)
	return d