import numpy as np
import resampy
from matplotlib import pyplot as plt

from util import io_ops
from util.resampling import resample_expand

# file_src = "../samples/flutter.flac"
file_src = "../samples/dehradun.mp3"

fft_size = 8192 * 2
fft_hop = fft_size // 128

signal, sr, num_channels = io_ops.read_file(file_src)
targ = 500
modulation = 50.0
t = np.arange(len(signal)) / sr
speed = np.sin(2 * np.pi * modulation * t) * 10.0 + targ
speed = targ / speed
# plt.plot(speed, label=f"speed")
# plt.legend()
# plt.show()
res = resample_expand(signal, speed, 100.0)

io_ops.write_file(file_src, res, sr, num_channels, "out")