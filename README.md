# pyrespeeder
Remove tape wow and flutter from audio recordings via their spectra. A simple solution for a mean problem.

![Imgur](https://i.imgur.com/yUg6TTn.jpg)

### Installation
You need to have installed:
- python 3.6
- numpy
- [resampy](http://resampy.readthedocs.io/en/latest/)
- [pysoundfile](https://pysoundfile.readthedocs.io/)
- scipy
- [librosa](https://librosa.github.io/) (OLD GUI)
- tkinter (OLD GUI)
- pyQt5 (NEW GUI)
- [vispy](vispy.org) (NEW GUI)
- [pyFFTW](https://github.com/pyFFTW/pyFFTW) (NEW GUI - optional speedup)

The bundled python platform [Anaconda](https://www.anaconda.com/download/) will have most of these dependencies installed by default. Best install the missing ones with `pip` (eg. in the Windows commandline, type `pip install vispy`).

Click the `Clone or Download` button at the right, then `Download ZIP`. Unzip to a folder of your choice. Assuming you have python configured as an environment variable and your commandline is in the same folder as the script, start it with

`python pyrespeeder_gui.py`

Otherwise, specify the full paths to `python.exe` and the script.

### NEW Version - How to use
1) Load an audio file
2) Adjust the spectrogram settings, choose a tracing algorithm
3) In the spectrum, trace sounds that should be of constant pitch with CTRL+LMB/drag
4) In the speed chart, move the traces up or down with CTRL+LMB/drag so that overlapping pieces match up. Delete bad traces.
5) If you are dealing with cyclic wow, you can use sine regression to get more accurate results quickly. Only sample parts of the file (like start and end) with the tracing methods, then select your suspected source medium and perform sine regression over your sampled areas.
6) Save your traces so you can go back to them later.
7) Adjust the resampling settings to your liking (see notes below for more instructions)
8) Click resample.

Example of cyclic wow removal:
![Imgur](https://i.imgur.com/tc3RDyo.gif)

### Hotkeys

#### Navigation
- LMB-Drag: Move the spectral and speed view.

- Scroll: Zoom in time only.

- Shift + Scroll: Zoom in frequency only.

- CTRL + Scroll: Zoom in time and frequency.

- You can also scroll (but currently not drag) the axes directly.

#### Functions
- CTRL + LMB-Drag: In spectral view: runs the current tracing function in the dragged area; In speed view: offsets the currently selected speed curves and changes the amplification of selected sine regression curves.

- RMB: Single line selection, deselects all previously selected lines.

- Shift + RMB: Multi line selection, click align again to deselect it.

### Tracing Modes
- Adaptive Center of Gravity [adapted from Czyzewski et al. (2007)]: The trace starts in the given frequency band, which should be relatively narrow and not too low.
- Correlation: Tries to correlate the whole spectrum sequentially. Only works well for big FFT sizes.
- Sine Draw: Not really useful
- Sine Regression: Sample the master speed curve in some points to get the parameters of cyclic wow in these areas, which are then used to extrapolate the sine over the whole duration.

### Resampling Modes
- Linear: Fastest, but causes some overtone artifacts.
- Expansion: Excellent precision in time and frequency, but induces clicks. Repeats each input sample (N * speed + dithering) times and then downsamples everything by N. [based on Feaster, P. (2017)]
- Sinc: Slowest, but most accurate. No clicks. Accurate because of digital sampling theory. [based on endolith (2011) and Hope (2015)]
- Blocks: Fast, but inaccurate in time, induces clicks. Resamples segements according to their mean speed.
- Note that Linear & Sinc resampling currently require a relatively sparsely sampled speed curve (like FFT size = 4096, Overlap = 1) to work accurately. This may be fixed in the future.

### References
- Czyzewski et al. (2007). DSP Techniques for Determining "Wow" Distortion. Journal of the Audio Engineering Society. 55.
- endolith (2011). [Perfect Sinc Interpolation in Matlab and Python.](https://gist.github.com/endolith/1297227)
- Feaster, P. (2017). [The Wow Factor in Audio Restoration.](https://griffonagedotcom.wordpress.com/2017/02/16/the-wow-factor-in-audio-restoration/)
- Hope, G. (2015). [Sinc Interpolation of Signal using Numpy / Python.](https://gist.github.com/gauteh/8dea955ddb1ed009b48e)
