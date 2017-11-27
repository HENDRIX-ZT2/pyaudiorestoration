# pyrespeeder
Remove tape wow and flutter from audio recordings via their spectra. A simple solution for a mean problem.

-Installation-
You need to have installed:
- python 3
- numpy
- librosa: https://librosa.github.io/
- resampy: http://resampy.readthedocs.io/en/latest/
- pysoundfile: https://pysoundfile.readthedocs.io/
- scipy (some of the others probably depend on it)
- tkinter

Best install these with pip (eg. pip install resampy).

Assuming you have python as an environment variable and are in the same folder as the script, start it with

python pyrespeeder_gui.py

Otherwise, specify the full paths to python.exe and the script.



-Modes-
1) Manual Tracing:
Run the spectrogram and "trace" the sound with CTRL+leftclicks, point by point (linear interpolation!), then close the spectrogram. ALT+Click deletes a point, DELETE+Click deletes all. Your trace is written to a text file upon closing.

2) Resampling:
- Blocks method:
Good for long-range, gradual wow. Fast, good frequency precision, little temporal precision.
- Expansion method [adapted from Feaster, P. (2017)]:
General purpose, but slower. Much faster and more accurate with dithering enabled.
- Sinc method [based on endolith (2011)]:
General purpose and best quality, but slowest. Ideal resampling, hence no overtones or distortion.

3) Automatic Tracing:
a) Trace Adaptive Center of Gravity [adapted from Czyzewski et al. (2007)]:
The trace starts in the given frequency band, which should be relatively narrow and not too low.


-Notes-
- Manual and automatic tracing always uses the first selected channel, resampling uses all selected channels
- Resampling causes tiny clicks at the segment boundaries

References:
- Czyzewski et al. (2007). DSP Techniques for Determining "Wow" Distortion. Journal of the Audio Engineering Society. 55.
- endolith (2011). Perfect Sinc Interpolation in Matlab and Python.
- Feaster, P. (2017). The Wow Factor in Audio Restoration. [https://griffonagedotcom.wordpress.com/2017/02/16/the-wow-factor-in-audio-restoration/]
