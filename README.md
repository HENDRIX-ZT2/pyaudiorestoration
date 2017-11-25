# pyrespeeder
Remove tape wow and flutter from audio recordings via their spectra. A simple solution for a mean problem.

-Installation-
You need to have installed:

python 3
numpy
librosa: https://librosa.github.io/
resampy: http://resampy.readthedocs.io/en/latest/
pysoundfile: https://pysoundfile.readthedocs.io/
scipy (some of the others probably depend on it)
tkinter

Best install these with pip (eg. pip install resampy).

Assuming you have python as an environment variable and are in the same folder as the script, start it with

python pyrespeeder_gui.py

Otherwise, specify the full paths to python.exe and the script.

-Usage-
Once started, a small GUI appears. Open a file and adjust the settings as you like.

1) identify the wow in the spectrum
Run the spectrogram and "trace" the sound with CTRL+leftclicks, point by point (linear interpolation!), then close the spectrogram. ALT+Click deletes a point, DELETE+Click deletes all. Your trace is written to a text file.

2) remove the wow
Then run the resampler. It will use your trace to respeed the track.

Channels:
-Spectrum uses always the first selected channel, resampling uses all selected channels
