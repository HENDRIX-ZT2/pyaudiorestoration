# pyrespeeder
Remove tape wow and flutter from audio recordings via their spectra. A simple solution for a mean problem.

![Imgur](https://i.imgur.com/yUg6TTn.jpg)

### Installation
You need to have installed:
- python 3
- numpy
- [resampy](http://resampy.readthedocs.io/en/latest/)
- [pysoundfile](https://pysoundfile.readthedocs.io/)
- scipy (some of the others probably depend on it)
- [librosa](https://librosa.github.io/) (OLD GUI)
- tkinter (OLD GUI)
- pyQt5 (NEW GUI)
- [vispy](vispy.org) (NEW GUI)

Anaconda will have most of these installed by default. Best install the missing ones with pip (eg. pip install resampy).

Assuming you have python as an environment variable and are in the same folder as the script, start it with

`python pyrespeeder_gui.py`

Otherwise, specify the full paths to `python.exe` and the script.

### NEW Version - How to use
1) Load an audio file
2) Adjust the spectrogram settings, choose a tracing algorithm
3) In the spectrum, trace sounds that should be of constant pitch with CTRL+LMB/drag
4) In the speed chart, move the traces up or down with CTRL+LMB/drag so that overlapping pieces match up. Delete bad traces.
5) Save your traces so you can go back to them later.
6) Adjust the resampling settings to your liking (see OLD version notes below for more instructions)
7) Click resample.

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
- CTRL + LMB-Drag: In spectral view: runs the current tracing function in the dragged area; In speed view: offsets the currently selected speed curves by the drag

- RMB: Single line selection, deselects all previously selected lines.

- Shift + RMB: Multi line selection, click align again to deselect it.



### OLD Version Modes
#### Manual Tracing
Run the spectrogram and "trace" the sound with CTRL+leftclicks, point by point (linear interpolation!), then close the spectrogram. ALT+Click deletes a point, DELETE+Click deletes all. Your trace is written to a text file upon closing.

#### Resampling
- Blocks method:
Good for long-range, gradual wow. Fast, good frequency precision, little temporal precision.
- Expansion method [adapted from Feaster, P. (2017)]:
General purpose, but slower. Much faster and more accurate with dithering enabled.
- Sinc method [based on endolith (2011)]:
General purpose and best quality, but slowest. Ideal resampling, hence no overtones or distortion.

#### Automatic Tracing
- Trace Adaptive Center of Gravity [adapted from Czyzewski et al. (2007)]:
The trace starts in the given frequency band, which should be relatively narrow and not too low.


### Further Notes
- Manual and automatic tracing always uses the first selected channel, resampling uses all selected channels
- Resampling causes tiny clicks at the segment boundaries

### References
- Czyzewski et al. (2007). DSP Techniques for Determining "Wow" Distortion. Journal of the Audio Engineering Society. 55.
- endolith (2011). [Perfect Sinc Interpolation in Matlab and Python.](https://gist.github.com/endolith/1297227)
- Feaster, P. (2017). [The Wow Factor in Audio Restoration.](https://griffonagedotcom.wordpress.com/2017/02/16/the-wow-factor-in-audio-restoration/)
