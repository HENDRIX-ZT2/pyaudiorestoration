# Python Audio Restoration Suite
A set of tools to restore audio quality from a variety of old analog sources, such as tape, cassettes, acetates and vinyl.

### Features
- Wow & Flutter Removal
- EQ matching with differential EQ
- Spectral Alignment

### Installation
You need to have installed:
- python 3.6.6
- numpy
- numba
- [pysoundfile](https://pysoundfile.readthedocs.io/)
- scipy
- pyQt5
- [vispy](vispy.org)
- [pyFFTW](https://github.com/pyFFTW/pyFFTW) (_optional_ speedup) Note for python 3.6 users on windows - to install pyFFTW, download the correct .whl file from [ ![Download](https://api.bintray.com/packages/hgomersall/generic/PyFFTW-development-builds/images/download.svg) ](https://bintray.com/hgomersall/generic/PyFFTW-development-builds/_latestVersion#files)  (scroll down!) and install it with `pip install PATH_TO_FILE.whl`.

The bundled python platform [Anaconda](https://www.anaconda.com/download/) will have most of these dependencies installed by default. Best install the missing ones with `pip` (eg. in the Windows commandline, type `pip install vispy`).

Click the `Clone or Download` button at the right, then `Download ZIP`. Unzip to a folder of your choice. Assuming you have python configured as an environment variable and your commandline is in the same folder as the script, start it with

`python pyrespeeder_gui.py` or `python pytapesynch_gui.py` or `python difeq.py`

Otherwise, specify the full paths to `python.exe` and the script.

### How to Use
See the [wiki](https://github.com/HENDRIX-ZT2/pyaudiorestoration/wiki) for detailed instructions for the individual tools.
