# Python Audio Restoration Suite
A set of tools to restore audio quality from a variety of old analog sources, such as tape, cassettes, acetates and vinyl.

### Features
- Wow & Flutter Removal
- Speed matching to hum frequency
- EQ matching with differential EQ
- Spectral Temporal Alignment
- Automatic Dropout Restoration
- Spectral Expander / Decompressor

### Installation
You need to have installed:
- python 3.6+
- [FreeType](https://www.freetype.org/download.html)
On Windows, make sure that freetype.dll is located in a folder in the system path.
- numpy
- numba
- [resampy](https://resampy.readthedocs.io/)
- [pysoundfile](https://pysoundfile.readthedocs.io/)
- scipy
- pyQt5
- [vispy](vispy.org)
- [pyFFTW](https://github.com/pyFFTW/pyFFTW) (_optional_ speedup) Note for python 3.6 users on windows - to install pyFFTW, download the correct .whl file from [ ![Download](https://api.bintray.com/packages/hgomersall/generic/PyFFTW-development-builds/images/download.svg) ](https://bintray.com/hgomersall/generic/PyFFTW-development-builds/_latestVersion#files)  (scroll down!) and install it with `pip install PATH_TO_FILE.whl`.

Install the missing dependencies with python's package manager, `pip` (eg. in the Windows commandline, type `pip install vispy`).

Click the `Clone or Download` button at the right, then `Download ZIP`. Unzip to a folder of your choice. Assuming you have configured python path as an environment variable during installation, you can run the programs by double clicking their `.bat` files.

These are just shortcuts that are equivalent to typing the following in the command line:

`python pyrespeeder_gui.py` or `python pytapesynch_gui.py` or `python difeq_gui.py` etc...

Otherwise, specify the full paths to `python.exe` and the script.


### How to Use
See the [wiki](https://github.com/HENDRIX-ZT2/pyaudiorestoration/wiki) for detailed instructions for the individual tools.
