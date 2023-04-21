# Python Audio Restoration Suite
A set of tools to restore audio quality from a variety of old analog sources, such as tape, cassettes, acetates and vinyl.

### Features
- Wow & Flutter Removal
- Speed matching to hum frequency
- EQ matching with differential EQ
- Sub-sample accurate Spectral Temporal Alignment
- Automatic Dropout Restoration
- Spectral Expander / Decompressor

### Installation
1) You need to install a suitable version of [Python](https://www.python.org/downloads/) first. Make sure you check `Add Python to PATH` during installation.
   - at least 3.7+
   - tested on 3.79
   - 3.11 does not support all dependencies as of 2023-04-21; 3.10 seems to support them

2) Download `pyaudiorestoration`. To do so, click the `Clone or Download` button at the right, then `Download ZIP`. Unzip to a folder of your choice.

3) Install the required Python modules using `pip`. To do so, open a command prompt with admin rights inside the `pyaudiorestoration` folder you have unzipped and run: `pip install -r requirements.txt` If you get a message that tells you to update pip, do so.

4) In some cases, you have to troubleshoot the installation of some dependencies. Here is a list of known issues:
   - `freetype-py` may have trouble to downloading `freetype.dll`. In that case, download it from [FreeType](https://www.freetype.org/download.html) and place it in a folder included in your system's path.


### How to Use
See the [wiki](https://github.com/HENDRIX-ZT2/pyaudiorestoration/wiki) for detailed instructions for the individual tools.
