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
1) You need to install a suitable version of [Python](https://www.python.org/downloads/) first.
   - Recommended Python version: 
     - 3.11 as of 2025-05-22
   - Use 64-bit python as 32-bit python doesn't work for all dependencies.
   - Make sure you check `Add Python to PATH` during installation.

2) Download `pyaudiorestoration`. To do so, click the `<> Code` button at the right, then `Download ZIP`. Unzip to a folder of your choice.

3) Install the required Python modules using `pip`. 
   - To do so, open a command prompt with admin rights inside the `pyaudiorestoration` folder you have unzipped.
   - Run `python -m pip install --upgrade pip` to make sure your `pip` is up to date.
   - Run `pip install -r requirements.txt` to install the dependencies.

4) If you have a supported GPU, install [pytorch](https://pytorch.org/get-started/locally/) with CUDA to massively speed up slow calculations by running them on the GPU. 

5) In some cases, you have to troubleshoot the installation of some dependencies. Here is a list of known issues:
   - `freetype-py` may have trouble to downloading `freetype.dll`. In that case, download it from [FreeType](https://www.freetype.org/download.html) and place it in a folder included in your system's path.


### How to Use
See the [wiki](https://github.com/HENDRIX-ZT2/pyaudiorestoration/wiki) for detailed instructions for the individual tools.
