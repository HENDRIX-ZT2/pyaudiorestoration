from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='pyaudiorestoration',
   version='1.0',
   description='A set of tools to restore audio quality from a variety of old analog sources',
   license="GPL-2.0",
   long_description=long_description,
   author='Hendrix',
   author_email='',
   url="https://github.com/HENDRIX-ZT2/pyaudiorestoration",
   packages=['pyaudiorestoration'],  # same as name
   install_requires=['pyqt5', 'librosa', 'resampy', 'pysoundfile', 'matplotlib', 'scipy', 'numpy', 'vispy', 'pyfftw', 'numba'], # external packages as dependencies
   # scripts=[
   #          'scripts/cool',
   #          'scripts/skype',
   #         ]
)