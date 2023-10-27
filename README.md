
# EMPAD-G2 Raw Reader

Utility for reading raw data files produced by the prototype EMPAD-G2. 

## Installation 
Currently, to install you must download the source from git and install using pip. A simple way to do this (assuming you have git avaialble) is:
```bash
pip install git+https://github.com/sezelt/empad2
```


## Usage
Loading an EMPAD2 dataset requires two `.raw` files, one for the background, and one for the experiment, plus a number of calibration files. 

At the moment, the calibration files for the one existing sensor are packaged with the repo and do not need to be explicitly loaded. This may change in the future. 
~~First, load the calibration files by specifying the location:~~
```python
import empad2

empad2.load_calibrations("/path/to/folder/with/calibrations/")

# 
```
~~`empad2` attempts to detect the correct calibration files in the directory specified, but if the file names are too ambiguous then an error will be raised and you will have to speficy the files manually~~

Then, load the background data:
```python
background = empad2.load_background("/path/to/background/file.raw")
```

Then use the background to import experimental datasets:
```python
dataset = empad2.load_dataset("/path/to/experiment.raw", background)
```
If the scan region is square or if the `xml` metadata file is present, the shape is automatically detected. If the shape cannot be detected, specify it manually:
```python
dataset = empad2.load_dataset("/path/to/experiment.raw", background, shape=(256,256))
```




### License

GNU GPLv3

**empad2** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from **empad2** is also kept free and open.
