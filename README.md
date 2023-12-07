
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
First, load the calibration files for your sensor (`"cryo-titan"` or `"andromeda"` are included with the package):
```python
import empad2

calibrations = empad2.load_calibrations("andromeda")

# 
```

Then, load the background data:
```python
background = empad2.load_background("/path/to/background/file.raw", calibration_data=calibrations)
```

Then use the background to import experimental datasets:
```python
dataset = empad2.load_dataset("/path/to/experiment.raw", background, calibrations)
```
If the scan region is square, the shape is automatically detected. If the shape cannot be detected, specify it manually:
```python
dataset = empad2.load_dataset("/path/to/experiment.raw", background, calibrations, shape=(256,256))
```




### License

GNU GPLv3

**empad2** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from **empad2** is also kept free and open.
