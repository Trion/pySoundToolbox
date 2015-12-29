# pySoundToolbox
A small library for generating and saving sinusoidal sounds, especially for microphone arrays.
I am implementing this library to test sound localization algorithms for my master thesis.

## Requirements
* [numpy](https://github.com/numpy/numpy)

## Simple usage

```
from pySoundToolbox import *
import numpy as np

# Sinusoidal generation function for a source with 300Hz
sourceSine = genSine(frequency=300)
print(sourceSine(np.arange(0, 0.2, 0.01)))

# Get array response function with a source at 45° (the microphone positions are the ones of
# the kinect)
arrayResponse = genArrayResponseFunc(45 * np.pi / 180)
print(arrayResponse(np.arange(0, 0.2, 0.01)))

# Sample array response at 16kHz for one second
data = sample24PCM(arrayResponse)

# Save sampled array response as uncompressed 24 bit wav
# Every channel represents the response of the microphone
save24PCM('test.wav', data)

# Read saved file
loadedData = read24PCM('test.wav')
# loadedData == data
```
