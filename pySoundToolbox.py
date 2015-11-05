import numpy as np
import wave
import struct

def genSine(frequency=1.0, amplitude=1.0, phaseShift=0.0):
    """
    Generates a sine function.

    @param frequency frequency of the wave in Hz
    @param amplitude amplitude of the sine
    @param phaseShift phase shift of the wave
    @return A function with a parameter t. t is the time which needs to be a float
        or an numpy array of floats. The function returns a float, which is the value of the wave
        at time t or a numpy array with the value of the wave for all given values in t.
    """

    # -j is convetion to keep the vector in the complex plane moving clockwise, when a positive frequency is given
    return lambda t: amplitude * np.exp(-1j * 2 * np.pi * frequency * t + phaseShift).imag

def genArrayResponseFunc(angle, antennaPositions=np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]]), **kwargs):
    """
    Generates an array response function.

    @param angle angle of source
    @param antennaPositions positions of the antennas to an abitrary origin as numpy array
    @param frequeny @see generateSine
    @param amplitude @see generateSine
    @return array response function with parameter t. t is numpy array, a row represents the response of one microphone
    """

    # Antenna positions in m
    speedOfSound = 343.2 # m/s
    doa = np.array([np.cos(angle), np.sin(angle)]) # Normalized direction of arrival
    # Project antenna position orthogonally onto doa to get the desired time delay
    sineFunctions = []
    delays = np.zeros(antennaPositions.shape[1])
    for i in range(antennaPositions.shape[1]):
        delays[i] = antennaPositions[:, i].dot(doa) / speedOfSound
        sineFunctions.append(genSine(**kwargs))

    def func(t):
        """
        Array response function.

        @param t time, which needs to be a float or a numpy array with times
        @return a numpy array with the elongation of the wave at times t.
            A row represents one microphone within the array.
        """
        if type(t) != np.ndarray:
            t = np.array([t])

        data = np.zeros((antennaPositions.shape[1], t.shape[0]))
        for i in range(antennaPositions.shape[1]):
            data[i, :] = sineFunctions[i](t + delays[i])

        return data

    return func

def sample24PCM(arrayResponseFunction, amplitudeScale=1.0, samplingTime=1.0, samplingRate=16000):
    """
    Samples 24 PCM data from an array response function

    @param arrayResponseFunction array response function provided by genArrayResponseFunc
    @param amplitudeScale scale of the amplitude relative to 2^24-1 (just sets the pcm value of the
        amplitude)
    @param samplingTime time interval of sampling in seconds
    @param samplingRate sampling rate in Hz
    @return numpy int32 array with 24 PCM data. Every row represents one channel.
    """

    assert(amplitudeScale > 0 and amplitudeScale <= 1 and samplingTime > 0 and samplingRate > 0 and type(samplingRate) in (np.uint32, np.uint64, np.uint16, int, np.uint))

    t = np.arange(0, samplingTime * samplingRate, dtype=np.float64) / samplingRate # Timesteps
    arrayResponse = arrayResponseFunction(t)
    # Scaling to generate 24 PCM data
    arrayResponse /= arrayResponse.max()
    return np.int32(arrayResponse * ((2**23) - 1) * amplitudeScale)

def save24PCM(fileName, data, samplingRate=16000):
    """
    Saves a 24 PCM wav file.

    @param fileName name of the file
    @param data already sampled data as numpy uint32 array. The cols represent the time and the rows
        are the channels.
    @param samplingRate sampling rate in Hz
    """

    with wave.open(fileName, 'w') as wav:
        wav.setframerate(samplingRate)
        wav.setsampwidth(3)
        wav.setnchannels(data.shape[0])
        for i in range(data.shape[1]):
            for k in range(data.shape[0]):
                sample = data[k, i]
                # Revert byte order, because wav is little endian
                frame = bytes([sample & 0xff, (sample >> 8) & 0xff, (sample >> 16) & 0xff])
                wav.writeframes(frame)
        wav.close()

def read24PCM(fileName):
    """
    Reads a 24 PCM wav file.

    @param fileName name of the file
    @return tupel (data, samplingRate):
        data is a numpy int32 array with the read data. A row represents one channel.
        samplingRate is the provided sampling rate.
    """

    with wave.open(fileName, 'r') as wav:
        samplingRate = wav.getframerate()
        data = np.zeros((wav.getnchannels(), wav.getnframes()), dtype=np.int32)
        for i in range(data.shape[1]):
            frame = wav.readframes(1)
            for k in range(data.shape[0]):
                # Seems pretty ugly and only works in python, but I don't know a better way to keep the Two's complement
                data[k, i] = struct.unpack('<i', bytes([0]) + frame[k*3:k*3+3])[0] >> 8
        wav.close()

    return (data, samplingRate)
