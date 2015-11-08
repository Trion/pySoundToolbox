import numpy as np
import wave
import struct

def genSine(frequency=1.0, amplitude=1.0, phaseShift=0.0):
    """
    Generates complex signal.

    @param frequency frequency of the wave in Hz
    @param amplitude amplitude of the sine
    @param phaseShift phase shift of the wave
    @return A function with a parameter t. t is the time which needs to be a float
        or an numpy array of floats. The function returns a float, which is the value of the wave
        at time t or a numpy array with the value of the wave for all given values in t.
    """

    # -j is convetion to keep the vector in the complex plane moving clockwise, when a positive frequency is given
    return lambda t: amplitude * np.exp(-1j * 2 * np.pi * frequency * t + phaseShift)

def genArrayResponseFunc(angles, antennaPositions=np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]]), frequencies=1, amplitudes=1, phaseShifts=0):
    """
    Generates an array response function.

    @param angles numpy array with angles of the sources in rad
    @param antennaPositions positions of the antennas to an abitrary origin as numpy array
    @param frequenies numpy array or float with the frequencies of the sound sources in Hz
    @param amplitudes numpy array or float with the amplitude of the sound source signals
    @param phaseShifts numpy array or float with the phaseShifts of the sound source sinals
    @return array response function with parameter t. t is numpy array, a row represents the response of one microphone
    """

    # Amount of antennas
    antennaNum = antennaPositions.shape[1]

    # Convert signal parameters to array, to keep signal generation easy

    if np.isscalar(angles):
        angles = np.array([angles], dtype=np.float128)
    # Amount of sources
    sourcesNum = angles.shape[0]

    if np.isscalar(frequencies):
        frequencies = np.ones(sourcesNum, dtype=np.float128) * frequencies
    if np.isscalar(phaseShifts):
        phaseShifts = np.ones(sourcesNum, dtype=np.float128) * phaseShifts
    if np.isscalar(amplitudes):
        amplitudes = np.ones(sourcesNum, dtype=np.float128) * amplitudes

    # Generate source signals
    sourceSignals = []
    for i in range(angles.shape[0]):
        sourceSignals.append(genSine(frequency=frequencies[i], amplitude=amplitudes[i], phaseShift=phaseShifts[i]))

    # Create steering matrix
    # A row is the mapping of all sources to one microphone
    # A column is the mapping of one source to all microphones
    speedOfSound = 343.2 # m/s
    steeringMat = np.matrix(np.empty((antennaNum, 1), dtype=np.complex256)) # TODO extend to multiple sound sources
    for i in range(sourcesNum):
        doa = np.array([np.cos(angles[i]), np.sin(angles[i])]) # Normalized direction of arrival
        for k in range(antennaNum):
            delay = antennaPositions[:, k].dot(doa) / speedOfSound
            steeringMat[k, i] = np.exp(-1j * 2 * np.pi * frequencies[i] * delay)

    def func(t):
        """
        Array response function.

        @param t time, which needs to be a float or a numpy array with times
        @return a numpy array with the elongation of the wave at times t.
            A row represents one microphone within the array.
        """
        if type(t) != np.ndarray:
            t = np.array([t])

        data = np.matrix(np.empty((antennaNum, t.shape[0]), dtype=np.complex256))
        for i in range(t.shape[0]):
            sourceData = np.matrix([[sourceSignals[k](t[i])] for k in range(sourcesNum)], dtype=np.complex256)
            data[:, i] = steeringMat * sourceData

        return np.array(data, dtype=data.dtype)

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
    # Scaling amplitude to generate 24 PCM data
    arrayResponse /= np.abs(arrayResponse).max()
    return np.int32(arrayResponse.real * ((2**23) - 1) * amplitudeScale)

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

def genAnalyticSignal(data):
    """
    Generates an analytic signal to given real data (converts real signal into a complex one).
    I'm using "Computing the discrete-time analytic signal via fft" by S. Lawrence Marple.

    @param data real data as 24 PCM numpy array. A row represents the response of one microphone.
    @return the complex response of the microphone array. A row represents the response of one microphone.
    """

    # Step 1: Compute FFT
    spectrum = np.fft.fft(data)
    # Step 2: Half the spectrum
    n = spectrum.shape[0]
    h = np.empty(n, dtype=np.complex256)
    h[0] = spectrum[0]
    h[n / 2] = spectrum[n / 2]
    h[1:n / 2] = 2 * spectrum[1:n / 2]
    h[n / 2 + 1:] = 0
    # Step 3: Apply IFFT
    analyticSignal = np.fft.ifft(h).conjugate() # Need to conjugate to preserve the convention, that the vector should run clockwise for positive frequencies

    return analyticSignal
