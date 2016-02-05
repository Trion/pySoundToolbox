import numpy as np
from scipy.stats import logistic
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

def genArrayResponseFunc(angles, antennaPositions=np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]]), frequencies=1, amplitudes=1, phaseShifts=0, noiseStd=0):
    """
    Generates an array response function.

    @param angles 1-D numpy array with angles of the sources in rad
    @param antennaPositions positions of the antennas to an abitrary origin as numpy array
    @param frequenies numpy array or float with the frequencies of the sound sources in Hz
    @param amplitudes numpy array or float with the amplitude of the sound source signals
    @param phaseShifts numpy array or float with the phaseShifts of the sound source sinals
    @param noiseStd standard deviation of the white noise present (0 means no noise)
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
    steeringMat = np.matrix(np.empty((antennaNum, sourcesNum), dtype=np.complex256))
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

        data = np.matrix(np.empty((antennaNum, t.shape[0]), dtype=np.complex))
        for i in range(t.shape[0]):
            sourceData = np.matrix([[sourceSignals[k](t[i])] for k in range(sourcesNum)], dtype=np.complex256)
            noise = 0
            if noiseStd != 0:
                noise = np.matrix(np.random.normal(size=antennaNum, scale=noiseStd), dtype=np.complex256).transpose()\
                    + 1j * np.matrix(np.random.normal(size=antennaNum, scale=noiseStd), dtype=np.complex256).transpose()
            data[:, i] = steeringMat * sourceData + noise

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

    # Determine number of channels
    if len(data.shape) == 1:
        data = np.matrix(data)

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
    spectrum = np.fft.fft(data.flat)
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


class HarmonicSource:
    """
    Entity for sound sources with several harmonics
    """

    def __init__(self, baseFreq, harmonicsNum=0, amplitudes=1):
        """
        constructor

        @param baseFreq first frequency (lowest frequency)
        @param hamonicsNum amount of harmonics, withoud base frequency
        @param amplitudes amplitudes of the waves as scalar or numpy array. If it is a scalar all
            harmonics including the base frequency will have the same amplitude. If it is a array
            its size must be (harmonicsNum+1)
        """

        if np.isscalar(amplitudes):
            amplitudes = np.ones(harmonicsNum + 1) * amplitudes

        if amplitudes.shape[0] != harmonicsNum + 1:
            raise ValueError("Shape of amplitudes must be (harmonicsNum + 1,) (current shape is ({:d},) but ({:d},) is required)!".format(amplitudes.shape[0], harmonicsNum + 1))

        self.samplingRate = -1

        self.sines = []
        for i, amplitude in enumerate(amplitudes):
            self.sines.append(genSine(frequency=(i+1)*baseFreq, amplitude=amplitude))

    def getSample(self, m):
        """
        Returns the mth sample of the source or 0.0 if m is negative. This is needed, because depending on the angle of arrival of the source
        the MicrophoneArray instance have to access samples by "negative" time (i.e. the sound did not arrive at the given time).

        @param m disrete time
        @return the mth sample of the source of 0.0 if m is negative
        """

        if m < 0:
            return 0

        sample = 0
        for sine in self.sines:
            sample += sine(m / self.samplingRate).real

        return sample

    def get(self, m):
        """
        Returns a numpy array of the values of the samples identified by the values of m or 0.0 if m is negative.

        @param m disrete time as iteratable or integer
        @return the mth sample of the source of 0.0 if m is negative
        """

        if np.isscalar(m):
            m = [m]

        mTime = np.asarray(m)

        if len(mTime.shape) > 1:
            ValueError('m must be a one dimensional array!')

        if mTime.dtype not in (int, np.int32, np.int64, np.int8, np.int16):
            raise ValueError('Values of m must be integers!')

        samples = np.zeros(mTime.size)
        for sine in self.sines:
            samples += sine(mTime / self.samplingRate).real

        return samples

    def onAdd(self, array):
        """
        Event handler that will be executed when this source is attached to an array.

        @param array MicrophoneArray object
        """
        self.samplingRate = array.samplingRate


class BroadbandSource:
    """
    entity for sochastic broadband sources
    """

    def __init__(self, sourceType='white', transformation=lambda x: x, sampleGain=1000):
        """
        constructor

        @param sourceType type of the emitted signal
            white: white noise
            logistic: temporal logistic distributed source
            const: constantly zero (the transformation parameter can add a constant component)
        @param transformation a function that performes transformationen of the output signal (e.g. a bandpass filter)
        @param sampleGain amount of samples that will be generated at initilization and everytime time exceeds the number of already sampled data
        """

        assert sampleGain > 0

        if sourceType not in ('white', 'const', 'logistic'):
            raise ValueError("Invalid sourceType!")

        self.sourceType = sourceType
        self.transformation = transformation
        self.sampleGain = sampleGain
        self.data = np.array([], dtype=np.float)
        self.transformedData = np.array([], dtype=np.float)

        self.generateSamples()

    def generateSamples(self):

        if self.sourceType == 'white':
            self.data = np.append(self.data, np.random.normal(size=self.sampleGain))
        elif self.sourceType == 'const':
            self.data = np.append(self.data, np.zeros(self.sampleGain))
        elif self.sourceType == 'logistic':
            self.data = np.append(self.data, logistic.rvs(size=self.sampleGain))
        self.transformedData = self.transformation(self.data)

    def getSample(self, m):
        """
        Returns the mth sample of the source or 0.0 if m is negative. This is needed, because depending on the angle of arrival of the source
        the MicrophoneArray instance have to access samples by "negative" time (i.e. the sound did not arrive at the given time).

        @param m disrete time
        @return the mth sample of the source of 0.0 if m is negative
        """

        if m < 0:
            return 0

        if m >= self.data.size:
            self.generateSamples()

        return self.transformedData[m]

    def get(self, m):
        """
        Returns a numpy array of the values of the samples identified by the values of m or 0.0 if m is negative.

        @param m disrete time as iteratable or integer
        @return the mth sample of the source of 0.0 if m is negative
        """

        if np.isscalar(m):
            m = [m]

        mTime = np.asarray(m)

        if len(mTime.shape) > 1:
            ValueError('m must be a one dimensional array!')

        if mTime.dtype not in (int, np.int32, np.int64, np.int8, np.int16):
            raise ValueError('Values of m must be integers!')

        samples = np.empty(mTime.size)
        for i in range(mTime.size):
            samples[i] = self.getSample(mTime[i])

        return samples

    def onAdd(self, array):
        """
        Event handler that will be executed when this source is attached to an array.

        @param array MicrophoneArray object
        """
        pass


class FileSource:
    """
    generic source defined by a file
    """

    def __init__(self, fileName):
        """
        constructor

        @param fileName path to uncompressed wav file, which contains the samples for this source
        """

        samples, self.fileSamplingRate = read24PCM(fileName)
        # Only use the first channel and normalize
        self.samples = samples[0, :] / samples[0, :].max()

    def getSample(self, m):
        """
        Returns the mth sample of the source or 0.0 if m is negative. This is needed, because depending on the angle of arrival of the source
        the MicrophoneArray instance have to access samples by "negative" time (i.e. the sound did not arrive at the given time).

        @param m disrete time
        @return the mth sample of the source of 0.0 if m is negative
        """

        if m < 0 or m >= self.samples.size:
            return 0

        return self.samples[m]

    def get(self, m):
        """
        Returns a numpy array of the values of the samples identified by the values of m or 0.0 if m is negative.

        @param m disrete time as iteratable or integer
        @return the mth sample of the source of 0.0 if m is negative
        """

        if np.isscalar(m):
            m = [m]

        mTime = np.asarray(m)

        if len(mTime.shape) > 1:
            ValueError('m must be a one dimensional array!')

        if mTime.dtype not in (int, np.int32, np.int64, np.int8, np.int16):
            raise ValueError('Values of m must be integers!')

        samples = np.empty(mTime.size)
        for i, m in enumerate(mTime):
            samples[i] = self.getSample(m)

        return samples

    def onAdd(self, array):
        """
        Event handler that will be executed when this source is attached to an array.

        @param array MicrophoneArray object
        """

        if array.samplingRate != self.fileSamplingRate:
            raise ValueError("Array sampling rate and file sampling rate must be the same!")
        # TODO maybe interpolate samples if sampling rates are not compatible


class MicrophoneArray:
    """
    Virtual microphone array class
    """

    def __init__(self, antennaPositions=np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]]), samplingRate=16000, noiseStds=0.0):
        """
        constructor

        @param antennaPositions positions of the antennas to an abitrary origin as numpy array. The positions must be given
            by 2D coordinates. (Example: numpy.array([[x_1, x_2, x_3], [y_1, y_2, y_3]]))
        @param samplingRate sampling rate of the microphones in Hz
        @param noiseStds standard deviation of the white noise present (0 means no noise). If noise is a scalar all microphones
            got the same noise level. If it is an numpy array it has to be the same size like antennaPositions in dimension 1.
        """

        # Only accept "point-sequence-like" arrays
        if len(antennaPositions.shape) != 2 or (antennaPositions.shape[0] != 2 and antennaPositions.shape[1] != 2):
            raise ValueError('antennaPositions dimensions does not fit requirements!')

        # Set consistent shape, so I do not have to do it every time I need this array
        if antennaPositions.shape[0] == 2:
            self.antennaPositions = antennaPositions
        else:
            self.antennaPositions = antennaPositions.transpose()

        # Set consistens shape, so I do not have to do it every time I need this array
        if np.isscalar(noiseStds):
            if not np.isreal(noiseStds):
                raise ValueError('noiseStds must be a real value!')
            self.noiseStds = np.empty(self.antennaPositions.shape[1])
            self.noiseStds[:] = noiseStds
        else:
            if noiseStds.shape != (self.antennaPositions.shape[1],):
                raise ValueError('noiseStds must have the shape ({:d},)'.format(self.antennaPositions.shape[1]))
            self.noiseStds = noiseStds

        if type(samplingRate) != int:
            raise ValueError('samplingRate must be of type int!')
        self.samplingRate = samplingRate

        # Sources list with (DOA, BroadbandSource instance)
        self.sources = []

    def addSource(self, angle, source):
        """
        Adds a source that will be "recorded" by the microphone array.

        @param angle angle of arrival in rad of the sound emitted by the source. The angle is relative to the x-axis of
            the coordinate plane.
        @param source a source object that has a get and an onAdd method
        """

        if not np.isscalar(angle):
            raise ValueError('Only scalar types are allowed for angle!')
        if not np.isreal(angle):
            raise ValueError('angle must be a real value!')

        # Generate direction of arrival (DOA) from angle of arrival
        doa = np.array([np.cos(angle), np.sin(angle)])

        source.onAdd(self)
        self.sources.append((doa, source))

    def get(self, m):
        """
        Returns array response at time m (the mth sample). As reference the origin of the coordinate
        coordinate system (implicitly defined by the microphone positions) is used, i.e. get(0) returns
        is the first recording of an microphone positioned at (0, 0). Negative values for m are allowed,
        so you can see the propagations from on microphone to another.

        @param m dicrete time as integer or iteratable of integers
        @return numpy array with the microphone array response data. The first dimensions represents
            the microphones, the second dimension the time (e.g. [2, 5] is the fifth sample of the
            third microphone).
        """

        if np.isscalar(m):
            m = [m]

        mTime = np.asarray(m)

        if mTime.dtype not in (int, np.int32, np.int64, np.int8, np.int16):
            raise ValueError('Values of m must be integers!')

        if len(mTime.shape) > 1:
            ValueError('m must be a one dimensional array!')

        speedOfSound = 343.2 # m/s
        antennaNum = self.antennaPositions.shape[1]

        response = np.zeros((self.antennaPositions.shape[1], mTime.size))

        # Compute noiseless response
        for doa, source in self.sources:
            for k in range(antennaNum):
                # Delay in samples
                delay = int((self.antennaPositions[:, k].dot(doa) / speedOfSound) * self.samplingRate)
                response[k, :] += source.get(mTime + delay)

        # Add noise
        for k in range(antennaNum):
            if self.noiseStds[k] != 0:
                response[k] += np.random.normal(scale=self.noiseStds[k], size=mTime.size)

        return response
