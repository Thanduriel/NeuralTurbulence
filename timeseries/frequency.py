import numpy as np

# converts a complex 2d array into a 3d real array
def stackComplex(a):
	shape = (a.shape[0], a.shape[1], 1)
	re = np.reshape(a.real, shape)
	im = np.reshape(a.imag, shape)

	return np.concatenate((re,im), axis=2)

# inverse of stackComplex
def flattenComplex(a):
	re = a[...,0]
	im = 1j * a[...,1]

	return re + im

def decompose(scalarField, lowPass):
	freq = np.fft.fft2(scalarField)
	# center 
	freq = np.fft.fftshift(freq)
	begin = ((np.array(scalarField.shape) - lowPass) / 2).astype(int)
	end = (begin + lowPass).astype(int)
	lowFreq = freq[begin[0]:end[0],begin[1]:end[1]]

	return stackComplex(freq), stackComplex(lowFreq)

def invTransform(freqs):
	freqs = np.fft.ifftshift(freqs)

	return np.fft.ifft2(freqs)

def decomposeReal(scalarField, lowPass):
	freq = np.fft.rfftn(scalarField)
	freq = np.fft.fftshift(freq, axes=0)

	begin = (freq.shape[0] - lowPass[0]) // 2
	end = begin + lowPass[0]
	lowFreq = freq[begin:end,0:lowPass[1]]

	return stackComplex(freq), stackComplex(lowFreq)

def invTransformReal(freqs):
	flatFreqs = flattenComplex(freqs)
	flatFreqs = np.fft.ifftshift(flatFreqs,axes=0)
	return np.fft.irfftn(flatFreqs)

def shrink(freqs, lowPass):
	begin = (freqs.shape[0] - lowPass[0]) // 2
	end = begin + lowPass[0]

	redSize = freqs.shape[0]*freqs.shape[1] - lowPass[0] * lowPass[1]
	flatFreqs = np.zeros((redSize,2))
	ind = 0
	for y in range(freqs.shape[1]):
		for x in range(freqs.shape[0]):
			if x < begin or x >= end or y >= lowPass[1]:
				flatFreqs[ind,:] = freqs[x,y,:]
				ind += 1

	return flatFreqs

def composeReal(highFreqs, lowFreqs, highFreqShape):
	begin = (highFreqShape[0] - lowFreqs.shape[0]) // 2
	end = begin + lowFreqs.shape[0]

	freqs = np.zeros(highFreqShape)

	ind = 0
	for y in range(highFreqShape[1]):
		for x in range(highFreqShape[0]):
			if x < begin or x >= end or y >= lowFreqs.shape[1]:
				freqs[x,y,:] = highFreqs[ind,:]
				ind += 1

	freqs[begin:end,0:lowFreqs.shape[1]] = lowFreqs

	return freqs