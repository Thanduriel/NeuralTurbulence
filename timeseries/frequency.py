import numpy as np

# converts a complex 2d array into a 3d real array
def stackComplex(a):
	re = np.reshape(a.real, a.shape + (1,))
	im = np.reshape(a.imag, a.shape + (1,))

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

# extract symmetric parts from which a real signal can be fully constructed
def extractSymmetric(a):
	return a[:, a.shape[1] // 2 - 1:]

def extendSymmetric(a):
	s = a.shape
	res = np.zeros((s[0], (s[1]-1)*2), dtype=np.complex)
	end = res.shape[1]-s[1]
	res[:, end:] = a
	res[:,0:end] = a[::-1,0:end]#.real -1j * a[::-1,::-1].imag
	return res