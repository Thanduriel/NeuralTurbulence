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