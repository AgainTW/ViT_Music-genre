from scipy.io.wavfile import read, write
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
audio process
"""
# read audio
def read_audio(loc):
	fs, w = read(loc)
	return fs, w

# Generate Spectrogram
def spectrogram(w, rate):
	N = 256
	S = []
	for k in range(0, w.shape[0]+1, N):
		x = fft.fftshift(fft.fft(w[k:k+N], n=N))[N//2:N]
		# assert np.allclose(np.imag(x*np.conj(x)), 0)
		Pxx = 10*np.log10(np.real(x*np.conj(x)))
		S.append(Pxx)
	S = np.array(S)

	f = fft.fftshift(fft.fftfreq(N, d=1/rate))[N//2:N]

	return S, f

# spectrogram to batch
def s2batch(s):
	batch_size = 512
	step = int(batch_size/1)
	batch = np.array([s[0:batch_size]])
	count = step
	while( count+batch_size < s.shape[0] ):
		batch = np.vstack( (batch, np.array([s[count:count+batch_size]]) ) )
		count = count + step

	return batch


"""
train data
"""
# Positional encoding
