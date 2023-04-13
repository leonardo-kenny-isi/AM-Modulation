import numpy as np
from scipy.io.wavfile import write

rate = 80_000
f = 100
t = np.arange(0, 1, 1/rate) # time vector (s)
wave = np.sin(2*np.pi*f*t)

scaled = np.int16(wave / np.max(np.abs(wave)) * 32767)
write('test.wav', rate, scaled)