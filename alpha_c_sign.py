#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from scipy.signal import convolve

def gauss(x, a, b):
    return np.exp(-a*(x**2) / b)

x = np.arange(0, 10, 0.001)
a1, a2 = 1, 1
b1, b2 = 0.5, 0.5

f1 = gauss(x, a1, b1)
f2 = gauss(x, a2, b2)

c1 = np.convolve(f1, f2)
c2 = np.real(np.fft.ifft(np.fft.fft(f1) * np.fft.fft(f2)))
print(c1.shape, c2.size, x.size)

# fig, ax = plt.subplots()
# ax.plot(x, c1, c='b', label='scipy')
# ax.plot(x, c2, c='r', label='manual')
# plt.legend()
# plt.show()
