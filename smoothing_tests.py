#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from zel import eulerian_sampling as es

N = 1024
L = 2 * np.pi
dx = L / N
x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

Lambda = 6
sm = (Lambda ** 2) / 2

dist = x - 0
dist[dist < 0] += L
dist[dist > L/2] = - L + dist[dist > L/2]

W_x = np.exp(-sm * ((dist)**2))
W_k_an = (np.sqrt(np.pi / sm)) * np.exp(- (k ** 2) / (4 * sm))

W_k_num = np.fft.fft(W_x) * dx


# A = [-0.25, 1, 0, 11]
# f = es(x, a, A)[0] + 1

fig, ax = plt.subplots()
ax.plot(x, W_k_an, color='k', ls='dashed', lw=2, label='analytical')
ax.plot(x, W_k_num, color='b', lw=2, label='numerical')
plt.legend()
plt.show()
