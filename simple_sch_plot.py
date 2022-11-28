#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from functions import spectral_calc

Nx = 2**14
L = 2 * np.pi
dx = L / Nx
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))

order = 8

x = np.arange(0, L, dx)
y = np.cos(2 * np.pi * x / L)
dy = spectral_calc(y, L, o=order, d=0)
dy_an = -(2 * np.pi / L)**order * np.sin(2 * np.pi * x / L)
j = 0
dy = y
while j < 9:
    dy = np.gradient(dy, x)
    j += 1
plt.plot(x, dy, label='num')
plt.plot(x, dy_an, label='an')
plt.legend()
plt.show()
