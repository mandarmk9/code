#!/usr/bin/env python3
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve2d

#define grids, generate the initial field
L = 1.0
N = 1024
dx = L/N
x = np.arange(0, L, dx)
y = x
a = 1.0
n = -0.24

kx = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
ky = kx
k0 = 1.0
k = np.sqrt(kx**2 + ky**2)
Pk_3d = np.zeros(k.size, dtype='complex')
Pk_3d[1:] = a * (k[1:]/k0)**n
Pk_2d = k * Pk_3d / np.pi

# Pk_2d += 1/N**2

phases = np.random.random((N, N)) #random phases
den_k = np.sqrt(Pk_2d * (L**2)) * np.exp(1j*phases)
den = np.real(np.fft.ifft(den_k))

#distort/mask the density field by convolving with a gaussian of a given beamwidth
KX, KY = np.meshgrid(kx, ky)
X, Y = np.meshgrid(x, y)

sigma = 0.25
gauss = np.exp(-(X**2 + Y**2) / sigma**2)

def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = np.fft.fft2(x)
    fr2 = np.fft.fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(np.fft.ifft2(fr*fr2))
    return cc

den_new = fft_convolve2d(den, gauss)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im = ax[0].pcolormesh(X, Y, den, cmap='inferno', shading='auto')
fig.colorbar(im, ax=ax[0], label=r'$\delta(\mathbf{x})$')

ax[1].plot(k[1:], Pk_2d[1:], c='b', lw=2)
for i in range(2):
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].yaxis.set_ticks_position('both')

ax[0].set_xlabel(r'$x\;[h^{-1}\,\mathrm{Mpc}]$', fontsize=14)
ax[0].set_ylabel(r'$y\;[h^{-1}\,\mathrm{Mpc}]$', fontsize=14)
ax[1].set_xlabel(r'$k$', fontsize=14)
ax[1].set_ylabel(r'$P(k)$', fontsize=14)

plt.tight_layout()
plt.show()
