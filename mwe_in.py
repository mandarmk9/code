#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import h5py

from mwe_sch import *
from functions import *
from zel import eulerian_sampling

L = 2 * np.pi
Nx = 2**12
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

#parameters
h = 1e-4
H0 = 100
rho_0 = 27.755
m = rho_0 * dx

a0 = 0.1
an = 0.3

dt = 1e-5 #specify the maximum allowed time step in a (the actual time step depends on this and the Courant factor)

A = [-0.01, 1, -0.5, 11] #the 0th and 2nd elements are the two amplitudes, 1st and 3rd are the frequencies

nd = eulerian_sampling(x, a0, A)[1] + 1
phi_v = phase(nd, k, H0, a0)

psi = np.zeros(x.size, dtype=complex)
psi = np.sqrt(nd) * np.exp(-1j * phi_v * m / h)

psi, a = time_ev(psi, k, a0, an, m, h, H0, dt, L=2*np.pi)
# a = a0
# dc_sch = np.abs(psi)**2
# dc_zel = eulerian_sampling(x, a0, A)[1] + 1

sm = 100
W_k_an = np.exp(- (k ** 2) / (4 * sm))

psi_star = np.conj(psi)
grad_psi = spectral_calc(psi, k, o=1, d=0)
grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
lap_psi = spectral_calc(psi, k, o=2, d=0)
lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)
MW_0 = np.abs(psi ** 2)
MW_00 = np.abs(psi ** 2) - 1
MW_1 = (1j * h / m) * ((psi * grad_psi_star) - (psi_star * grad_psi))
MW_2 = - ((h**2) / 4) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))

MH_0_k = np.fft.fft(MW_0) * W_k_an
MH_0 = np.real(np.fft.ifft(MH_0_k))

MH_00_k = np.fft.fft(MW_00) * W_k_an
MH_00 = np.real(np.fft.ifft(MH_00_k))

MH_1_k = np.fft.fft(MW_1) * W_k_an
MH_1 = np.real(np.fft.ifft(MH_1_k))

MH_2_k = np.fft.fft(MW_2) * W_k_an
MH_2 = np.real(np.fft.ifft(MH_2_k))

CH_1 = MH_1 / MH_0
v_pec = CH_1 / a

Psi_q = -Psi_q_finder(x, A)
Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
x_zel = x + Psi_t #eulerian position
v = H0 * np.sqrt(a) * Psi_q #peculiar velocity

v_k = np.fft.fft(v)
v_k *= (W_k_an)
v_zel = np.real(np.fft.ifft(v_k))

fig, ax = plt.subplots()
ax.set_title('a = {}'.format(np.round(a, 4)))
ax.plot(x, v_pec, color='b')
ax.plot(x, v_zel, color='k', ls='dashed')
plt.savefig('/vol/aibn31/data1/mandar/plots/test/tt.png')

tn = time.time()
print("Run finished in {}s".format(np.round(tn-t0, 5)))
