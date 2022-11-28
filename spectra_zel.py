#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from zel import eulerian_sampling
from scipy.interpolate import interp1d
from functions import spectral_calc, smoothing, dn, read_density

def SPT(dc_in, k, L, Nx, a):
  """Returns the SPT PS upto 2-loop order"""
  F = dn(5, k, L, dc_in)
  d1k = (np.fft.fft(F[0]) / Nx)
  d2k = (np.fft.fft(F[1]) / Nx)
  d3k = (np.fft.fft(F[2]) / Nx)
  d4k = (np.fft.fft(F[3]) / Nx)
  d5k = (np.fft.fft(F[4]) / Nx)

  P11 = (d1k * np.conj(d1k)) * (a**2)
  P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
  P22 = (d2k * np.conj(d2k)) * (a**4)
  P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
  P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
  P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
  P33 = (d3k * np.conj(d3k)) * (a**6)
  P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
  P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

  P_lin = P11
  P_1l = P_lin + P12 + P13 + P22
  P_2l = P_1l + P14 + P15 + P23 + P24 + P33

  return np.real(P_lin), np.real(P_1l), np.real(P_2l)

def sigma2(dc_in, x, k, L, Nx, a):
  """calculates the integral in Eq 2.35 of M&W"""
  F = dn(1, k, L, dc_in)
  d1k = (np.fft.fft(F[0]) / Nx)
  P_lin = (d1k * np.conj(d1k)) * (a**2)
  factor = np.zeros(k.size)
  factor[1:] = P_lin[1:] / (k[1:]**2)
  sig_inf = np.trapz(4 * factor, k)
  sig_q = np.fft.ifft(2 * factor * Nx)
  sig2 = sig_inf - sig_q
  return sig2

path = 'cosmo_sim_1d/nbody_new_run/'
j = 0
nbody_filename = 'output_{0:04d}.txt'.format(j)
nbody_file = np.genfromtxt(path + nbody_filename)
x_nbody = nbody_file[:,-1]
v_nbody = nbody_file[:,2]
q_nbody = nbody_file[:,0]

Psi_nbody = nbody_file[:,1]

moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
moments_file = np.genfromtxt(path + moments_filename)
a = moments_file[:,-1][0]
x_cell = moments_file[:,0]
M0_nbody = moments_file[:,2]

dk_par, a, dx = read_density(path, j)
x0 = 0.0
xn = 1.0 #+ dx
x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
k_nb = np.fft.ifftshift(2.0 * np.pi / xn * np.arange(-x_cell.size/2, x_cell.size/2))
M0_par = np.real(np.fft.ifft(dk_par))
M0_par /= np.mean(M0_par)
f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')
dc_nb = f_M0(x_cell)
dk_nb = np.fft.fft(dc_nb-1) / dc_nb.size
P_nb = np.real(dk_nb * np.conj(dk_nb))

L = 1#2 * np.pi
Nx = 10000
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))

A = [-0.05, 1, -0.5, 11, 0]
dc_in = ((A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))) * a

sig = sigma2(dc_in, x, k, L, k.size, a)
P_1l_int = np.fft.fft(-k**2 * sig**2 / 4) / k.size
# P_ZA_int = (np.fft.fft(np.exp(-k**2 * sig**2 / 2) - 1) / k.size)

P_lin, P_1l, P_2l = SPT(dc_in, k, L, k.size, a)

d0 = eulerian_sampling(x, a, A, L)[1]
d0_k = np.fft.fft(d0) / d0.size
P_ZA_dir = d0_k * np.conj(d0_k)

H0 = 100
from functions import Psi_q_finder
Psi = -Psi_q_finder(x, A, L)
x_zel = x + a*Psi
v_zel = H0 * np.sqrt(a) * (Psi) #peculiar velocity


k /= (2 * np.pi)
k_nb /= (2 * np.pi)

# dq_Psi = spectral_calc(Psi_nbody, 1.0, o=1, d=0)
# dc_nb = -dq_Psi / (a)**(0.5)
# f_M0 = interp1d(q_nbody, dc_nb, fill_value='extrapolate')
# dc_nb = f_M0(x_nbody)
#
# dk_nb = np.fft.fft(dc_nb) / dc_nb.size
# P_nb = np.real(dk_nb * np.conj(dk_nb))

fig, ax = plt.subplots()
ax.set_title('a = {}'.format(a))
ax.scatter(k, P_ZA_dir, c='k', s=30, label='ZA: direct')

# ax.scatter(k_nb, P_nb, c='b', s=20, label=r'$N$-body')
# ax.scatter(k, P_1l_int, c='magenta', s=40, label='SPT: 1-loop int')
ax.scatter(k, P_1l, c='r', s=15, label=r'SPT: 1-loop')
# ax.scatter(k, P_2l, c='cyan', s=10, label=r'SPT: 2-loop')
ax.scatter(k, P_lin, c='cyan', s=10, label=r'SPT: lin')

# ax.plot(x, d0, c='k', lw=2, label='ZA: direct')
# ax.plot(x, M0_par-1, c='b', lw=2, ls='dashdot', label=r'$N$-body')
# ax.plot(q_nbody, dc_nb, c='r', lw=2, ls='dashed', label=r'$N$-body: from $\Psi$')

plt.legend()
plt.show()
