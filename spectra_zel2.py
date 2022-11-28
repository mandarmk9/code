#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from functions import plotter, dn, read_density, EFT_sm_kern, smoothing
from scipy.interpolate import interp1d
from EFT_nbody_solver import *
from zel import * #eulerian_sampling
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/nbody_new_run6/'
# path = 'cosmo_sim_1d/nbody_multi_k/'

j = 0
# for j in range(0, 101, 5):
# a_list = np.arange(0.5, 15, 0.5)
# for j in range(0, 76, 5):
H0 = 100

def SPT(dc_in, k, L, Nx, a):
  """Returns the SPT PS upto 2-loop order"""
  F = dn(5, k, L, dc_in)
  d1k = (np.fft.fft(F[0]) / Nx)
  d2k = (np.fft.fft(F[1]) / Nx)
  d3k = (np.fft.fft(F[2]) / Nx)
  d4k = (np.fft.fft(F[3]) / Nx)
  d5k = (np.fft.fft(F[4]) / Nx)

  # print(d1k[1])
  # print(d2k[1])
  # print(d3k[1])
  # print(d4k[:8])
  # print(d5k[:8])

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

# for j in range(0, 51):
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
dk_par, a, dx = read_density(path, j)
L = 1.0
x = np.arange(0, L, dx)
Nx = x.size
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
k_nb = np.fft.ifftshift(2.0 * np.pi / x_cell[-1] * np.arange(-x_cell.size/2, x_cell.size/2))


M0_par = np.real(np.fft.ifft(dk_par))
M0_par /= np.mean(M0_par)
f_M0 = interp1d(x, M0_par, fill_value='extrapolate')
M0_par = f_M0(x_cell)

M0_k = np.fft.fft(M0_par - 1) / M0_par.size
P_nb = np.real(M0_k * np.conj(M0_k))

# a_list = np.arange(0.5, 1.82, 0.05)
# a_list = [1.8]
a = 0.5
# for j in range(len(a_list)):
# a = a_list[j]
A = [-0.05, 1, -0.5, 11, -0.01, 2, -0.01, 3, -0.01, 4]
# A = [-0.05, 1]
dc_in = initial_density(x, A, L) #(A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))
P_lin, P_1l, P_2l = SPT(dc_in, k, L, Nx, a)
# n = 21
#
# dat_lin = np.around(P_lin[1:n], 12)
# dat_nb = np.around(P_nb[1:n], 12)
# dat_1l = np.around((P_1l - P_lin)[1:n], 12)
# dat_2l = np.around((P_2l - P_1l)[1:n], 12)
#
# import csv
# file = open('../plots/P_data_k1/dat_{0:03d}.csv'.format(j), mode='w')
# header = ['$k$', '$P_{N-\mathrm{body}}$', '$P_\mathrm{lin}$', '$P_\mathrm{1-loop}$', '$P_\mathrm{2-loop}$']
# footer = ['a={}'.format(np.round(a, 4)), '', '', '', '']
# writer = csv.writer(file)
# writer.writerow(header)
# for i in range(dat_lin.size):
#     writer.writerow([i+1, dat_nb[i], dat_lin[i], dat_1l[i], dat_2l[i]])
# writer.writerow(footer)
# file.close()
# print(a)
#
sig = 0
for j in range(len(A) // 2):
    sig += A[2*j]**2 * (1 - np.cos(2 * np.pi * A[2*j + 1] * x / L)) #/ (A[2*j+1]**2)

sig *= a**2 #* k**2
# sig = sigma2(dc_in, x, k, L, k.size, a)

# sig_1l = np.exp(-(sig / 2)) - 1
sig_1l =  -(sig / 2) + ((sig**2) / 8)
# # sig_1l = np.fft.fft(sig_1l) / k.size
# P_ZA_int = np.fft.fft(np.exp(-(k**2 * sig / 2) - 1)) / k.size

# sig = A[0]**2 * (1 - np.cos(2 * np.pi * A[1] * x / L))
# sig += A[2]**2 * (1 - np.cos(2 *np.pi * A[3] * x / L))
# sig *= a**2
# sig_1l = 0
# for n in range(1, 2):
#     sig_1l += sig**n / (2**n * np.math.factorial(n))
# sig_1l *= -1
P_ZA_int = np.fft.fft(sig_1l) / k.size

# P_ZA_int = np.fft.fft(-sig**2 * k / 4) / k.size

zel_sol = eulerian_sampling(x, a, A, L)
q = zel_sol[0]
d0 = zel_sol[1]
d0_k = np.fft.fft(d0) / d0.size
P_ZA_dir = d0_k * np.conj(d0_k)

# # print(np.real(P_ZA_dir[1:14] - P_ZA_int[1:14]))
# sig = sum((Psi_nbody - np.mean(Psi_nbody))**2) / (Psi_nbody.size - 1)
# P_ZA_int = np.fft.fft(np.exp(-(k_nb**2) * sig / 2) - 1) / k_nb.size

k /= (2 * np.pi)
k_nb /= (2 * np.pi)

fig, ax = plt.subplots()
ax.set_title('a = {}'.format(np.round(a, 4)))

ax.scatter(k, P_ZA_dir, c='k', s=50, label='Zel')
# ax.scatter(k_nb, P_nb, c='b', s=40, label=r'$N$-body')
ax.scatter(k, P_ZA_int, c='magenta', s=30, label='Zel: integral')
# ax.scatter(k, P_lin, c='r', s=25, label=r'SPT: lin')
# ax.scatter(k, P_1l, c='g', s=20, label=r'SPT: 1-loop')
# ax.scatter(k, P_2l, c='cyan', s=10, label=r'SPT: 2-loop')

# ax.set_xlim(-0.5, 15.5)
# ax.set_ylim(1e-7, 10)
ax.set_xlim(-0.1, 15.1)
ax.set_ylim(1e-9, 1)
ax.set_xlabel(r'k', fontsize=14)
ax.set_ylabel(r'$P(k)$', fontsize=14)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
ax.yaxis.set_ticks_position('both')
plt.yscale('log')
# print(P_1l[11], P_2l[11], P_nb[11])
# ax.plot(x, d0, c='k', lw=2, label='ZA: direct')
# ax.plot(x, M0_par-1, c='b', lw=2, ls='dashdot', label=r'$N$-body')
# ax.plot(q_nbody, dc_nb, c='r', lw=2, ls='dashed', label=r'$N$-body: from $\Psi$')

plt.savefig('../plots/test/ZA_int/P_ZA.png'.format(j), bbox_inches='tight', dpi=150)
plt.close()
# plt.show()
