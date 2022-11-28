#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from functions import Psi_q_finder, spectral_calc, kde_gaussian_moments, smoothing
from zel import eulerian_sampling

#grid
L = 1
Nx = 8192
dx = L / Nx
q = np.arange(0, L, dx)
k = np.fft.fftfreq(q.size, dx) * 2.0 * np.pi

k1 = 1
k2 = 11
A1 = -0.05
A2 = -0.5
A = [A1, k1, A2, k2]
H0 = 100

path = 'cosmo_sim_1d/EFT_nbody_run3/'

i = 80
j = 175
sch_filename = '../data/sch_nbody_test5/psi_{0:05d}.hdf5'.format(i)
# zel_filename = 'output_initial.txt'
# zel_filename = 'output_ana_end.txt'
# nbody_filename = 'output_0009.txt'
moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)


with h5py.File(sch_filename, 'r') as hdf:
   ls = list(hdf.keys())
   A = np.array(hdf.get(str(ls[0])))
   a_sch = np.array(hdf.get(str(ls[1])))
   L, h, H0 = np.array(hdf.get(str(ls[2])))
   psi = np.array(hdf.get(str(ls[3])))

Nx = psi.size
dx = L / Nx
x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * L

sigma_x = np.sqrt(h / 2) / 50
sigma_p = h / (2 * sigma_x)
sm = 1 / (4 * (sigma_x**2))
W_k_an = np.exp(- (k ** 2) / (4 * sm))

psi_star = np.conj(psi)
grad_psi = spectral_calc(psi, k, o=1, d=0)
grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
lap_psi = spectral_calc(psi, k, o=2, d=0)
lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

#we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
MW_0 = np.abs(psi ** 2)
MW_1 = ((1j * h ) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))

MH_0 = smoothing(MW_0, W_k_an)
MH_1 = smoothing(MW_1, W_k_an)
MH_2 = smoothing(MW_2, W_k_an) + ((sigma_p**2) * MH_0)

CH_1 = MH_1 / MH_0
CH_2 = MH_2  - (MH_1**2 / MH_0)

# zel_file = np.genfromtxt(path + zel_filename)
# nbody_file = np.genfromtxt(path + nbody_filename)
moments_file = np.genfromtxt(path + moments_filename)

x_cell = moments_file[:,0]
n_streams = moments_file[:,1]
M0_nbody = moments_file[:,2]
M1_nbody = moments_file[:,4]
C1_nbody = moments_file[:,5]
M2_nbody = moments_file[:,6]
# C2_nbody = moments_file[:,7]
C2_nbody = M2_nbody - (M1_nbody**2 / M0_nbody)
a = moments_file[:,-1][0]

print('a_sch = {}, a_nbody = {}'.format(a_sch, a))

Psi = -Psi_q_finder(q, A, L)
x_zel_o = q + a*Psi
v_zel_o = H0 * np.sqrt(a) * (Psi) * a #peculiar velocity

fig, ax = plt.subplots()
ax.set_title(r'$a = {}, a_{{\mathrm{{sch}}}} = {}$'.format(a, a_sch))
# ax.set_ylabel(r'$1 + \delta$', fontsize=14)
# ax.set_ylabel(r'$v$', fontsize=14)
ax.set_ylabel(r'$\sigma$', fontsize=14)

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_xlim(0.4, 0.6)

# ax.plot(x, MH_0, c='brown', lw=2, ls='dashed', label='Sch')
# ax.plot(x_zel_o, v_zel_o, c='k', lw=2, label='Zel')
# ax.plot(xll, M2, c='b', lw=2, label='Sch')
ax.plot(x_cell, M0_nbody, c='r', lw=2, label='Nbody')
ax.plot(x, MH_0, c='b', lw=2, ls='dashed', label='Sch')

# ax.plot(x_nbody, v_nbody, c='b', lw=2, ls='dashed', label='Nbody')

# ax.plot(q_zel, M0_zel, c='k', lw=2, label='Zel')

ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()
ax.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))

plt.savefig('../plots/cosmo_sim/C0.png', bbox_inches='tight', dpi=120)
plt.close()
