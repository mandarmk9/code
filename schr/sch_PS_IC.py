#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fnmatch

from adaptive_ts_sch import *
from zel import eulerian_sampling
from functions import spectral_calc, smoothing, phase
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

L = 2 * np.pi
Nx = 250000
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
# print(k.size, x.size)

#parameters
h = 0.01
H0 = 100

# print(h / m)
i_nb = 0
a0 = 0.5
an = 15
dt_max = 1e-4 #specify the maximum allowed time step in a (the actual time step depends on this and the Courant factor)

gamma = 5.0

def amp_calc(A, k, gamma, a0):
        np.random.seed(seed=12345)
        zeta = np.random.uniform(0, L, k.size)
        A = 1e9
        while True:
                P_k = np.zeros(k.size)
                P_k[1:] = A*np.abs(k[1:])**(-gamma/2)
                dc_in = np.real(np.fft.ifft(np.sqrt(np.abs(P_k * L)) * np.exp(1j*zeta)))
                max_del_Psi = np.max(np.abs(a0*dc_in)) - 1
        if
        return dc_in

del_Psi = -a0*dc_in
print(np.max(np.abs(del_Psi)))

Psi = spectral_calc(del_Psi, L, o=1, d=1)

x_zel = x + a0*Psi
v_zel = H0 * np.sqrt(a0) * (Psi) #peculiar velocity

nd = (1 / (1 - a0*dc_in))
phi_v = phase(nd, L, H0, a0)
v_sch = -spectral_calc(phi_v, L, o=1, d=0)
psi = np.zeros(x.size, dtype=complex)
psi = np.sqrt(nd) * np.exp(-1j * phi_v / h)

M0 = nd-1#np.abs(psi**2) - 1
# # N_out = 200 #the number of time steps after which an output file is written
# # N_out2 = 125 #the number of time steps after which an output file is written after a=1
# # C = 0.5 #the courant factor for the run
# # loc2 = '/vol/aibn31/data1/mandar/data/sch_multi_k_large/'
# #
# # r_ind = 0 #the index of the file you want to restart the run from
# # restart = 1
# # try:
# #     print(os.listdir(loc2))
# #     restart_file = loc2 + fnmatch.filter(os.listdir(loc2), 'psi_*.hdf5')[r_ind]
# # except IndexError:
# #     r_ind = 0
# #     print("No restart file found, starting from the initial condition...\n")
# #     restart = 0
# #
# # if restart != 0:
# #     with h5py.File(restart_file, 'r') as hdf:
# #         ls = list(hdf.keys())
# #         a0 = np.array(hdf.get(str(ls[1])))
# #         psi = np.array(hdf.get(str(ls[3])))
# #         print(a0)
# #         assert a0 < an, "Final time cannot be less than the restart time"
# # else:
# #     pass
# #
# # print("The solver will run from a = {} to a = {}".format(a0, an))
# #
# # time_ev(psi, L, a0, an, h, H0, dt_max, loc2, N_out, C, A, r_ind, N_out2)
# #
# # tn = time.time()
# # print("Run finished in {}s".format(np.round(tn-t0, 5)))
#
# sigma_x = np.sqrt(h / 2)
# sigma_p = h / (2 * sigma_x)
# sm = 1 / (4 * (sigma_x**2))
# W_k_an = np.exp(- (k ** 2) / (4 * sm))
#
# psi_star = np.conj(psi)
# grad_psi = spectral_calc(psi, L, o=1, d=0)
# grad_psi_star = spectral_calc(np.conj(psi), L, o=1, d=0)
# lap_psi = spectral_calc(psi, L, o=2, d=0)
# lap_psi_star = spectral_calc(np.conj(psi), L, o=2, d=0)
#
# #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
# MW_0 = np.abs(psi ** 2)
# MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
# MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))
#
# MH_0 = smoothing(MW_0, kernel=W_k_an)
# MH_1 = smoothing(MW_1, kernel=W_k_an)
# MH_2 = (smoothing(MW_2, kernel=W_k_an) + ((sigma_p**2) * MH_0))
#
# CH_1 = MH_1 / MH_0
# CH_2 = (MH_2 - (MH_1**2 / MH_0))

fig, ax = plt.subplots()
ax.set_title('a = {}'.format(a0))
ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
ax.set_ylabel(r'$\mathrm{M}^{(0)}$')
# ax.set_ylabel(r'$\mathrm{M}^{(0)}$')
# ax.plot(x, MH_0, c='k', lw=2, label=r'Husimi from $\Psi$')
ax.plot(x, M0, c='k', lw=2, label=r'Sch')
# ax.plot(x, M0, c='b', lw=2, label=r'Sch from $\psi$')

ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

plt.savefig('/vol/aibn31/data1/mandar/plots/test/M0.png', bbox_inches='tight')
plt.close()

# fig, ax = plt.subplots()
# ax.set_title('a = {}'.format(a0))
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
# ax.set_ylabel(r'$\mathrm{M}^{(1)}$')
#
# ax.plot(x, MH_1, c='k', lw=2, label=r'Sch')
# ax.plot(x_cell, M1_nbody, c='b', ls='dotted', lw=2, label=r'N-body')
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('/vol/aibn31/data1/mandar/plots/cosmo_sim/M1.png')
# plt.close()
#
# fig, ax = plt.subplots()
# ax.set_title('a = {}'.format(a0))
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
# ax.set_ylabel(r'$\mathrm{M}^{(2)}$')
#
# ax.plot(x, MH_2, c='k', lw=2, label=r'Sch')
# ax.plot(x_cell, M2_nbody, c='b', ls='dotted', lw=2, label=r'N-body')
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('/vol/aibn31/data1/mandar/plots/cosmo_sim/M2.png')
# plt.close()
#
# fig, ax = plt.subplots()
# ax.set_title('a = {}'.format(a0))
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
# ax.set_ylabel(r'$\mathrm{C}^{(1)}$')
#
# ax.plot(x, CH_1, c='k', lw=2, label=r'Sch')
# # ax.plot(x, v_sch, c='r', lw=2, ls='dashed', label=r'Sch: from phase')
# ax.plot(x_zel, v_zel * a0, c='r', ls='dashed', lw=2, label=r'Zel')
# ax.plot(x_cell, C1_nbody, c='b', ls='dotted', lw=2, label=r'N-body')
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('/vol/aibn31/data1/mandar/plots/cosmo_sim/C1.png')
# plt.close()
#
#
# fig, ax = plt.subplots()
# ax.set_title('a = {}'.format(a0))
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
# ax.set_ylabel(r'$\mathrm{C}^{(2)}$')
#
# ax.plot(x, CH_2, c='k', lw=2, label=r'Sch')
# ax.plot(x_cell, C2_nbody, c='b', ls='dotted', lw=2, label=r'N-body')
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('/vol/aibn31/data1/mandar/plots/cosmo_sim/C2.png')
# plt.close()
