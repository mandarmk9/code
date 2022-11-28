#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from functions import smoothing, spectral_calc, SPT_real_tr, read_density, write_hier, read_hier
from scipy.interpolate import interp1d
from zel import initial_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'

#
# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
# moments_file = np.genfromtxt(path + moments_filename)
# a0 = moments_file[:,-1][0]
# Lambda = 3 * (2*np.pi)
# kind = 'sharp'
# sm = False
#
# j = 20
# nbody_filename = 'output_{0:04d}.txt'.format(j)
# nbody_file = np.genfromtxt(path + nbody_filename)
# x_nbody = nbody_file[:,-1]
# v_nbody = nbody_file[:,2]
#
# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
# moments_file = np.genfromtxt(path + moments_filename)
# a = moments_file[:,-1][0]
# print(a)
# x = moments_file[:,0]
# M0_hier = moments_file[:,2]
# C0_hier = moments_file[:,3]
# M1_hier = moments_file[:,4]
# C1_hier = moments_file[:,5]
# M2_hier = moments_file[:,6]
# C2_hier = moments_file[:,7]

# sim_name = 'sim_k_1_11'
# zero, Nfiles, Nruns, dx_grid = 0, 51, 8, 1e-4
# tasks = []
# for j in range(Nruns):
#     path = 'cosmo_sim_1d/{}/run{}/'.format(sim_name, j+1) #test_run2/'
#     print(path)
#     p = mp.Process(target=write_hier, args=(zero, Nfiles, path, dx_grid,))
#     tasks.append(p)
#     p.start()
# for task in tasks:
#     p.join()

# # moments = ['M0', 'C0', 'M1', 'M2', 'C1', 'C2']#, 'M1', 'C1', 'M2', 'C1', 'C2']
# moments = ['C2']
# # C2_hier = M2_hier
# # C2 -= C1**2
#
# g = 500
# i1, i2 = 0, -1 #250000-g, 250000+g
#
# for MorC, nM in moments:
#     ylabel = r"$\mathrm{{{MorC}}}^{{({nM})}}$".format(MorC=MorC, nM=nM)
#
#     nbody_hier = '{}{}_hier'.format(MorC, nM)
#     nbody_m = '{}{}'.format(MorC, nM)
#
#     fig, ax = plt.subplots()
#     ax.set_title(r'$a = {}$'.format(a))
#     ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
#     # ax.set_xlabel(r'$k\;[2\pi h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
#     # ax.set_ylabel(r'$P(k)$', fontsize=14)
#
#     ax.set_ylabel(ylabel, fontsize=14)
#
#     # ax.scatter(k, P_nb, c='k', s=20, label=r'$N-$body')
#     # ax.plot(x, dc_in, c='b', lw=2, label=r'analytical')
#     # ax.plot(x, dc_in_Psi, c='r', ls='dashdot', lw=2, label=r'numerical')
#
#     ax.plot(x[i1:i2], locals()[nbody_hier][i1:i2], c='k', lw=2, label='hierarchy')
#     ax.plot(x_grid[i1:i2], locals()[nbody_m][i1:i2], c='b', lw=2, ls='dashed', label='coarse-grained')
#     # ax.plot(x_nbody[i1:i2], locals()[nbody_m][i1:i2], c='b', lw=2, ls='dashed', label='coarse-grained')
#
#
#     # ax.plot(x_nbody[i1:i2], v_nbody[i1:i2], c='k', lw=2, label='hierarchy')
#
#     # ax.plot(x_in, dc_SPT_Psi, c='r', ls='dashdot', lw=2, label='SPT')
#     # ax.plot(x, dc_SPT, c='b', ls='dashed', lw=2, label='SPT an')
#     # ax.set_xlim(0, 10)
#     # plt.legend()
#     ax.tick_params(axis='both', which='both', direction='in')
#     ax.ticklabel_format(scilimits=(-2, 3))
#     ax.grid(lw=0.2, ls='dashed', color='grey')
#     ax.yaxis.set_ticks_position('both')
#     ax.minorticks_on()
#
#     # plt.savefig('../plots/nbody_gauss_run/{}{}/{}{}_{}.png'.format(MorC, nM, MorC, nM, j), bbox_inches='tight', dpi=120)
#     # plt.savefig('../plots/nbody_gauss_run/PS_{}.png'.format(j), bbox_inches='tight', dpi=120)
#
#     # plt.savefig('../plots/amps_sim_k_1_11/hier/{}/{}_{}.png'.format(nbody_m, nbody_m, j), bbox_inches='tight', dpi=150)
#     # plt.close()
#     plt.show()
