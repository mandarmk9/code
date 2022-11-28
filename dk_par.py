#!/usr/bin/env python3
"""A script for reading and plotting Schroedinger and N-body power spectra."""

import os
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from functions import spectral_calc, smoothing, dn, EFT_sm_kern, write_density
from functions import read_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# tmp_st = time.time()
# file_num, Nfiles = 0, 51
# tasks = []
# for j in range(2,9):
#     # path = 'cosmo_sim_1d/final_phase_run{}/'.format(j) #test_run2/'
#     path = 'cosmo_sim_1d/sim_k_1_7_run{}/'.format(j) #test_run2/'
#     dx_grid = 1#0.001
#     print(path)
#     p = mp.Process(target=write_density, args=(path, file_num, Nfiles, dx_grid,))
#     tasks.append(p)
#     p.start()
# for task in tasks:
#     p.join()
# tmp_end = time.time()
# print('multiprocessing takes {}s'.format(np.round(tmp_end-tmp_st, 3)))

# # sim_name = 'sim_k_1_11'
# sim_name = 'sim_k_1'#_15'
# sims = ['sim_k_1', 'sim_k_1_7', 'sim_k_1_11', 'sim_k_1_15']
# sims = ['amp_ratio_test']

# sims = ['another_sim_k_1_11']
sims = ['final_sim_k_1_11']
nruns = 16

# n = 5
for sim_name in sims:
    file_num, Nfiles = 0, 101
    tasks = []
    for j in range(0, nruns): #j=1 represents run2
        # path = 'cosmo_sim_1d/final_phase_run{}/'.format(j) #test_run2/'
        path = 'cosmo_sim_1d/{}/run{}/'.format(sim_name, j+1) #test_run2/'
        dx_grid = 0.00075
        print(path)
        p = mp.Process(target=write_density, args=(path, file_num, Nfiles, dx_grid,))
        tasks.append(p)
        p.start()
    for task in tasks:
        p.join()


# def func(args):
#     """the function you want to run in parallel."""
#     return None
#
# tasks = [] #append the task for a given core to this list
# for j in range(0, ntasks):
#     p = mp.Process(target=func, args=(args))
#     tasks.append(p)
#     p.start()
#
# # Process.start does not execute the process, you need to `join` it
# # This is why we need to append the tasks
# for task in tasks:
#     p.join()


# n = 1
# k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
# dA = spectral_calc(A, L, o=n, d=0)
# dA_an = np.gradient(A, x[1]-x[0])
# # signs = [+1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1]
# # if n%2 == 0:
# #    A_ = A
# # else:
# #    A_ = 0.5 * np.cos(2 * np.pi * x / L)
# #
# # dA_an = signs[n-1] * ((2*np.pi)**n / L**n) * A_ #np.cos(2 * np.pi * x / L)
# # intA_an = A_ / (2 * np.pi / L)**n
#
# fig, ax = plt.subplots()
# ax.set_title('order = {}'.format(n))
# ax.plot(x, dA_an, c='k', lw=2, label='An')
# ax.plot(x, dA, c='b', lw=2, ls='dashed', label='Num')
# plt.savefig('../plots/cosmo_sim/spec_test.png', bbox_inches='tight', dpi=120)

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# j = 3
# dk, a, dx = read_density(path, j)
# print(dx)
# x_grid = np.arange(0, 1, dx)
# M0 = np.real(np.fft.ifft(dk))
# M0 /= np.mean(M0)
#
# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
# moments_file = np.genfromtxt(path + moments_filename)
# a = moments_file[:,-1][0]
#
# x_cell = moments_file[:,0]
# M0_nbody = moments_file[:,2]
#
# fig, ax = plt.subplots()
# ax.set_title(r'$a = {}$'.format(a))
# ax.set_xlabel(r'$x$', fontsize=14)
# ax.set_ylabel(r'$\mathrm{M}^{(0)}$', fontsize=14)
#
# ax.plot(x_cell, M0_nbody, c='k', lw=2, label='Nbody: hierarchy')
# ax.plot(x_grid, M0, c='b', lw=2, ls='dashed', label='Nbody')
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.minorticks_on()
# ax.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/cosmo_sim/M0_{}.png'.format(j), bbox_inches='tight', dpi=120)
# plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# from functions import spectral_calc, smoothing, EFT_sm_kern, dn
#
# dx = 1e-3
# L = 1.0
# x = np.arange(0, L, dx)
# k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
# Nx = x.size
# Lambda = 5
# W_EFT = EFT_sm_kern(k, Lambda)
# A = [-0.05, 1, -0.5, 11, 0]
# dc_in = (A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))
# # dc_in = smoothing(dc_in, W_EFT)
# dc_in_k = np.fft.fft(dc_in) / Nx
#
# H0 = 100
#
# n = 3 #overdensity order of the SPT
# F = dn(n, k, L, dc_in)
#
# d1k = (np.fft.fft(F[0]) / Nx) #* W_EFT
# d2k = (np.fft.fft(F[1]) / Nx) #* W_EFT
# d3k = (np.fft.fft(F[2]) / Nx) #* W_EFT
#
# # print(d1k[1])
# print(d2k[2])
# # print(d3k[1])
#
# # a_list = np.arange(0.5, 15, 0.5)
# # # for j in range(a_list.size):
# # j = 0
# # a = 0.5#a_list[j]
# order_2 = np.real(d1k * np.conj(d1k))# * (a**2)
# order_3 = np.real((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))#  * (a**3)
# order_13 = np.real((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) #* (a**4)
# order_22 = np.real(d2k * np.conj(d2k)) #* (a**4)
# order_4 = order_22 + order_13
# order_list = [order_2, order_3, order_13, order_22]#, order5, order6]
# dk_spt_list = sum(order_list)
# # print(dk_spt_list[1])
# # print(order_2[1])
# # print(order_13[1])
# #
# #
# # P_in = dc_in_k * np.conj(dc_in_k)
# # P_11 = order_2
# # print(P_in[1], P_in[11])
# # print(P_11[1], P_11[11])
# #
# # fig, ax = plt.subplots()
# # ax.set_title('a = {}'.format(a), fontsize=12)
# # # ax.scatter(k, W_EFT, c='r', s=20, label='')
# # ax.scatter(k, dc_in_k * np.conj(dc_in_k), c='b', s=25, label=r'$\delta$')
# #
# # # ax.scatter(k, dc_in_k * W_EFT, c='k', s=15, label=r'$\delta_{l}$')
# # # ax.scatter(k, dk_spt_list, c='b', s=25, label=r'SPT: 1-loop')
# # # ax.scatter(k, order_2, c='k', s=40, label=r'SPT: $P_{11}$')
# # # ax.scatter(k, order_3, c='b', s=30, label=r'SPT: $P_{12}$')
# # # ax.scatter(k, order_13, c='brown', s=20, label=r'SPT: $P_{13}$')
# # # ax.scatter(k, order_22, c='r', s=10, label=r'SPT: $P_{22}$')
# #
# # mode = 1
# # print([order[mode] for order in order_list])
# # ax.axvline(Lambda, c='brown', lw=2, label=r'$\Lambda$')
# # ax.set_xlabel('k', fontsize=14)
# # ax.set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14)
# #
# # ax.set_xlim(0, 12)#Lambda+1)
# # # ax.set_ylim(-1e-4, 1e-3)
# #
# # ax.legend()
# # plt.savefig('../plots/EFT_nbody/SPT_ps/a.png'.format(j), bbox_inches='tight', dpi=120)
# # plt.close()
