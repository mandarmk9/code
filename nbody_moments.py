#!/usr/bin/env python3
"""A script for reading and plotting Schroedinger and N-body power spectra."""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from functions import spectral_calc, smoothing, EFT_sm_kern, Psi_q_finder
from zel import eulerian_sampling

L = 1
N = 1000
dx = L / N
x = np.arange(0, L, dx)
A = 0.5 * np.sin(2 * np.pi * x / L)
n = 5
k = np.fft.fftfreq(x.size, dx) * L
dA = spectral_calc(A, k, L, o=n, d=0)
signs = [+1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1]
if n%2 == 0:
   A_ = A
else:
   A_ = 0.5 * np.cos(2 * np.pi * x / L)
dA_an = signs[n-1] * ((2*np.pi)**n / L**n) * A_ #np.cos(2 * np.pi * x / L)

fig, ax = plt.subplots()
ax.set_title('order = {}'.format(n))
ax.plot(x, dA_an, c='k', lw=2, label='An')
ax.plot(x, dA, c='b', lw=2, ls='dashed', label='Num')
plt.savefig('../plots/cosmo_sim/spec_test.png', bbox_inches='tight', dpi=120)

# i_sch, i_nb = 0, 0
# sch_filename = '../data/sch_nbody_k1/psi_{0:05d}.hdf5'.format(i_sch)
#
# path = 'cosmo_sim_1d/nbody_test_k1/'
# nbody_filename = 'output_{0:04d}.txt'.format(i_nb)
# nbody_file = np.genfromtxt(path + nbody_filename)
#
# with h5py.File(sch_filename, 'r') as hdf:
#    ls = list(hdf.keys())
#    A = np.array(hdf.get(str(ls[0])))
#    a = np.array(hdf.get(str(ls[1])))
#    print('a_sch = ', a)
#    L, h, H0 = np.array(hdf.get(str(ls[2])))
#    psi = np.array(hdf.get(str(ls[3])))
#
# Nx = psi.size
# dx = L / Nx
# x = np.arange(0, L, dx)
# sigma_x = np.sqrt(h / 2) #/ 50
# k = np.fft.fftfreq(x.size, dx) * L
# sigma_p = h / (2 * sigma_x)
# sm = 1 / (4 * (sigma_x**2))
# W_k_an = np.exp(- (k ** 2) / (4 * sm))
# psi_star = np.conj(psi)
# grad_psi = spectral_calc(psi, k, L, o=1, d=0)
# grad_psi_star = spectral_calc(np.conj(psi), k, L, o=1, d=0)
# lap_psi = spectral_calc(psi, k, L, o=2, d=0)
# lap_psi_star = spectral_calc(np.conj(psi), k, L, o=2, d=0)
#
# #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
# MW_0 = np.abs(psi ** 2)
# MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
# MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))
#
# MH_0 = smoothing(MW_0, W_k_an)
# MH_1 = smoothing(MW_1, W_k_an) #* a
# MH_2 = (smoothing(MW_2, W_k_an) + ((sigma_p**2) * MH_0)) #* a**2
# CH_1 = MH_1 / MH_0
# CH_2 = MH_2 - (MH_1**2 / MH_0)
# x_nbody = nbody_file[:,-1]
# v_nbody = nbody_file[:,2]
# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(i_nb)
# moments_file = np.genfromtxt(path + moments_filename)
# a_nb = moments_file[:,-1][0]
# print('a_nbody = ', a_nb)
# x_cell = moments_file[:,0]
# M0_nbody = moments_file[:,2]
# M1_nbody = moments_file[:,4]
# M2_nbody = moments_file[:,6]
# C1_nbody = moments_file[:,5]
# C2_nbody = moments_file[:,7]
#
# Psi = -Psi_q_finder(x, A, L)
# x_zel_o = x + a*Psi
# v_zel_o = H0 * np.sqrt(a) * (Psi) * a #peculiar velocity
# # C2_nbody = M2_nbody - (M1_nbody**2 / M0_nbody)
#
# nd = eulerian_sampling(x, a, A, L)[1] + 1
#
#
# moments = ['M0', 'M1', 'M2', 'C1', 'C2']
# for MorC, nM in moments:
#    ylabel = r"$\mathrm{{{MorC}}}^{{({nM})}}$".format(MorC=MorC, nM=nM)
#    # nbody_m = '{}{}'.format(MorC, nM)
#    sch_m = '{}H_{}'.format(MorC, nM)
#    nbody_m = '{}{}_nbody'.format(MorC, nM)
#
#    fig, ax = plt.subplots()
#    ax.set_title(r'$a = {}$'.format(a))
#    ax.set_xlabel(r'$x$', fontsize=14)
#    ax.set_ylabel(ylabel, fontsize=14)
#    # ax.set_xlim(0.4, 0.6)
#    # ax.set_ylim(-250, 250)
#
#    # ax.plot(x_cell, locals()[hier_m], c='k', lw=2, label='Nbody: hierarchy')
#    ax.plot(x_cell, locals()[nbody_m], c='k', lw=2, label=r'$N$-body')
#    ax.plot(x, locals()[sch_m], c='b', lw=2, ls='dashed', label='Sch')
#    if MorC+nM == 'C1':
#       ax.plot(x_zel_o, v_zel_o, c='r', lw=2, ls='dotted', label='Zel')
#
#    if MorC+nM == 'M0':
#       ax.plot(x, nd, c='r', lw=2, ls='dotted', label='Zel')
#
#    ax.tick_params(axis='both', which='both', direction='in')
#    ax.ticklabel_format(scilimits=(-2, 3))
#    ax.grid(lw=0.2, ls='dashed', color='grey')
#    ax.yaxis.set_ticks_position('both')
#    ax.minorticks_on()
#    ax.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))
#
#    plt.savefig('../plots/cosmo_sim/{}{}.png'.format(MorC, nM, i_nb), bbox_inches='tight', dpi=120)
#    plt.close()
#
# print(np.max(MH_0) / np.max(M0_nbody))
# print(np.max(MH_1) / np.max(M1_nbody))
# print(np.max(MH_2) / np.max(M2_nbody))
# print(np.max(CH_1) / np.max(C1_nbody))
# print(np.max(v_zel_o) / np.max(CH_1))
#
# print(np.max(CH_2) / np.max(C2_nbody))
