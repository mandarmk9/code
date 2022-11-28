#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functions import smoothing, spectral_calc, SPT_real_tr, read_density
from scipy.interpolate import interp1d
from zel import initial_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# path = 'cosmo_sim_1d/phase_full_run1/'
# path = 'cosmo_sim_1d/sim_k_1 (copy)/run1/'
path = 'cosmo_sim_1d/sim_k_1_11/run6/'
# path = 'cosmo_sim_1d/amps_sim_k_1_11/run1/'


moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
moments_file = np.genfromtxt(path + moments_filename)
a0 = moments_file[:,-1][0]
Lambda = 3 * (2*np.pi)
kind = 'sharp'
sm = True
for j in range(1):
    j = 23
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]
    print(x_nbody.size)
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    print(a)
    x = moments_file[:,0]
    L = 1.0#np.max(x)
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
    if sm == True:
        M0 = smoothing(moments_file[:,2], k, Lambda, kind) #with the -1, this is \delta
        C0 = smoothing(moments_file[:,3], k, Lambda, kind) #with the -1, this is \delta
        M1 = smoothing(moments_file[:,4], k, Lambda, kind) #with the -1, this is \delta
        C1 = smoothing(moments_file[:,5], k, Lambda, kind) #with the -1, this is \delta
        M2 = smoothing(moments_file[:,6], k, Lambda, kind) #with the -1, this is \delta
        C2 = smoothing(moments_file[:,7], k, Lambda, kind) #with the -1, this is \delta
        M3 = smoothing(moments_file[:,8], k, Lambda, kind) #with the -1, this is \delta
        C3 = smoothing(moments_file[:,9], k, Lambda, kind) #with the -1, this is \delta
        M4 = smoothing(moments_file[:,10], k, Lambda, kind) #with the -1, this is \delta
        C4 = smoothing(moments_file[:,11], k, Lambda, kind) #with the -1, this is \delta
        M5 = smoothing(moments_file[:,12], k, Lambda, kind) #with the -1, this is \delta
        C5 = smoothing(moments_file[:,13], k, Lambda, kind) #with the -1, this is \delta
    else:
        M0 = moments_file[:,2]
        C0 = moments_file[:,3]
        M1 = moments_file[:,4]
        C1 = moments_file[:,5]
        M2 = moments_file[:,6]
        C2 = moments_file[:,7]
        M3 = moments_file[:,8]
        C3 = moments_file[:,9]
        M4 = moments_file[:,10]
        C4 = moments_file[:,11]
        M5 = moments_file[:,12]
        C5 = moments_file[:,13]

    # fields = [M0]#, C0, M1, C1, M2, C2, M3, C3, M4, C4, M5, C5]
    # for j in range(len(fields)):
    #     fields[j] = smoothing(, k, Lambda, kind)
    #     # print(field)

    # from scipy.interpolate import interp1d
    # initial_file = np.genfromtxt(path + 'output_initial.txt')
    # q = initial_file[:,0]
    # Psi = initial_file[:,1]
    #
    # nbody_file = np.genfromtxt(path + 'output_{0:04d}.txt'.format(j))
    # x_in = nbody_file[:,-1]
    #
    # Nx = x_in.size
    # L = np.max(x_in)
    # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    # dc_in_Psi = -spectral_calc(Psi, L, o=1, d=0) / a0
    # dc_SPT_Psi = SPT_real_tr(smoothing(dc_in_Psi, k, Lambda, kind='gaussian'), k, L, Lambda=1, a=a, kind='gaussian')
    # x_in = np.sort(x_in)
    #
    # k_nb= np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
    # M0 = smoothing(M0, k_nb, Lambda, kind='gaussian')

    # A = [-0.05, 1, -0.5, 11]
    # dc_in = initial_density(x, A, L)
    # Nx = x.size
    # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    # dc_SPT = SPT_real_tr(dc_in, k, L, Lambda=1, a=a, kind='gaussian')

    # f_dc = interp1d(q, dc_SPT_Psi, kind='cubic', fill_value='extrapolate')
    # dc_SPT_Psi = f_dc(x)

    # dk_par, a, dx = read_density(path, j)
    # L = 1.0
    # x_grid = np.arange(0, L, dx)
    #
    # M0_par = np.real(np.fft.ifft(dk_par))
    # M0_par /= np.mean(M0_par)
    # f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')
    # M0_par = f_M0(x)
    # M0_k = np.fft.fft(M0_par - 1) / M0_par.size
    # P_nb = np.real(M0_k * np.conj(M0_k))
    # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-P_nb.size/2, P_nb.size/2)) / (2*np.pi)
    # C1 = v_nbody
    moments = ['M0', 'C0', 'M1', 'M2', 'C1', 'C2', 'M3', 'C3', 'M4', 'C4', 'M5', 'C5']#, 'M1', 'C1', 'M2', 'C1', 'C2']
    # moments = ['C2']
    g = 500
    i1, i2 = 0, -1 #250000-g, 250000+g

    for MorC, nM in moments:
        ylabel = r"$\mathrm{{{MorC}}}^{{({nM})}}$".format(MorC=MorC, nM=nM)

        nbody_m = '{}{}'.format(MorC, nM)

        fig, ax = plt.subplots()
        ax.set_title(r'$a = {}$'.format(a))
        ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
        # ax.set_xlabel(r'$k\;[2\pi h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
        # ax.set_ylabel(r'$P(k)$', fontsize=14)

        ax.set_ylabel(ylabel, fontsize=14)

        # ax.scatter(k, P_nb, c='k', s=20, label=r'$N-$body')
        # ax.plot(x, dc_in, c='b', lw=2, label=r'analytical')
        # ax.plot(x, dc_in_Psi, c='r', ls='dashdot', lw=2, label=r'numerical')

        ax.plot(x[i1:i2], locals()[nbody_m][i1:i2], c='k', lw=2)#, label='hierarchy')


        # ax.plot(x_nbody[i1:i2], v_nbody[i1:i2], c='k', lw=2, label='hierarchy')

        # ax.plot(x_in, dc_SPT_Psi, c='r', ls='dashdot', lw=2, label='SPT')
        # ax.plot(x, dc_SPT, c='b', ls='dashed', lw=2, label='SPT an')
        # ax.set_xlim(0, 10)
        # plt.legend()
        ax.tick_params(axis='both', which='both', direction='in')
        ax.ticklabel_format(scilimits=(-2, 3))
        ax.grid(lw=0.2, ls='dashed', color='grey')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()

        # plt.savefig('../plots/nbody_gauss_run/{}{}/{}{}_{}.png'.format(MorC, nM, MorC, nM, j), bbox_inches='tight', dpi=120)
        # plt.savefig('../plots/nbody_gauss_run/PS_{}.png'.format(j), bbox_inches='tight', dpi=120)

        # plt.savefig('../plots/new_sim_k_1_11/sm/{}/{}_{}.png'.format(nbody_m, nbody_m, j), bbox_inches='tight', dpi=150)
        # plt.close()
        plt.show()
