#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from functions import plotter, dn, read_density, EFT_sm_kern, smoothing
from scipy.interpolate import interp1d
from EFT_nbody_solver import *
from zel import *


path = '../data/sch_multi_k_large'
# plots_folder = 'test/sch_spectra/' #_k7_L3'

# zero = 0
Nfiles = 251
# j = 0
mode = 1
H0 = 100
Lambda = 5

# #define lists to store the data
# a_list = np.zeros(Nfiles)

# #the densitites
# P_sch = np.zeros(Nfiles)
# P_lin = np.zeros(Nfiles)
# P_1l = np.zeros(Nfiles)
# P_2l = np.zeros(Nfiles)

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

for j in range(0, Nfiles, 5):
    with h5py.File(path + '/psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))

    Nx = psi.size
    dx = L / Nx
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density

    sigma_x = 0.1 * np.sqrt(h / 2) #10 * dx
    sigma_p = h / (2 * sigma_x)
    sm = 1 / (4 * (sigma_x**2))
    W_k_an = np.exp(- (k ** 2) / (4 * sm))
    W_EFT = EFT_sm_kern(k, Lambda)

    psi_star = np.conj(psi)
    grad_psi = spectral_calc(psi, L, o=1, d=0)
    grad_psi_star = spectral_calc(np.conj(psi), L, o=1, d=0)
    #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
    MW_0 = np.abs(psi ** 2) - 1
    M0_k = np.fft.fft(MW_0) * W_k_an / k.size #* W_EFT
    P_sch = np.real(M0_k * np.conj(M0_k))

    dc_in = initial_density(x, A, L)

    ##for truncation
    # dc_in = smoothing(dc_in, W_EFT)
    P_lin, P_1l, P_2l = SPT(dc_in, k, L, Nx, a)


    # #for smoothing
    # P_lin_a *= W_EFT
    # P_1l_a *= W_EFT
    # P_2l_a *= W_EFT


    # # #we now extract the solutions for a specific mode
    # # P_sch[j] = P_sch_a[mode]
    # # P_lin[j] = P_lin_a[mode]
    # # P_1l[j] = P_1l_a[mode]
    # # P_2l[j] = P_2l_a[mode]
    # # a_list[j] = a
    #
    # # print('a = ', a, '\n')
    #
    # d0 = eulerian_sampling(x, a, A, L)[1]
    # d0_k = np.fft.fft(d0) / d0.size
    # P_ZA = d0_k * np.conj(d0_k)
    #
    fig, ax = plt.subplots()
    ax.set_title('a = {}'.format(np.round(a, 4)))
    #
    # ax.scatter(k, P_ZA, c='k', s=30, label='Zel')
    ax.scatter(k, P_sch, c='b', s=30, label=r'Sch')
    ax.scatter(k, P_lin, c='r', s=25, label=r'SPT: lin')
    ax.scatter(k, P_1l, c='magenta', s=15, label=r'SPT: 1-loop')
    ax.scatter(k, P_2l, c='cyan', s=10, label=r'SPT: 2-loop')

    # ax.set_xlim(-0.5, 15.5)
    # ax.set_ylim(1e-7, 1)
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

    plt.savefig('../plots/test/spec_sch_multi_large/PS_{0:03d}.png'.format(j), bbox_inches='tight', dpi=150)
    plt.close()
    # plt.show()

# #for plotting the spectra
# xaxis = a_list
# yaxes = [P_sch / a_list**2, P_lin / a_list**2, P_1l / a_list**2, P_2l / a_list**2]
# colours = ['b', 'r', 'brown', 'k']
# labels = [r'Sch', 'SPT: lin', 'SPT: 1-loop', 'SPT: 2-loop']
# linestyles = ['solid', 'dashed', 'dashed', 'dashed']
# savename = 'k1_sm_L5'
# ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# plotter(mode, Lambda, xaxis, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, which='Sch')
