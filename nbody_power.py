#!/usr/bin/env python3

#import libraries
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

from SPT import SPT_final
from functions import read_density, dc_in_finder, dn, smoothing
from zel import eulerian_sampling
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'

# path = 'cosmo_sim_1d/amps_sim_k_1_11/run1/'
# path_k1 = 'cosmo_sim_1d/amps_sim_k_1/run1/'

def SPT(dc_in, k, L, a):
  """Returns the smoothed SPT PS upto 1-loop order"""
  Nx = k.size
  F = dn(4, L, dc_in)
  #smoothing the overdensity solutions
  d1k = (np.fft.fft(F[0]) / Nx)
  d2k = (np.fft.fft(F[1]) / Nx)
  d3k = (np.fft.fft(F[2]) / Nx)

  P11 = (d1k * np.conj(d1k)) * (a**2)
  P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
  P22 = (d2k * np.conj(d2k)) * (a**4)
  P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)

  P_1l = P11 + P12 + P13 + P22
  return d1k, np.real(P_1l)

def spec(path, j, sm=False, Lambda=1, kind=''):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    dk_par, a, dx = read_density(path, j)
    M0_par = np.real(np.fft.ifft(dk_par))
    M0_par = (M0_par / np.mean(M0_par)) - 1

    x = np.arange(0, 1, dx)
    Nx = x.size
    L = 1.0
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    dc_in, k_in = dc_in_finder(path, x, interp=True)
    if sm == True:
        M0_par = smoothing(M0_par, k, Lambda, kind)
        dc_in = smoothing(dc_in, k, Lambda, kind)

    M0_k = np.fft.fft(M0_par) / M0_par.size
    P_nb = (np.real(M0_k * np.conj(M0_k)))

    d1k, P_1l = SPT(dc_in, k_in, L, a)

    return k, P_nb, P_1l, a

nums = [0, 13, 23, 35]
# nums = [0, 2, 5, 6]#, 1]#, 2, 3, 5]

diff_list, P_nb_k11, P_nb_k1, P_1l_k11, P_1l_k1, a_list = [], [], [], [], [], []
Lambda_int = 3
Lambda = Lambda_int * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
sm = False
for j in nums:
    sol_k11 = spec(path, j, sm=sm, Lambda=Lambda, kind=kind)
    k = sol_k11[0] / (2*np.pi)
    sol_k1 = spec(path_k1, j, sm=sm, Lambda=Lambda, kind=kind)
    k1 = sol_k1[0] / (2*np.pi)

    P_nb_k11.append(sol_k11[1])
    P_nb_k1.append(sol_k1[1])
    P_1l_k11.append(sol_k11[2])
    P_1l_k1.append(sol_k1[2])

    diff = (sol_k1[1][1] - sol_k11[1][1]) * 100 / sol_k11[1][1]
    diff_spt_1 = (sol_k1[2][1] - sol_k1[1][1]) * 100 / sol_k1[1][1]
    diff_spt_11 = (sol_k11[2][1] - sol_k11[1][1]) * 100 / sol_k11[1][1]

    print(diff, diff_spt_1, diff_spt_11)
    # diff_list.append(np.round(diff, 3))
    a_list.append(sol_k11[-1])

P_nb_k1 = np.array(P_nb_k1)
P_1l_k1 = np.array(P_1l_k1)
P_nb_k11 = np.array(P_nb_k11)
P_1l_k11 = np.array(P_1l_k11)

nplots = len(nums)
#density plot
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots(nplots, 1, figsize=(6, 3*nplots), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': np.ones(nplots, dtype=int)})

fig.suptitle(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(Lambda_int, kind_txt), fontsize=16, y=0.91)

n = 200

# Del = [[(P_nb_k1[i][0:n] - P_nb_k11[i][0:n]) * 100 / P_nb_k11[i][0:n]] for i in range(4)]
for i in range(nplots):
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.12, y=0.85)
    # ax[i].scatter(k, P_nb_k11[i], c='b', s=50, label=r'$N-$body: \texttt{sim\_k\_1\_11}')
    # ax[i].scatter(k1, P_nb_k1[i], c='k', s=20, label=r'$N-$body: \texttt{sim\_k\_1}')

    ax[i].scatter(k[:100], (P_nb_k1[i][:100] - P_nb_k11[i][:100]) * 100 / P_nb_k11[i][:100], c='b', s=50, label=r'$N-$body: \texttt{sim\_k\_1\_11}')
    # ax[i].scatter(k1, P_nb_k1[i], c='k', s=20, label=r'$N-$body: \texttt{sim\_k\_1}')

    # ax[i].scatter(k, P_1l_k11[i], c='c', s=20, label=r'SPT-4: \texttt{sim\_k\_1\_11}')
    # ax[i].scatter(k1, P_1l_k1[i], c='r', s=10, label=r'SPT-4: \texttt{sim\_k\_1}')
    # ax[i].scatter(k[0:n], Del[i], c='b', s=50, label=r'$\Delta$')


    ax[i].set_xlim(-0.5, 5.5) #30.5)
    ax[i].set_ylim(1e-9, 1)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
    ax[i].yaxis.set_ticks_position('both')
    ax[i].set_yscale('log')
    # ax[i].set_ylabel(r'$P(k)$', fontsize=14)
    ax[i].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14)
    # ax[i].text(0.75, 0.05, '{}\%'.format(diff_list[i]), bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i].transAxes)

fig.align_labels()
plt.rcParams.update({"text.usetex": True})
ax[0].legend(fontsize=12.5, loc=1, bbox_to_anchor=(1, 1.4))
plt.rcParams.update({"text.usetex": False})
ax[-1].set_xlabel(r'$k\;[2\pi h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
plt.subplots_adjust(hspace=0)
plt.show()
# # plt.savefig('../plots/test/new_paper_plots/den_ev_k1.png', bbox_inches='tight', dpi=150)#, pad_inches=0.3)
# plt.savefig('../plots/test/spec_full.png', bbox_inches='tight', dpi=300)
# plt.close()


# for j in range(0, 1):
#     k, P_nb_k11, a = spec(path, j)
#     k1, P_nb_k1, a1 = spec(path_k1, 0)
#     print('a = ', a)
#
#     k /= (2 * np.pi)
#     k1 /= (2 * np.pi)
#
#     plt.rcParams.update({"text.usetex": True})
#     fig, ax = plt.subplots()
#     ax.set_title(r'$a = {}$'.format(a))
#     ax.scatter(k, P_nb_k11, c='b', s=30, label=r'\texttt{sim\_k\_1\_11}')
#     ax.scatter(k1, P_nb_k1, c='k', s=15, label=r'$sim\_k\_1$')
#
#     ax.set_xlim(-0.5, 50.5)
#     ax.set_ylim(1e-9, 1)
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$k\;[2\pi h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
#     ax.set_ylabel(r'$P(k)$', fontsize=14)
#     ax.minorticks_on()
#     ax.tick_params(axis='both', which='both', direction='in')
#     # ax.ticklabel_format(scilimits=(-2, 3))
#     ax.grid(lw=0.2, ls='dashed', color='grey')
#     ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#     ax.yaxis.set_ticks_position('both')
#     # plt.savefig('../plots/sim_k_1_11/spectra/PS_{0:03d}.png'.format(j), bbox_inches='tight', dpi=120)
#     # plt.close()
#     plt.show()
