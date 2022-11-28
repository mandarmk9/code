#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from EFT_solver_gauss import *
# from EFT_nbody_solver import *

from SPT import SPT_final
from functions import plotter, SPT_real_sm, SPT_real_tr, smoothing, alpha_to_corr
from zel import eulerian_sampling

for Nfiles in range(0, 51):
    print(Nfiles)
    #define directories, file parameteres

    #run5: k2 = 7; Nfiles = 101
    #run2: k2 = 11; Nfiles = 81
    #run6: only k1; Nfiles = 51

    path = 'cosmo_sim_1d/nbody_gauss_run4/'

    Lambda = 3 * (2 * np.pi)
    H0 = 100

    #define lists to store the data
    a_list = np.zeros(Nfiles)


    #An and Bn for the integral over the Green's function
    An = np.zeros(Nfiles)
    Bn = np.zeros(Nfiles)
    Pn = np.zeros(Nfiles)
    Qn = np.zeros(Nfiles)

    An2 = np.zeros(Nfiles)
    Bn2 = np.zeros(Nfiles)
    Pn2 = np.zeros(Nfiles)
    Qn2 = np.zeros(Nfiles)

    An3 = np.zeros(Nfiles)
    Bn3 = np.zeros(Nfiles)
    Pn3 = np.zeros(Nfiles)
    Qn3 = np.zeros(Nfiles)

    #initial scalefactor
    a0 = EFT_solve(0, Lambda, path)[0]

    for file_num in range(Nfiles):
       # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
       #the function 'EFT_solve' return solutions of all modes + the EFT parameters
       ##the following line is to keep track of 'a' for the numerical integration
       if file_num > 0:
          a0 = a

       a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody = param_calc(file_num, Lambda, path)

       a_list[file_num] = a


       ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
       if file_num > 0:
          da = a - a0

          #for α_c using c^2 from fit to τ_l
          Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
          Qn[file_num] = ctot2
          An[file_num] = da * Pn[file_num]
          Bn[file_num] = da * Qn[file_num]

          #for α_c using τ_l directly (M&W)
          Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
          Qn2[file_num] = ctot2_2
          An2[file_num] = da * Pn2[file_num]
          Bn2[file_num] = da * Qn2[file_num]

          #for α_c using correlations (Baumann)
          Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
          Qn3[file_num] = ctot2_3
          An3[file_num] = da * Pn3[file_num]
          Bn3[file_num] = da * Qn3[file_num]

       print('a = ', a, '\n')

    #A second loop for the integration
    for j in range(1, Nfiles):
       An[j] += An[j-1]
       Bn[j] += Bn[j-1]

       An2[j] += An2[j-1]
       Bn2[j] += Bn2[j-1]

       An3[j] += An3[j-1]
       Bn3[j] += Bn3[j-1]


    #calculation of the Green's function integral
    C = 2 / (5 * H0**2)
    An /= (a_list**(5/2))
    An2 /= (a_list**(5/2))
    An3 /= (a_list**(5/2))

    k_ = k
    k_[0] = 1
    alpha_c_naive = C * (An - Bn)[-1]
    alpha_c_naive2 = C * (An2 - Bn2)[-1]
    alpha_c_naive3 = C * (An3 - Bn3)[-1]
    alpha_c_true = (P_nb_a - P_1l_a_tr) / (2 * P_lin_a * k_**2)

    P_eft_tr = P_1l_a_tr + ((2 * alpha_c_naive) * (k**2) * P_lin_a)
    P_eft2_tr = P_1l_a_tr + ((2 * alpha_c_naive2) * (k**2) * P_lin_a)
    P_eft3_tr = P_1l_a_tr + ((2 * alpha_c_naive3) * (k**2) * P_lin_a)
    P_eft_fit = P_1l_a_tr + ((2 * alpha_c_true) * (k**2) * P_lin_a)

    k /= (2 * np.pi)
    fig, ax = plt.subplots()
    ax.set_title(r'a = {}, $\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a_list[-1], int(Lambda/(2*np.pi))))
    ax.scatter(k, P_nb_a, c='b', s=50, label=r'$N-$body')
    # ax.scatter(k, P_2l_a_sm, c='r', s=40, label=r'SPT: 2-loop sm')
    # ax.scatter(k, P_1l_a_tr, c='magenta', s=32, label=r'SPT: 1-loop tr')
    # ax.scatter(k, P_2l_a_tr, c='cyan', s=25, label=r'SPT: 2-loop tr')
    # ax.scatter(k, P_eft3_tr, c='orange', s=15, label=r'EFT: baumann')
    # ax.scatter(k, P_eft_tr, c='k', s=15, label=r'EFT: from fit to $\tau_{l}$')

    ax.set_xlim(-0.1, 15.1)
    ax.set_ylim(1e-9, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$k\;[2\pi h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$P(k)$', fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    # ax.ticklabel_format(scilimits=(-2, 3))
    ax.grid(lw=0.2, ls='dashed', color='grey')
    ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
    ax.yaxis.set_ticks_position('both')
    plt.savefig('../plots/test/eft_spec/PS_{0:03d}.png'.format(Nfiles), bbox_inches='tight', dpi=120)
    plt.close()
