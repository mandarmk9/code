#!/usr/bin/env python3
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import pandas
import pickle
from functions import read_sim_data, plotter, param_calc_ens, initial_density
from EFT_ens_solver import EFT_solve
from spectra import *
path = 'cosmo_sim_1d/final_phase_run1/'
plots_folder = 'test/paper_plots/'

Nfiles = 61
mode = 1
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
H0 = 100
A = [-0.05, 1, -0.5, 11]

colours = iter(['b', 'r', 'k', 'cyan', 'orange'])
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
for Lambda_int in range(2, 7):
    Lambda = Lambda_int * (2*np.pi)
    print('Lambda = ', Lambda_int)
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)
    P_1l_tr = np.zeros(Nfiles)

    for file_num in range(Nfiles):
        # sol = read_sim_data(path, Lambda, kind, file_num)
        # a_list[file_num] =  sol[0]
        # P_nb[file_num] = sol[-2][mode]
        # P_1l_tr[file_num] = sol[-1][mode]

        # sol = EFT_solve(file_num, Lambda, path, A, kind)
        sol = fun(file_num, path, Lambda, kind)
        a_list[file_num] =  sol[0]
        P_nb[file_num] = sol[1][mode]
        P_1l_tr[file_num] = sol[2][mode]

        print('a = ', sol[0], '\n')

    P_nb *= 1e4 / (a_list**2)
    P_1l_tr *= 1e4 / (a_list**2)

    err = (P_1l_tr - P_nb) * 100 / P_nb

    col = next(colours)
    ax[0].plot(a_list, P_nb, c=col, ls='solid', lw=2.5, label=r'$\Lambda = {}$'.format(Lambda_int))
    ax[0].plot(a_list, P_1l_tr, c=col, ls='dashed', lw=2.5)
    ax[1].plot(a_list, err, c=col, ls='dashed', lw=2.5)

ax[0].set_title(r'$k = {}$ ({})'.format(mode, kind_txt), fontsize=14)
ax[1].set_xlabel(r'$a$', fontsize=16)
ax[0].set_ylabel(r'$a^{-2}P(k=1, a) \times 10^{4}$', fontsize=16)
# ax[1].axhline(0, c=colours[0])

for i in range(2):
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=12)
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11)#, loc=2, bbox_to_anchor=(1,1))
ax[1].set_ylabel('% err', fontsize=16)

plt.subplots_adjust(hspace=0)
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.savefig('../plots/test/spec_lam_dep.png', bbox_inches='tight', dpi=150)
plt.show()
# plt.close()
