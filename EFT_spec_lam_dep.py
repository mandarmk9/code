#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import h5py
import numpy as np

from EFT_nbody_solver import *
from zel import initial_density
from functions import plotter2
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def EFT_spec_calc(path, Nfiles, A, mode, Lambda, kind):
    print('\npath = {}'.format(path))

    H0 = 100

    #define lists to store the data
    a_list = np.zeros(Nfiles)

    #An and Bn for the integral over the Green's function
    An = np.zeros(Nfiles)
    Bn = np.zeros(Nfiles)
    Pn = np.zeros(Nfiles)
    Qn = np.zeros(Nfiles)


    #the densitites
    P_nb = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)
    P_1l_tr = np.zeros(Nfiles)
    P_2l_tr = np.zeros(Nfiles)

    #initial scalefactor
    a0 = EFT_solve(0, Lambda, path, A, kind)[0]

    for file_num in range(Nfiles):
       # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
       #the function 'EFT_solve' return solutions of all modes + the EFT parameters
       ##the following line is to keep track of 'a' for the numerical integration
       if file_num > 0:
          a0 = a

       a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody = param_calc(file_num, Lambda, path, A, mode, kind)

       a_list[file_num] = a

       ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
       if file_num > 0:
          da = a - a0

          #for α_c using c^2 from fit to τ_l
          Pn[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
          Qn[file_num] = ctot2_3

       #we now extract the solutions for a specific mode
       P_nb[file_num] = P_nb_a[mode]
       P_lin[file_num] = P_lin_a[mode]
       P_1l_tr[file_num] = P_1l_a_tr[mode]
       P_2l_tr[file_num] = P_2l_a_tr[mode]

       print('a = {}'.format(a))

    #A second loop for the integration
    for j in range(1, Nfiles):
        An[j] = np.trapz(Pn[:j], a_list[:j])
        Bn[j] = np.trapz(Qn[:j], a_list[:j])

    #calculation of the Green's function integral
    C = 2 / (5 * H0**2)
    An /= (a_list**(5/2))


    alpha_c_true = (P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2)
    sign = -alpha_c_true / np.abs(alpha_c_true)
    alpha_c = sign * C * (An - Bn)

    P_eft = P_1l_tr + ((2 * alpha_c) * (k[mode]**2) * P_lin)

    return a_list, P_nb / a_list**2 / 1e-4, P_eft / a_list**2 / 1e-4, P_1l_tr / a_list**2 / 1e-4, P_2l_tr / a_list**2 / 1e-4

mode = 1
path = 'cosmo_sim_1d/nbody_new_run2/'
A = [-0.05, 1, -0.5, 11]
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
Nfiles = 33
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
L1, L2, L3 = [3, 4, 5]

a_list, P_nb_L3, P_eft_L3, P_1l_L3, P_2l_L3 = EFT_spec_calc(path, Nfiles, A, mode, L1 * (2 * np.pi), kind)
a_list, P_nb_L4, P_eft_L4, P_1l_L4, P_2l_L4 = EFT_spec_calc(path, Nfiles, A, mode, L2 * (2 * np.pi), kind)
a_list, P_nb_L5, P_eft_L5, P_1l_L5, P_2l_L5 = EFT_spec_calc(path, Nfiles, A, mode, L3 * (2 * np.pi), kind)

# path = 'cosmo_sim_1d/nbody_new_run6/'
# Nfiles = 21
# a_list_k1, P_nb_k1 = spec_nbody(path, Nfiles, mode, Lambda, kind)

x = np.arange(0, 1, 1/1000)
a_sc = 1 / np.max(initial_density(x, A, 1))
# plots_folder = 'nbody_multi_k_run'
plots_folder = 'test/paper_plots'

#for plotting the spectra
xaxes = []
yaxes = [P_nb_L3, P_nb_L4, P_nb_L5, P_1l_L3, P_1l_L4, P_1l_L5, P_eft_L3, P_eft_L4, P_eft_L5]
for j in range(len(yaxes)):
    xaxes.append(a_list)

yaxes_err = [P_nb_L3, P_nb_L4, P_nb_L5, P_nb_L3, P_nb_L4, P_nb_L5]
errors = [(yaxes[j] - yaxes_err[j-3]) * 100 / yaxes_err[j-3] for j in range(3, 9)]
colours = ['b', 'k', 'r', 'b', 'k', 'r', 'b', 'k', 'r']
labels=[]
linestyles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'dotted']
savename = 'spec_comp_k{}_sim_k_1_11_{}'.format(mode, kind)
ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
title = r'$k = {}$ ({})'.format(mode, kind_txt)

patch1 = mpatches.Patch(color='b', label=r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(L1))
patch2 = mpatches.Patch(color='k', label=r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(L2))
patch3 = mpatches.Patch(color='r', label=r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(L3))
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N-$body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='tSPT-4')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dotted', label='EFT')

handles = [patch1, patch2, patch3]
handles2 = [line1, line2, line3]

plotter2(mode, 2, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, handles=handles, handles2=handles2)
