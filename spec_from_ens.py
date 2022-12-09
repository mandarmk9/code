#!/usr/bin/env python3
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import pandas
import pickle
from functions import read_sim_data, plotter, param_calc_ens, smoothing, spec_from_ens
from zel import *

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# path = 'cosmo_sim_1d/final_phase_run1/'
# path = 'cosmo_sim_1d/sim_k_1_11/run1/'

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8

# path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
# n_runs = 24

# path = 'cosmo_sim_1d/sim_k_1/run1/'


# path = 'cosmo_sim_1d/multi_k_sim/run1/'

# A = [-0.05, 1, -0.5, 11]
# A = [-0.05, 1, 0, 11]
A = [-0.05, 1, -0.5, 11]

# plots_folder = '/sim_k_1_11/'
# plots_folder = '/sim_k_1/'
plots_folder = '/test/new_paper_plots/'
# plots_folder = '/new_sim_k_1_11/'


Nfiles = 21
mode = 1
Lambda = 3 * (2 * np.pi)
Lambda_int = int(Lambda / (2*np.pi))
kind = 'sharp'
kind_txt = 'sharp'# cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian'# smoothing'
leg = True
H0 = 100
n_use = n_runs-1
zel = False
modes = True
folder_name = '' # '/new_data_{}/L{}'.format('sharp', Lambda_int)
save = True
# fitting_method = 'WLS'
# fitting_method = 'lmfit'
fitting_method = 'curve_fit'
# fitting_method = ''
nbins_x, nbins_y, npars = 15, 15, 6


#for plotting the spectra
if zel == True:
    a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft4_tr, P_eft_fit, P_zel, a_zel, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel)
    yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft3_tr / a_list**2, P_eft4_tr / a_list**2, P_eft_fit / a_list**2, P_zel / a_zel**2]
else:
    a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft4_tr, P_eft_fit, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, fitting_method='', nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
    yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft3_tr / a_list**2, P_eft4_tr / a_list**2, P_eft_fit / a_list**2]
for spec in yaxes:
    spec /= 1e-4
err_Int /= (1e-4 * a_list**2)

xaxis = a_list
a_sc = 0# 1 / np.max(initial_density(x, A, 1))

colours = ['b', 'brown', 'k', 'cyan', 'orange', 'xkcd:dried blood', 'r', 'seagreen']
labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', 'EFT: $B^{+12}$', r'EFT: FDE', r'EFT: from matching $P_{\mathrm{N-body}}$', 'Zel']
linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted']
savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
xlabel = r'$a$'
# ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'

# ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)

plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg)

# f1 = 'curve_fit'
# # f1 = 'weighted_ls_fit'
#
# #for plotting the spectra
# a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes, fitting_method=f1)
# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2]
# for spec in yaxes:
#     spec /= 1e-4
# err_Int /= (1e-4 * a_list**2)
#
# xaxis = a_list
# a_sc = 0# 1 / np.max(initial_density(x, A, 1))
#
# colours = ['b', 'brown', 'k']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$']
# linestyles = ['solid', 'dashdot', 'dashed']
# # savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# # savename = 'sim_old'
# savename = 'sim_new'
#
# xlabel = r'$a$'
# # ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
# ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
#
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
# #             # err_str = r'$C_{{0}} = {}$'.format(np.round(C_list[i][j][0], 3)) + '\n' + r'$C_{{1}} = {}$'.format(np.round(C_list[i][j][1], 3)) + '\n' + r'$C_{{2}} = {}$'.format(np.round(C_list[i][j][2], 3))
#
# text_str = r'$N_{{\mathrm{{runs}}}} = {}$'.format(n_runs) + '\n' + r'$N_{{\mathrm{{points}}}} = {}$'.format(n_use)
# text_loc = (0.35, 0.05)
#
# texts = [text_str, text_loc]
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg, texts=texts)

# #for plotting the spectra
# a_list, x, P_nb, P_1l_tr, P_eft_tr, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes, fitting_method)
# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft_fit / a_list**2]
# for spec in yaxes:
#     spec /= 1e-4
# err_Int /= (1e-4 * a_list**2)
#
# xaxis = a_list
# a_sc = 0# 1 / np.max(initial_density(x, A, 1))
#
# colours = ['b', 'brown', 'k', 'cyan', 'g']
# labels = [r'$N$-body', 'tSPT-4', r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$',  r'EFT: M\&W', r'EFT: from matching $P_{\mathrm{N-body}}$', 'Zel']
# linestyles = ['solid', 'dashdot', 'dashed',  'dashed', 'dotted']
# savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# # ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
# ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'
#
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, Lambda_int, kind_txt)
#
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=a_sc, title_str=title, terr=err_Int, zel=zel, save=save, leg=leg)
#
# df = pandas.DataFrame(data=[mode, Lambda, xaxis, yaxes, errors, err_Int, a_sc])
# pickle.dump(df, open("spec_plot_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "wb"))
# for Lambda in range(2, 7):
#    Lambda *= (2*np.pi)
# [mode, Lambda, xaxis, yaxes, errors, err_Int, a_sc] = pickle.load(open("spec_plot_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "rb" ))[0]
