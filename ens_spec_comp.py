#!/usr/bin/env python3

#import libraries
import pickle
import pandas
import h5py

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from functions import plotter2, read_density, dn, smoothing, spec_nbody, spec_from_ens
from scipy.interpolate import interp1d
from zel import initial_density

mode = 1
H0 = 100
n_runs = 8
n_use = 10
zel = False
sm = True

Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
Nfiles = 51

# def saver(Lambda_int):
#     Lambda = Lambda_int * (2 * np.pi)
#     print('Lambda = {}'.format(Lambda_int))
#     path = 'cosmo_sim_1d/sim_k_1_7/run1/'
#     A = [-0.05, 1, -0.5, 7]
#     a_list_k7, x, P_nb_k7, P_1l_k7, P_eft_k7, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int_k7 = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel)
#
#     path = 'cosmo_sim_1d/sim_k_1_11/run1/'
#     A = [-0.05, 1, -0.5, 11]
#     a_list_k11, x, P_nb_k11, P_1l_k11, P_eft_k11, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int_k11 = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel)
#
#     path = 'cosmo_sim_1d/sim_k_1_15/run1/'
#     A = [-0.05, 1, -0.5, 15]
#     a_list_k15, x, P_nb_k15, P_1l_k15, P_eft_k15, P_eft2_tr, P_eft3_tr, P_eft_fit, err_Int_k15 = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel)
#
#     path = 'cosmo_sim_1d/sim_k_1/run1/'
#     A = [-0.05, 1, -0.0, 11]
#     a_list_k1, P_nb_k1 = spec_nbody(path, Nfiles, mode, sm=sm, kind=kind, Lambda=Lambda)
#
#
#     xaxes = [a_list_k1, a_list_k7, a_list_k11, a_list_k15, a_list_k7, a_list_k11, a_list_k15, a_list_k7, a_list_k11, a_list_k15]
#     yaxes = [P_nb_k1, P_nb_k7, P_nb_k11, P_nb_k15, P_1l_k7, P_1l_k11, P_1l_k15, P_eft_k7, P_eft_k11, P_eft_k15]
#
#     df = pandas.DataFrame(data=[xaxes, yaxes])
#     pickle.dump(df, open("./data/spec_comp_plot_y_{}_L{}.p".format(kind, Lambda_int), "wb"))
#     print("spec_comp_plot_{}_L{}.p written!\n".format(kind, Lambda_int))
#     return None
#
# tasks = []
# for Lambda_int in range(2,7):
#     p = mp.Process(target=saver, args=(Lambda_int,))
#     tasks.append(p)
#     p.start()
#
# for task in tasks:
#     p.join()


data = pickle.load(open("./data/spec_comp_plot_y_{}_L{}.p".format(kind, int(Lambda/(2*np.pi))), "rb" ))
# xaxes, yaxes = data[:,0], data[:,1]
xaxes = [data[j][0] for j in range(data.shape[1])]
yaxes = [data[j][1] for j in range(data.shape[1])]

plots_folder = 'test/paper_plots/'
# x = np.arange(0, 1, 1/1000)
a_sc = 1 #/ np.max(initial_density(x, A, 1))

#for plotting the spectra
if sm == True:
    title = r'$k = {},\; \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
else:
    title = r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)

colours = ['b', 'r', 'k', 'magenta', 'r', 'k', 'magenta', 'r', 'k', 'magenta']


# xaxes = [a_list_k1, a_list_k5, a_list_k7, a_list_k11, a_list_k5, a_list_k7, a_list_k11]
# yaxes = [P_nb_k1, P_1l_k5, P_1l_k7, P_1l_k11, P_nb_k5, P_nb_k7, P_nb_k11]
# # yaxes_err = [P_nb_k5, P_nb_k7, P_nb_k11]
# errors = [(yaxes[j] - yaxes[j+3]) * 100 / yaxes[j+3] for j in range(1, 4)]
# colours = ['b', 'magenta', 'k', 'r', 'magenta',                                                                        'k', 'r']
# # labels = [r'$N$-body: $k_{1} = 1$', r'1-loop SPT: $k_{1} = 1, k_{2} = 5$', r'1-loop SPT: $k_{1} = 1, k_{2} = 7$', r'1-loop SPT: $k_1 = 1, k_{2} = 11$', r'$N$-body: $k_{1} = 1, k_{2} = 5$', r'$N$-body: $k_{1} = 1, k_{2} = 7$', r'$N$-body: $k_{1} = 1, k_{2} = 11$']
labels = []
patch1 = mpatches.Patch(color='b', label=r'\texttt{sim\_k\_1}')
patch2 = mpatches.Patch(color='r', label=r'\texttt{sim\_k\_1\_7}')
patch3 = mpatches.Patch(color='k', label=r'\texttt{sim\_k\_1\_11}')
patch4 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_k\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N-$body')
if sm == True:
    line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashdot', label='tSPT-4')
else:
    line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashdot', label='SPT-4')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label=r'EFT: from fit to $\langle[\tau]_{\Lambda}\rangle$')

handles = [patch1, patch2, patch3, patch4]#[patch1, patch2, patch3, patch4]
handles2 = [line1, line2, line3]

# linestyles = ['solid', 'solid', 'solid', 'dashdot', 'dashdot', 'dashed', 'dashed']#, 'solid']
linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'dotted']#, 'solid']

savename = 'spec_comp_k{}_L{}_{}'.format(mode, int(Lambda / (2*np.pi)), kind)
ylabel = r'$a^{-2}P(k, a) \times 10^{4}$'

# yaxes = [P_nb_k1, P_nb_k11, P_nb_k7, P_1l_k11, P_1l_k7, P_eft_k11, P_eft_k7]
err_x, err_y, err_c, err_ls = [], [], [], []
for j in range(len(yaxes)):
    yaxes[j] *= 1e4 / xaxes[j]**2
    if 3<j<7:
        err_y.append((yaxes[j] - yaxes[j-3]) * 100 / yaxes[j-2])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

    elif 6<j<10:
        err_y.append((yaxes[j] - yaxes[j-6]) * 100 / yaxes[j-4])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

errors = [err_x, err_y, err_c, err_ls]
# print(len(colours), len(linestyles), len(yaxes))
plotter2(mode, Lambda, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, handles=handles, handles2=handles2)
