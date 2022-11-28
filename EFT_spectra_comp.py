#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import h5py
import numpy as np

# from EFT_nbody_solver import *
from zel import initial_density
from scipy.interpolate import interp1d
from functions import plotter2, spec_from_ens, read_density, smoothing
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



Nfiles = 3
mode = 1
Lambda = 3 * (2 * np.pi)
Lambda_int = int(Lambda / (2*np.pi))
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
H0 = 100
n_runs = 8
n_use = 10
zel = False
modes = True
folder_name = '' # '/new_data_{}/L{}'.format('sharp', Lambda_int)
save = False

def spec_nbody(path, Nfiles, mode, Lambda, kind):
    print('\npath = {}'.format(path))

    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)

    for j in range(Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        x_cell = moments_file[:,0]
        dk_par, a, dx = read_density(path, j)
        L = 1.0
        x = np.arange(0, L, dx)
        k = np.fft.ifftshift(2.0 * np.pi / x_cell[-1] * np.arange(-x_cell.size/2, x_cell.size/2))

        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par /= np.mean(M0_par)
        f_M0 = interp1d(x, M0_par, fill_value='extrapolate')
        M0_par = f_M0(x_cell)

        #for smoothing the N-body
        rho_b = 27.755 / a**3
        rho_par = (M0_par) * rho_b
        rho_par_l = smoothing(rho_par, k, Lambda, kind)
        M0_par = (rho_par_l / rho_b)

        M0_k = np.fft.fft(M0_par - 1) / M0_par.size
        P_nb[j] = (np.real(M0_k * np.conj(M0_k)))[mode]
        a_list[j] = a
        print('a = {}'.format(a))

    return a_list, P_nb / a_list**2

Nfiles = 51
path = 'cosmo_sim_1d/sim_k_1_7/run1/'
A = [-0.05, 1, -0.5, 7]
sol = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes)
a_list_k7 = sol[0]
P_nb_k7 = sol[2] / (a_list_k7**2)
P_1l_k7 = sol[3] / (a_list_k7**2)
P_eft_k7 = sol[4] / (a_list_k7**2)
err_Int_k7 = sol[-1]

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = [-0.05, 1, -0.5, 11]
sol = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes)
a_list_k11 = sol[0]
P_nb_k11 = sol[2] / (a_list_k11**2)
P_1l_k11 = sol[3] / (a_list_k11**2)
P_eft_k11 = sol[4] / (a_list_k11**2)
err_Int_k11 = sol[-1]

path = 'cosmo_sim_1d/sim_k_1_15/run1/'
A = [-0.05, 1, -0.5, 15]
sol = spec_from_ens(Nfiles, Lambda, path, A, mode, kind, n_runs, n_use, H0, zel, folder_name, modes)
a_list_k15 = sol[0]
P_nb_k15 = sol[2] / (a_list_k15**2)
P_1l_k15 = sol[3] / (a_list_k15**2)
P_eft_k15 = sol[4] / (a_list_k15**2)
err_Int_k15 = sol[-1]

path = 'cosmo_sim_1d/sim_k_1/run1/'
A = [-0.05, 1, -0.0, 11]
a_list_k1, P_nb_k1 = spec_nbody(path, Nfiles, mode, Lambda, kind)


x = np.arange(0, 1, 1/1000)
a_sc = 1 / np.max(initial_density(x, A, 1))
# plots_folder = 'nbody_multi_k_run'
plots_folder = 'test/new_paper_plots/'

#for plotting the spectra
xaxes = [a_list_k1, a_list_k7, a_list_k11, a_list_k15, a_list_k7, a_list_k11, a_list_k15, a_list_k7, a_list_k11, a_list_k15]
yaxes = [P_nb_k1, P_nb_k7, P_nb_k11, P_nb_k15, P_1l_k7, P_1l_k11, P_1l_k15, P_eft_k7, P_eft_k11, P_eft_k15]
for spec in yaxes:
    spec /= 1e-4

yaxes_err = [P_nb_k7, P_nb_k11, P_nb_k15, P_nb_k7, P_nb_k11, P_nb_k15]
linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'dotted', 'dotted']
colours = ['magenta', 'k', 'b', 'r', 'k', 'b', 'r', 'k', 'b', 'r']
labels = []# [r'$N$-body: $k_{1} = 1$', r'$N$-body: $k_{1} = 1, k_{2} = 5$', r'$N$-body: $k_{1} = 1, k_{2} = 7$', r'$N$-body: $k_{1} = 1, k_{2} = 11$', r'1-loop SPT: $k_{1} = 1, k_{2} = 5$', r'1-loop SPT: $k_{1} = 1, k_{2} = 7$', r'1-loop SPT: $k_1 = 1, k_{2} = 11$', r'Baumann EFT: $k_1 = 1, k_{2} = 5$', r'Baumann EFT: $k_1 = 1, k_{2} = 7$', r'Baumann EFT: $k_1 = 1, k_{2} = 11$']
savename = 'spec_comp_k{}_L{}_{}'.format(mode, int(Lambda / (2*np.pi)), kind)
# ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'

title = r'$k = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
err_x = [xaxes[i] for i in range(4, 10)]
err_y = [(yaxes[j] - yaxes_err[j-4]) * 100 / yaxes_err[j-4] for j in range(4, 10)]
err_c = [colours[i] for i in range(4, 10)]
err_ls = [linestyles[i] for i in range(4, 10)]
errors = [err_x, err_y, err_c, err_ls]
patch1 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_k\_1}')
patch2 = mpatches.Patch(color='k', label=r'\texttt{sim\_k\_1\_7}')
patch3 = mpatches.Patch(color='b', label=r'\texttt{sim\_k\_1\_11}')
patch4 = mpatches.Patch(color='r', label=r'\texttt{sim\_k\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N-$body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='tSPT-4')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dotted', label='EFT')

handles = [patch1, patch2, patch3, patch4]
handles2 = [line1, line2, line3]
save = True
plotter2(mode, Lambda, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, handles=handles, handles2=handles2, save=save)
