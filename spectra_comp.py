#!/usr/bin/env python3

#import libraries
import os
import h5py
import pandas
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from functions import plotter2, read_density, dn, smoothing
from scipy.interpolate import interp1d
from zel import initial_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def SPT(dc_in, k, L, Nx, a):
  """Returns the SPT PS upto 2-loop order"""
  F = dn(5, L, dc_in)
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

def spectra(path, Nfiles, A, mode, sm=False, kind='sharp', Lambda=1):
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    P_lin = np.zeros(Nfiles)
    P_1l = np.zeros(Nfiles)
    P_2l = np.zeros(Nfiles)
    print('\npath = {}'.format(path))

    for j in range(Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        x_cell = moments_file[:,0]
        dk_par, a, dx = read_density(path, j)
        L = 1.0
        x = np.arange(0, L, dx)
        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))

        dc_in = initial_density(x, A, L)

        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par = (M0_par / np.mean(M0_par)) - 1
        if sm == True:
            dc_in = smoothing(dc_in, k, Lambda, kind)
            M0_par = smoothing(M0_par, k, Lambda, kind)

        M0_k = np.fft.fft(M0_par) / M0_par.size

        P_nb_a = np.real(M0_k * np.conj(M0_k))
        P_lin_a, P_1l_a, P_2l_a = SPT(dc_in, k, L, Nx, a)

        P_nb[j] = P_nb_a[mode]
        P_lin[j] = P_lin_a[mode]
        P_1l[j] = P_1l_a[mode]
        P_2l[j] = P_2l_a[mode]
        a_list[j] = a
        print('a = {}'.format(a))

    return a_list, P_nb / a_list**2 / 1e-4, P_lin / a_list**2 / 1e-4, P_1l / a_list**2 / 1e-4, P_2l / a_list**2 / 1e-4

def spec_nbody(path, Nfiles, mode, sm=False, kind='sharp', Lambda=1):
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    print('\npath = {}'.format(path))
    for j in range(Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        x_cell = moments_file[:,0]
        dk_par, a, dx = read_density(path, j)
        L = 1.0
        x = np.arange(0, L, dx)
        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))

        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par = (M0_par / np.mean(M0_par)) - 1
        if sm == True:
            M0_par = smoothing(M0_par, k, Lambda, kind)
        M0_k = np.fft.fft(M0_par) / M0_par.size

        P_nb[j] = (np.real(M0_k * np.conj(M0_k)))[mode]
        a_list[j] = a
        print('a = {}'.format(a))

    return a_list, P_nb / a_list**2 / 1e-4

mode = 1
# path = 'cosmo_sim_1d/nbody_new_run5/'
# Nfiles = 41
# A = [-0.05, 1, -0.5, 7]
# a_list_k7, P_nb_k7, P_lin_k7, P_1l_k7, P_2l_k7 = spectra(path, Nfiles, A, mode)

# path = 'cosmo_sim_1d/nbody_new_run2/'
# Nfiles = 33
# A = [-0.05, 1, -0.5, 11]
# a_list_k11, P_nb_k11, P_lin_k11, P_1l_k11, P_2l_k11 = spectra(path, Nfiles, A, mode)

sm = False
kind = 'sharp'
kind_txt = 'sharp cutoff'
Lambda = 3 * (2*np.pi)
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

Nfiles = 51
path = 'cosmo_sim_1d/sim_k_1_7/run1/'
A = [-0.05, 1, -0.5, 7]
a_list_k7, P_nb_k7, P_lin_k7, P_1l_k7, P_2l_k7 = spectra(path, Nfiles, A, mode)#, sm=sm, kind=kind, Lambda=Lambda)

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = [-0.05, 1, -0.5, 11]
a_list_k11, P_nb_k11, P_lin_k11, P_1l_k11, P_2l_k11 = spectra(path, Nfiles, A, mode)

path = 'cosmo_sim_1d/sim_k_1_15/run1/'
A = [-0.05, 1, -0.5, 15]
a_list_k15, P_nb_k15, P_lin_k15, P_1l_k15, P_2l_k15 = spectra(path, Nfiles, A, mode)


path = 'cosmo_sim_1d/sim_k_1/run1/'
a_list_k1, P_nb_k1 = spec_nbody(path, Nfiles, mode, sm=sm, kind=kind, Lambda=Lambda)

x = np.arange(0, 1, 1/1000)
a_sc = 1 / np.max(initial_density(x, A, 1))

#for plotting the spectra
if sm == True:
    title = r'$k = {},\; \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
else:
    title = r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)

# xaxes = [a_list_k1, a_list_k11, a_list_k11]
# yaxes = [P_nb_k1, P_1l_k11, P_nb_k11]
# errors = [(yaxes[1] - yaxes[2]) * 100 / yaxes[2]]
#
# colours = ['r', 'b', 'b']

xaxes = [a_list_k1, a_list_k7, a_list_k11, a_list_k15, a_list_k7, a_list_k11, a_list_k15]
yaxes = [P_nb_k1, P_nb_k7, P_nb_k11, P_nb_k15, P_1l_k7, P_1l_k11, P_1l_k15]

linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']
# # yaxes_err = [P_nb_k5, P_nb_k7, P_nb_k11]

# df_k1 = pandas.DataFrame(data=[a_list_k1, P_nb_k1])
# pickle.dump(df_k1, open("./data/unsm_spec_sim_k_1.p", "wb"))
#
# df_k11 = pandas.DataFrame(data=[a_list_k11, P_nb_k11])
# pickle.dump(df_k11, open("./data/unsm_spec_sim_k_1_11.p", "wb"))

# data_k1 = pickle.load(open("./data/unsm_spec_sim_k_1.p", "rb" ))
# a_list_k1 = [data[j][0] for j in range(data.shape[1])]
# print(a_list_k1)
# # data_k11 = pickle.load(open("./data/unsm_spec_sim_k_1_11.p", "rb" ))
# #
# # yaxes = [data[j][1] for j in range(data.shape[1])]


# xaxes = [a_list_k1, a_list_k11]
# yaxes = [P_nb_k1, P_nb_k11]
# errors = [(yaxes[j] - yaxes[0]) * 100 / yaxes[0] for j in range(2)]
colours = ['magenta', 'k', 'b', 'r', 'k', 'b', 'r']
# linestyles = ['solid', 'dashed', 'solid', 'dashed', 'dashed', 'solid', 'solid', 'solid']


# # labels = [r'$N$-body: $k_{1} = 1$', r'1-loop SPT: $k_{1} = 1, k_{2} = 5$', r'1-loop SPT: $k_{1} = 1, k_{2} = 7$', r'1-loop SPT: $k_1 = 1, k_{2} = 11$', r'$N$-body: $k_{1} = 1, k_{2} = 5$', r'$N$-body: $k_{1} = 1, k_{2} = 7$', r'$N$-body: $k_{1} = 1, k_{2} = 11$']
labels = []
patch1 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_k\_1}')
patch2 = mpatches.Patch(color='k', label=r'\texttt{sim\_k\_1\_7}')
patch3 = mpatches.Patch(color='b', label=r'\texttt{sim\_k\_1\_11}')
patch4 = mpatches.Patch(color='r', label=r'\texttt{sim\_k\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N-$body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='SPT-4')
line3 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dotted', label='EFT')

handles = []#patch1, patch2, patch3, patch4]
handles2 = [line1, line2]#, line3]
        # ax[1].plot(xaxes[1], errors[1], c=colours[1], ls=linestyles[1], lw=2.5) #if varying \Lambda, then it's i>2 and i-3, if varying simulations, then it's i>3 and i-4


# labels = [r'\texttt{sim\_k\_1}', r'\texttt{sim\_k\_1\_11}']
# colours = ['k', 'b']
# linestyles = ['dashed', 'solid']

err_y = [(yaxes[j] - yaxes[j-3]) * 100 / yaxes[j-3] for j in range(4, 7)]
err_x = [xaxes[i] for i in range(4, 7)]
err_c = [colours[i] for i in range(4, 7)]
err_ls = [linestyles[i] for i in range(4, 7)]
errors = [err_x, err_y, err_c, err_ls]

plots_folder = 'test/new_paper_plots'
save = True
savename = 'spec_comp_unsmoothed' #'spec_comp_k{}_L{}'.format(mode, int(Lambda / (2*np.pi)))
# ylabel = r'$a^{-2}P(k, a) \times 10^{4}$'
ylabel = r'$a^{-2}P(k, a) \times 10^{4}\;\;[h^{-2}\mathrm{Mpc}^{2}]$'

# plotter2(mode, Lambda, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, save=save)
plotter2(mode, Lambda, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, handles2=handles2, save=save)
