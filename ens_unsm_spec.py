#!/usr/bin/env python3

#import libraries
import h5py
import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from functions import plotter, dn, read_density, EFT_sm_kern, smoothing, dc_in_finder, plotter2
from scipy.interpolate import interp1d
from SPT import SPT_final

def SPT(dc_in, L, a):
    """Returns the SPT PS upto 2-loop order"""
    F = dn(5, L, dc_in)
    Nx = dc_in.size
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


def spectra_unsm(path, Nfiles, mode):
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    P_1l = np.zeros(Nfiles)

    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
    moments_file = np.genfromtxt(path + moments_filename)
    a0 = moments_file[:,-1][0]
    q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]
    q_zel = q[::50]
    k_zel = np.fft.ifftshift(2.0 * np.pi * np.arange(-q_zel.size/2, q_zel.size/2))
    P_zel, a_zel = [], []

    for j in range(0, Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        x_cell = moments_file[:,0]
        dk_par, a_, dx = read_density(path, j)
        L = 1.0
        x = np.arange(0, L, dx)
        Nx = x.size
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
        dc_in, k_in = dc_in_finder(path, x_cell)

        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par = (M0_par / np.mean(M0_par)) - 1

        M0_k = np.fft.fft(M0_par) / M0_par.size
        P_nb_a = np.real(M0_k * np.conj(M0_k))
        P_lin_a, P_1l_a, P_2l_a = SPT(dc_in, L, a)

        #we now extract the solutions for a specific mode
        P_nb[j] = P_nb_a[mode]
        P_1l[j] = P_1l_a[mode]
        a_list[j] = a
        print('a = ', a, '\n')
    return a_list, P_nb, P_1l

mode = 1
# Nfiles = 51
# path = 'cosmo_sim_1d/sim_k_1_7/run1/'
# a_list_k7, P_nb_k7, P_1l_k7 = spectra_unsm(path, Nfiles, mode)
#
# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# a_list_k11, P_nb_k11, P_1l_k11 = spectra_unsm(path, Nfiles, mode)
#
# path = 'cosmo_sim_1d/sim_k_1_15/run1/'
# a_list_k15, P_nb_k15, P_1l_k15 = spectra_unsm(path, Nfiles, mode)
#
# path = 'cosmo_sim_1d/sim_k_1/run1/'
# a_list_k1, P_nb_k1, P_1l_k1 = spectra_unsm(path, Nfiles, mode)
#
# xaxes = [a_list_k1, a_list_k7, a_list_k11, a_list_k15, a_list_k1, a_list_k7, a_list_k11, a_list_k15]
# yaxes = [P_nb_k1, P_nb_k7, P_nb_k11, P_nb_k15, P_1l_k1, P_1l_k7, P_1l_k11, P_1l_k15]
#
# df = pandas.DataFrame(data=[xaxes, yaxes])
# pickle.dump(df, open("./data/unsm_spec_comp.p", "wb"))
# print("unsm_spec_comp.p written!\n")

data = pickle.load(open("./data/unsm_spec_comp.p", "rb"))
xaxes = [data[j][0] for j in range(data.shape[1])]
yaxes = [data[j][1] for j in range(data.shape[1])]

plots_folder = 'test/paper_plots/'
a_sc = 1

title = r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)

colours = ['b', 'r', 'k', 'magenta', 'b', 'r', 'k', 'magenta']

labels = []
patch1 = mpatches.Patch(color='b', label=r'\texttt{sim\_k\_1}')
patch2 = mpatches.Patch(color='r', label=r'\texttt{sim\_k\_1\_7}')
patch3 = mpatches.Patch(color='k', label=r'\texttt{sim\_k\_1\_11}')
patch4 = mpatches.Patch(color='magenta', label=r'\texttt{sim\_k\_1\_15}')
line1 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='solid', label='$N-$body')
line2 = mlines.Line2D(xdata=[0], ydata=[0], c='seagreen', lw=2.5, ls='dashed', label='SPT-4')

handles = [patch1, patch2, patch3, patch4]
handles2 = [line1, line2]

linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed']

savename = 'spec_comp_unsmoothed'
ylabel = r'$a^{-2}P(k, a) \times 10^{4}$'

err_x, err_y, err_c, err_ls = [], [], [], []
for j in range(len(yaxes)):
    yaxes[j] *= 1e4 / xaxes[j]**2
    if 3<j<8:
        err_y.append((yaxes[j] - yaxes[j-4]) * 100 / yaxes[j-2])
        err_x.append(xaxes[j])
        err_c.append(colours[j])
        err_ls.append(linestyles[j])

errors = [err_x, err_y, err_c, err_ls]
# print(len(colours), len(linestyles), len(yaxes))
plotter2(mode, 1, xaxes, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, errors, a_sc, title_str=title, handles=handles, handles2=handles2)
