#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import pandas
import pickle
import numpy as np
from functions import plotter, initial_density, SPT_real_tr, smoothing, alpha_to_corr, alpha_c_finder, EFT_sm_kern, dc_in_finder
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


kind = 'sharp'
kind_txt = 'sharp'
# kind = 'gaussian'
# kind_txt = 'Gaussian'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
plots_folder = 'test/sim_k_1_11/real_space/{}'.format(kind)

Nfiles = 51
mode = 1
Lambda = 3 * (2 * np.pi)
H0 = 100
A = [-0.05, 1, -0.5, 11, 0]

a_list, x, alpha_c_list, err_Int = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, n_runs=8, n_use=6, H0=100, zel=False)
# df = pandas.DataFrame(data=[a_list, x, alpha_c_list, err_Int])
# pickle.dump(df, open("./data/alpha_c_{}.p".format(kind), "wb"))
# data = pickle.load(open("./data/alpha_c_{}.p".format(kind), "rb" ))

# a_list = np.array([data[j][0] for j in range(51)])
# x = np.array([data[j][1] for j in range(data.shape[1])])
# alpha_c_list = np.array([data[j][2] for j in range(51)])
# err_Int = np.array([data[j][3] for j in range(51)])

Nx = x.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dc_in, k_in = dc_in_finder(path, x, interp=True)
W_EFT = EFT_sm_kern(k, Lambda)

# j = 15
for j in range(5, 51):
    a = a_list[j]
    print('a = ', a)
    alpha_c = alpha_c_list[j]
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]

    den_eft, err_eft = alpha_to_corr(alpha_c, a, x, k, L, dc_in, Lambda, kind, err_Int)
    den_spt_tr = SPT_real_tr(dc_in, k, L, Lambda, a, kind)
    den_nbody = smoothing(M0_nbody-1, k, Lambda, kind)

    xaxis = x
    yaxes = [den_nbody+1, den_spt_tr+1, den_eft+1]
    title = 'a = {}, $\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda / (2 * np.pi)), kind_txt)
    xlabel = r'$x\;[h^{-1}\mathrm{Mpc}]$'
    ylabel = r'$1+\delta_{l}(x)$'
    colours = ['b', 'brown', 'k']
    labels = [r'$N$-body', r'SPT', r'EFT']
    linestyles = ['solid', 'dashed', 'dotted']
    savename = 'eft_real_{}_{}'.format(kind, j)
    plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc=0, title_str=title)
