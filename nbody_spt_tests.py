#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, EFT_sm_kern, smoothing, read_density, SPT_tr, nabla_Psi, Psi, dn
from zel import eulerian_sampling
from scipy.interpolate import interp1d
# from SPT import SPT_agg, SPT_final

path = 'cosmo_sim_1d/'
filenames = ['nbody_new_run4']#, 'nbody_hier_low']#, '13']#, '21', '22', '23', '31', '32', '33']
steps = [250000, 100000]
# filenames = ['_hier_time']
mode = 2
A = [-0.05, 1, -0.5, 11, 0]
L = 1.0
Nx = 8192
dx = L / Nx
x = np.arange(0, L, dx)
Lambda = (2 * np.pi) * 3
k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
dc_in = (A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))
# W_EFT_spt = EFT_sm_kern(k, Lambda)
# dc_in = smoothing(dc_in, W_EFT_spt)

for j in range(len(filenames)):
    file = filenames[j]
    N_steps = steps[j]
    ind = 2
    filepath = path + file + '/'
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(ind)
    moments_file = np.genfromtxt(filepath + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    L = x_cell[-1]
    k_cell = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-x_cell.size/2, x_cell.size/2))
    W_EFT = EFT_sm_kern(k_cell, Lambda)
    # M0_nbody = moments_file[:,2]
    # dc_par_l = M0_nbody - 1

    dk_par, a, dx = read_density(filepath, ind)
    print('a = ', a)
    x0 = 0.0
    xn = 1.0 #+ dx
    x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
    M0_par = np.real(np.fft.ifft(dk_par))
    M0_par /= np.mean(M0_par)

    ##interpolation code for M0_nbody
    ##the extrapolate argument allows a value in x_new to exceed the largest x_old;
    ##this gives a more accurate interpolation
    f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')
    M0_par = f_M0(x_cell)

    dc_par_l = M0_par-1 #smoothing(M0_par-1, W_EFT)

    dk_nb = np.fft.fft(dc_par_l) / x_cell.size
    P_nb = np.real(dk_nb * np.conj(dk_nb))[mode]# * W_EFT**2)[1]

    dc_zel = eulerian_sampling(x, a, A, L)[1]
    dk_zel = np.fft.fft(dc_zel) / dc_zel.size
    P_zel = np.real(dk_zel * np.conj(dk_zel))[mode]# * W_EFT**2)[1]

    dc_spt = a * dc_in

    F = dn(5, k, L, dc_in)
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

    P_lin = (P11)[mode]
    P_1l = (P_lin + P12 + P13 + P22)[mode]
    P_2l = (P_1l + P14 + P15 + P23 + P24 + P33)[mode]

    P_err_lin = np.abs(P_lin - P_zel) * 100 / P_zel
    P_err_nb = np.abs(P_nb - P_zel) * 100 / P_zel
    P_err_spt_1l = np.abs(P_1l - P_zel) * 100 / P_zel
    P_err_spt_2l = np.abs(P_2l - P_zel) * 100 / P_zel


    print("The linear SPT error for a = {}, N = {} is {}% \n".format(a, x_cell.size, P_err_lin))
    print("The 1-loop SPT error for a = {}, N = {} is {}% \n".format(a, x_cell.size, P_err_spt_1l))
    print("The 2-loop SPT error for a = {}, N = {} is {}% \n".format(a, x_cell.size, P_err_spt_2l))
    print("The N-body error for {} particles is {}% \n".format(N_steps, P_err_nb))

# the |\delta(k)|^{2} values have a really small error (for N-body vs SPT) if no smoothing is applied. If I smooth the SPT and the N-body, this of course doesn't change. But if I truncate the SPT instead, there's an error of roughly 4%. This is the source of the difference between the two.
