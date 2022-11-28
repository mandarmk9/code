#!/usr/bin/env python3
"""A script for reading and plotting Schroedinger and N-body power spectra."""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from functions import spectral_calc, smoothing, dn, EFT_sm_kern, assign_weight
from dk_par import read_density

Nfiles_sc = 40
Nfiles_nb = 50
Lambda = 5
a_sc_list = np.zeros(Nfiles_sc)
a_nb_list = np.zeros(Nfiles_nb)

dk_sch_list = np.zeros(Nfiles_sc)
dk_nbody_list = np.zeros(Nfiles_nb)
dk_order2_list = np.zeros(Nfiles_nb)
dk_order3_list = np.zeros(Nfiles_nb)
dk_order4_list = np.zeros(Nfiles_nb)
dk_order5_list = np.zeros(Nfiles_nb)
dk_order6_list = np.zeros(Nfiles_nb)
dk_par_list = np.zeros(Nfiles_nb)
# dk_arr_list = np.zeros(Nfiles_nb)

mode = 1

# ###calculate Schroedinger moments
# for i_sc in range(Nfiles_sc):
#    # isc = 81
#    sch_filename = '../data/sch_nbody_test5/psi_{0:05d}.hdf5'.format(i_sc)
#    with h5py.File(sch_filename, 'r') as hdf:
#       ls = list(hdf.keys())
#       A = np.array(hdf.get(str(ls[0])))
#       a_sch = np.array(hdf.get(str(ls[1])))
#       L, h, H0 = np.array(hdf.get(str(ls[2])))
#       psi = np.array(hdf.get(str(ls[3])))
#    print('a_sch = ', a_sch)
#
#    a_sc_list[i_sc] = a_sch
#
#    Nx = psi.size
#    dx = L / Nx
#    x = np.arange(0, L, dx)
#    k = np.fft.fftfreq(x.size, dx) * L
#    sigma_x = np.sqrt(h / 2) / 250
#    sigma_p = h / (2 * sigma_x)
#    sm = 1 / (4 * (sigma_x**2))
#    W_k_an = np.exp(- (k ** 2) / (4 * sm))
#    W_EFT = EFT_sm_kern(k, Lambda)
#
#    # N_spt = 1024
#    # x_spt = np.arange(0, L, L/N_spt)
#    # k_spt = np.fft.fftfreq(N_spt, x_spt[1] - x_spt[0]) * 2.0 * np.pi
#    # W_k_spt = np.exp(-(k_spt ** 2) / (4 * sm))
#    # W_EFT_s = EFT_sm_kern(k_spt, 5)
#    #
#    # dc_in = (A[0] * np.cos(2 * np.pi * x_spt * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x_spt * A[3] / L))
#    # dc_in = smoothing(dc_in, W_EFT_s)
#    # F = dn(3, k_spt, dc_in)
#    #
#    # d1k = (np.fft.fft(F[0]) / N_spt) #* W_EFT_s
#    # d2k = (np.fft.fft(F[1]) / N_spt) #* W_EFT_s
#    # d3k = (np.fft.fft(F[2]) / N_spt) #* W_EFT_s
#    #
#    # order_2 = (d1k * np.conj(d1k)) * (a_sch**2)
#    # order_3 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a_sch**3)
#    # order_13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a_sch**4)
#    # order_22 = (d2k * np.conj(d2k)) * (a_sch**4)
#    # order_4 = order_22 + order_13
#    # order_5 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a_sch**5)
#    # order_6 = (d3k * np.conj(d3k)) * (a_sch**6)
#    #
#    # dk_order2_list[i_sc] = np.real(order_2)[mode]
#    # dk_order3_list[i_sc] = np.real(order_3)[mode]
#    # dk_order4_list[i_sc] = np.real(order_4)[mode]
#    # dk_order5_list[i_sc] = np.real(order_5)[mode]
#    # dk_order6_list[i_sc] = np.real(order_6)[mode]
#
#    psi_star = np.conj(psi)
#    grad_psi = spectral_calc(psi, k, o=1, d=0)
#    grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
#    lap_psi = spectral_calc(psi, k, o=2, d=0)
#    lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)
#
#    #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
#    MW_0 = np.abs(psi ** 2)
#    MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
#    MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))
#
#    # dc_sch = smoothing(MW_0-1, W_k_an)# * W_EFT)
#    # dc_sch_k = np.fft.fft(dc_sch) / Nx
#    # dk2_sch = dc_sch_k * np.conj(dc_sch_k)
#    # dk_sch_list[i_sc] = np.real(dk2_sch[mode])
#
#    MH_0 = smoothing(MW_0, W_k_an)
#    MH_1 = smoothing(MW_1, W_k_an)
#    MH_2 = smoothing(MW_2, W_k_an) + ((sigma_p**2) * MH_0)
#
#    CH_1 = MH_1 / MH_0
#    CH_2 = MH_2 - (MH_1**2 / MH_0)
#    v_k = np.fft.fft(CH_2) / Nx #* W_EFT
#    dk2_sch = v_k * np.conj(v_k)
#    dk_sch_list[i_sc] = np.real(dk2_sch[mode])
#
#
# # dk_spt_sum_list = dk_order2_list + dk_order3_list + dk_order4_list + dk_order5_list + dk_order6_list


###calculate N-body moments
for i_nb in range(Nfiles_nb):
   # i_nb = 179
   path = 'cosmo_sim_1d/nbody_hier/'
   # path = 'cosmo_sim_1d/EFT_nbody_run7/'

   nbody_filename = 'output_{0:04d}.txt'.format(i_nb)
   nbody_file = np.genfromtxt(path + nbody_filename)
   a_nb = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(i_nb))
   print('a = ', a_nb)
   x_nbody = nbody_file[:,-1]

   N_spt = 8192
   L = 1.0
   x_spt = np.arange(0, L, L/N_spt)
   k_spt = np.fft.fftfreq(N_spt, x_spt[1] - x_spt[0]) * L
   W_EFT_s = EFT_sm_kern(k_spt, Lambda)
   A = [-0.05, 1, -0.5, 11, 0]
   dc_in = (A[0] * np.cos(2 * np.pi * x_spt * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x_spt * A[3] / L))
   # dc_in = smoothing(dc_in, W_EFT_s)
   F = dn(3, k_spt, L, dc_in)

   d1k = (np.fft.fft(F[0]) / N_spt) * W_EFT_s
   d2k = (np.fft.fft(F[1]) / N_spt) * W_EFT_s
   d3k = (np.fft.fft(F[2]) / N_spt) * W_EFT_s

   order_2 = (d1k * np.conj(d1k)) * (a_nb**2)
   order_3 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a_nb**3)
   order_13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a_nb**4)
   order_22 = (d2k * np.conj(d2k)) * (a_nb**4)
   order_4 = order_22 + order_13
   order_5 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a_nb**5)
   order_6 = (d3k * np.conj(d3k)) * (a_nb**6)

   dk_order2_list[i_nb] = np.real(order_2)[mode]
   dk_order3_list[i_nb] = np.real(order_3)[mode]
   dk_order4_list[i_nb] = np.real(order_4)[mode]
   dk_order5_list[i_nb] = np.real(order_5)[mode]
   dk_order6_list[i_nb] = np.real(order_6)[mode]


   # dk_par, a, dx = read_density(path, i_nb)
   # x_grid = np.arange(0, 1, dx)
   # # M0 = np.real(np.fft.ifft(dk))
   # # M0 /= np.mean(M0)
   # d2k_par = dk_par * np.conj(dk_par) / (x_nbody.size**2)
   # dk_par_list[i_nb] = np.real(d2k_par[mode])


   # v_nbody = nbody_file[:,2]
   a_nb_list[i_nb] = a_nb
   #
   # def is_between(x, x1, x2):
   #    """returns a subset of x that lies between x1 and x2"""
   #    indices = []
   #    values = []
   #    for j in range(x.size):
   #       if x1 <= x[j] <= x2:
   #          indices.append(j)
   #          values.append(x[j])
   #    values = np.array(values)
   #    indices = np.array(indices)
   #    return indices, values
   #
   # dx_grid = 1e-3
   # x_grid = np.arange(0, 1, dx_grid)
   # v_mean = np.zeros(x_grid.size)
   # rho_mean = np.zeros(x_grid.size)
   # for j in range(x_grid.size - 1):
   #    s = is_between(x_nbody, x_grid[j], x_grid[j+1])
   #    # vels = v_nbody[s[0]]
   #    # v_mean[j] = sum(vels) / len(vels)
   #    rho_mean[j] = s[1].size
   #
   # k_nb = np.fft.fftfreq(x_grid.size, dx_grid) * 2.0 * np.pi
   # W_k_nb = np.exp(-(k_nb ** 2) / (4 * sm))
   # W_EFT_n = EFT_sm_kern(k_nb, 5)
   #
   # del_mean = (rho_mean / np.mean(rho_mean)) - 1
   # del_mean[-1] = del_mean[0]
   # print(dx)

   # x = np.arange(0, 1.0, dx)
   # k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
   # W_EFT = EFT_sm_kern(k, Lambda)
   #
   # d2k_par = dk_par * np.conj(dk_par) * W_EFT**2 / (x_nbody.size**2)
   # dk_par_list[i_nb] = np.real(d2k_par[mode])

   # par_num = x_nbody.size
   # x_grid_par = np.arange(0.0, 1.0, 1/par_num)
   # k_grid_par = np.fft.fftfreq(x_grid_par.size, x_grid_par[1] - x_grid_par[0]) * 2.0 * np.pi
   # dk_par = np.zeros(x_grid_par.size, dtype=complex)
   # for j in range(par_num):
   #    print(j)
   #    dk_par += np.exp(1j * k_grid_par * x_nbody[j])
   #
   # dk_par /= par_num
   # d2k_par = dk_par * np.conj(dk_par)
   # dk_par_list[i_nb] = np.real(d2k_par[mode])

   moments_filename = 'output_hierarchy_{0:04d}.txt'.format(i_nb)
   moments_file = np.genfromtxt(path + moments_filename)
   a_nb = moments_file[:,-1][0]
   # print('a_nbody = ', a_nb)


   x_cell = moments_file[:,0]
   M0_nbody = moments_file[:,2]
   # C0_nbody = moments_file[:,3]

   M1_nbody = moments_file[:,4]
   M2_nbody = moments_file[:,6]
   C1_nbody = moments_file[:,5]
   C2_nbody = M2_nbody - (M1_nbody**2 / M0_nbody)

   dk_par, a, dx = read_density(path, i_nb)
   x0 = 0.0
   xn = 1.000005
   x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
   M0_par = np.real(np.fft.ifft(dk_par))
   M0_par /= np.mean(M0_par)

   # print(x_grid[0], x_grid[-1])
   # print(x_cell[0], x_cell[-1])
   # print(x_cell.size - x_grid.size)
   ##interpolation code for M0_nbody
   from scipy.interpolate import interp1d
   f_M0 = interp1d(x_grid, M0_par)
   M0_par = f_M0(x_cell)

   k_cell = np.fft.fftfreq(x_cell.size, x_cell[1] - x_cell[0]) * x_cell[-1]
   print(k_cell)
   W_EFT = EFT_sm_kern(k_cell, Lambda)

   dk_par = np.fft.fft(M0_par) / x_cell.size * W_EFT
   d2k_par = dk_par * np.conj(dk_par)
   dk_par_list[i_nb] = np.real(d2k_par[mode])

   # v_k_nb = np.fft.fft(C2_nbody) / x_cell.size #* W_EFT
   # v_k2_nb = v_k_nb * np.conj(v_k_nb)
   # dk_par_list[i_nb] = np.real(v_k2_nb[mode])

   # from scipy.interpolate import interp1d
   # f_M0 = interp1d(x_cell, M0_nbody)
   # dx_grid = 1e-7
   # x_grid = np.arange(np.min(x_cell), np.max(x_cell), dx_grid)
   # M0_nbody = f_M0(x_grid)

   # dx_grid = 1e-3
   # x_grid = np.arange(0, 1, dx_grid)
   # M0 = np.zeros(x_grid.size)
   # for j in range(x_nbody.size):
   #    i, W1, W2 = assign_weight(x_nbody[j], x_grid)
   #    M0[i] += W1
   #    try:
   #       M0[i+1] += W2
   #    except IndexError:
   #       M0[0] += W2
   # M0 /= np.mean(M0)
   # # x_grid = x_grid[2:-2]
   # # M0 = M0[2:-2]
   #
   # dc_nbody = M0_nbody - 1
   # dk2_nbody = np.fft.fft(dc_nbody) * np.conj(np.fft.fft(dc_nbody)) / (x_cell.size**2)
   # dk_nbody_list[i_nb] = np.real(dk2_nbody[mode])
   # # dk2_v = np.fft.fft(v_mean) * np.conj(np.fft.fft(v_mean)) / (x_grid.size**2)
   # # dk_par_list[i_nb] = np.real(dk2_v[mode])

dk_spt_sum_list = dk_order2_list + dk_order3_list + dk_order4_list + dk_order5_list + dk_order6_list


fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
ax[0].set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14)
# ax[0].set_ylabel(r'$|\tilde{v}(k)|^{2}$', fontsize=14)
# ax[0].set_ylabel(r'$|\tilde{\sigma}(k)|^{2}$', fontsize=14)

ax[1].set_xlabel(r'$a$', fontsize=14)

# ax[0].plot(a_sc_list, dk_sch_list, label='Sch', lw=2.5, c='b')
ax[0].plot(a_nb_list, dk_par_list, label='Nbody: particles', lw=2.5, c='k')
# ax[0].plot(a_nb_list, dk_nbody_list, label='Nbody: hierarchy', lw=2.5, c='k')
ax[0].plot(a_nb_list, dk_order2_list, label=r'SPT; $\mathcal{O}(2)$', ls='dashed', lw=2.5, c='r')
ax[0].plot(a_nb_list, dk_spt_sum_list, label=r'SPT; $\sum \mathcal{O}(6)$', ls='dashed', lw=2.5, c='brown')

# # #bottom panel; errors
err_order_2 = (dk_order2_list - dk_par_list) * 100 / dk_par_list
err_spt_all = (dk_spt_sum_list - dk_par_list) * 100 / dk_par_list
# # err_nb = (dk_par_list - dk_sch_list) * 100 / dk_sch_list
ax[1].axhline(0, c='b', lw=2.5)

ax[1].plot(a_nb_list, err_spt_all, ls='dashed', lw=2.5, c='brown')
ax[1].plot(a_nb_list, err_order_2, ls='dashed', lw=2.5, c='r')
# ax[1].plot(a_nb_list, err_nb, ls='dashed', lw=2.5, c='k')

ax[1].set_ylabel('% err', fontsize=14)
ax[1].minorticks_on()

for i in range(2):
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].grid(lw=0.2, ls='dashed', color='grey')
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

plt.savefig('../plots/cosmo_sim/sp_M0.png'.format(mode), bbox_inches='tight', dpi=120)
plt.close()


###interpolation code for M0_nbody
# from scipy.interpolate import interp1d
# f_dc = interp1d(x_cell, dc_nbody)
# x_cell = np.sort(x_cell)
# tol = 1e-4
# x_grid = np.arange(tol, 1-tol, 1/x_cell.size)
# k_grid = np.fft.fftfreq(x_grid.size, x_grid[1] - x_grid[0]) * 2.0 * np.pi
# dc_grid = f_dc(x_grid)
# dc_grid_k = np.fft.fft(dc_grid) / x_cell.size
# dk2_nbody = dc_grid_k * np.conj(dc_grid_k)
# dk_nbody_list[i_nb] = np.real(dk2_nbody[mode])
