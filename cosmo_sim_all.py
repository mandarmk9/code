#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from functions import Psi_q_finder, spectral_calc, kde_gaussian_moments, smoothing, read_density, det_is_bet, is_between, EFT_sm_kern
from scipy.interpolate import interp1d
from zel import eulerian_sampling
# from nbody_hi import *
# from dk_par import read_density
from itertools import repeat

path = 'cosmo_sim_1d/nbody_new_run/'

i_sc, i_nb = 0, 10
sch_filename = '../data/new_run2/psi_{0:05d}.hdf5'.format(i_sc)
nbody_filename = 'output_{0:04d}.txt'.format(i_nb)
moments_filename = 'output_hierarchy_{0:04d}.txt'.format(i_nb)

# with h5py.File(sch_filename, 'r') as hdf:
#    ls = list(hdf.keys())
#    A = np.array(hdf.get(str(ls[0])))
#    a_sch = np.array(hdf.get(str(ls[1])))
#    L, h, H0 = np.array(hdf.get(str(ls[2])))
#    psi = np.array(hdf.get(str(ls[3])))
#
# Nx = psi.size
# dx = L / Nx
# x = np.arange(0, L, dx)
# k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
#
# sigma_x = 0.1 * np.sqrt(h / 2) #/ 50
# sigma_p = h / (2 * sigma_x)
# sm = 1 / (4 * (sigma_x**2))
# W_k_an = np.exp(- (k ** 2) / (4 * sm))
#
# psi_star = np.conj(psi)
# grad_psi = spectral_calc(psi, L, o=1, d=0)
# grad_psi_star = spectral_calc(np.conj(psi), L, o=1, d=0)
# lap_psi = spectral_calc(psi, L, o=2, d=0)
# lap_psi_star = spectral_calc(np.conj(psi), L, o=2, d=0)
#
# #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
# MW_0 = np.abs(psi ** 2)
# MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
# MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))
#
# MH_0 = smoothing(MW_0, W_k_an)
# MH_1 = smoothing(MW_1, W_k_an)
# MH_2 = smoothing(MW_2, W_k_an) + ((sigma_p**2) * MH_0)
#
# CH_1 = MH_1 / MH_0
# CH_2 = MH_2 - (MH_1**2 / MH_0)


nbody_file = np.genfromtxt(path + nbody_filename)
v_nbody = nbody_file[:,2]
x_nbody = nbody_file[:,-1]

moments_file = np.genfromtxt(path + moments_filename)
x_cell = moments_file[:,0]

n_streams = moments_file[:,1]
M0_nbody = moments_file[:,2]
dk_par, a, dx = read_density(path, i_nb)
x0 = 0.0
xn = 1.0
x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
k = np.fft.ifftshift(2.0 * np.pi / xn * np.arange(-x_cell.size/2, x_cell.size/2))

M0_par = np.real(np.fft.ifft(dk_par))
M0_par /= np.mean(M0_par)
f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')
M0_nbody = f_M0(x_cell)
M1_nbody = moments_file[:,4]
C1_nbody = moments_file[:,5]
M2_nbody = moments_file[:,6]
C2_nbody = moments_file[:,7]


# MH_0 = M0_nbody
# MH_1 = M1_nbody
# CH_1 = C1_nbody
# CH_2 = C2_nbody * M0_nbody

# x_grid = np.arange(0, 1, 1e-4)
# C1_nbody = np.zeros(x_grid.size)
# print(x_grid.size)
# for j in range(x_grid.size - 1):
#     print(j)
#     s = is_between(x_nbody, x_grid[j], x_grid[j+1])
#     try:
#        vels = v_nbody[s[0]]
#     except IndexError:
#        pass
#     C1_nbody[j] = sum(vels) / len(vels)
#
# print(C1_nbody.size, x_grid.size)

# cell_left = 0.9999999999999062
# cell_right = 1.0000999999999063
# ind1 = np.where((x_nbody - cell_left) > 0, x_nbody - cell_left, np.inf).argmin()
# ind2 = np.where((x_nbody - cell_right) > 0, x_nbody - cell_right, np.inf).argmin()
# left_ind = min(ind1, ind2)
# right_ind = max(ind1, ind2)
# print(left_ind, right_ind)
#
# inds, vals = is_between(x_nbody[left_ind:right_ind+1], cell_left, cell_right)
# inds += left_ind
# print(inds)
# # print(det_is_bet(x_nbody, cell_left, cell_right))
# print(is_between(x_nbody, cell_left, cell_right))

#
# j = 1
# passes = 1
# cell_left, cell_right = 0, 0
# M0, C1, M2, cells = [], [], [], []
# dx_cell = 1e-4
# x_grid = np.arange(0, 1, dx_cell)
# while cell_right < 1:
#    cell_right = cell_left + j * dx_cell
#    cell_cen = (cell_left + cell_right) / 2
#    print("Cell {}, bounds: [{}, {}]\n".format(passes, cell_left, cell_right))
#    # s = det_is_bet(x_nbody, cell_left, cell_right)
#    s = is_between(x_nbody, cell_left, cell_right)
#
#    try:
#     vels = v_nbody[s[0]]
#     M0.append(s[1].size)
#     v_temp = sum(vels) / len(vels)
#     C1.append(v_temp)
#     M2.append(-((sum(vels**2)  / v_nbody.size) - (v_temp**2)))
#     cells.append(cell_cen)
#     j = 1
#     cell_left = cell_right
#     passes += 1
#
#    except:
#     print("No particles found, increasing cell size\n")
#     j += 1
#     pass
#
i1, i2 = 0, -1#62300, 62325
# # print(M0_nbody[i1:i2])
#
# # dk_par, a, dx = read_density(path, i_nb)
# # x0 = 0.0
# # xn = 1.0 #+ dx
# # x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
# # M0 = np.real(np.fft.ifft(dk_par))
# # M0 /= np.mean(M0)
# #
# M0_nbody = np.array(M0 / np.mean(M0))
# C1_nbody = np.array(C1)
# M2_nbody = np.array(M2)
#
# f_M0 = interp1d(cells, M0_nbody, fill_value='extrapolate')
# f_C1 = interp1d(cells, C1_nbody, fill_value='extrapolate')
# f_M2 = interp1d(cells, M2_nbody, fill_value='extrapolate')
#
# M0_nbody = f_M0(x_cell)
# C1_nbody = f_C1(x_cell)
# M2_nbody = f_M2(x_cell)
#
# M1_nbody = C1_nbody * M0_nbody
# # print(CH_1[i1:i2])
# # M0_nbody[M0_nbody < 1e-3] = 1
# # C1_nbody = M1_nbody / M0_nbody
# # print(M1_nbody[i1:i2])
# # print(M0_nbody[i1:i2])
# # print(C1_nbody[i1:i2])
# # print(CH_1[i1:i2])
#
# # M2_nbody /= M0_nbody
# C2_nbody = (M2_nbody - (M1_nbody**2 / M0_nbody)) / M0_nbody

a = moments_file[:,-1][0]

# from scipy.interpolate import interp1d
# x_grid = np.arange(0.05, 0.95, 1e-5)
# f_M0 = interp1d(x_cell, M0_nbody, fill_value='extrapolate')
# f_C1 = interp1d(x_cell, C1_nbody, fill_value='extrapolate')
# f_M2 = interp1d(x_cell, M2_nbody, fill_value='extrapolate')
# M0_nbody = f_M0(x)
# C1_nbody = f_C1(x)
# M2_nbody = f_M2(x)
# M1_nbody = C1_nbody * M0_nbody
CH_2 = M2_nbody - (M1_nbody**2 / M0_nbody)
# x_cell = x

# print('a_sch = {}, a_nbody = {}'.format(a_sch, a))
# print('a_nbody = {}'.format(a))

# Lambda = 5 * (2 * np.pi / 1)
# W_EFT = EFT_sm_kern(k, Lambda)
# # M0_nbody = smoothing(M0_nbody, W_EFT)
# # M1_nbody = smoothing(M1_nbody, W_EFT)
# CH_2 = smoothing(CH_2, W_EFT)

moments = ['M1']#['M0', 'M1', 'C1', 'M2', 'C1', 'C2']
# MorC, nM = 'C2'
for MorC, nM in moments:
    ylabel = r"$\mathrm{{{MorC}}}^{{({nM})}}$".format(MorC=MorC, nM=nM)
    # ylabel = r"$\bar{v}\;[\mathrm{{km\;s}}^{{-1}}]$"

    sch_m = '{}H_{}'.format(MorC, nM)
    nbody_m = '{}{}_nbody'.format(MorC, nM)
    hier_m = '{}{}_nbody'.format(MorC, nM)

    fig, ax = plt.subplots()
    ax.set_title(r'$a = {}$'.format(a))
    ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    # ax.set_xlim(0.48, 0.52)
    # ax.set_ylim(-0.1e3, 0.3e3)

    ax.plot(x_cell[i1:i2], locals()[hier_m][i1:i2], c='r', lw=2, ls='dashdot', label=r'$N$-body: hierarchy')
    # if MorC + nM == 'M2':
    # ax.plot(x_cell[i1:i2], locals()[sch_m][i1:i2], c='b', lw=2, ls='dashed', label=r'$N$-body')

    # ax.plot(x_cell, locals()[nbody_m], c='k', lw=2, label=r'$N$-body')

    # if MorC + nM == 'M0':
        # ax.plot(x, dc_zel, c='r', lw=2, ls='dotted', label=r'Zel')

    ax.tick_params(axis='both', which='both', direction='in')
    ax.ticklabel_format(scilimits=(-2, 3))
    ax.grid(lw=0.2, ls='dashed', color='grey')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))

    # plt.savefig('../plots/cosmo_sim/{}{}.png'.format(MorC, nM, i_nb), bbox_inches='tight', dpi=100)
    # plt.close()
    plt.show()

# # M0, M1, M2, C0, C1, C2, a_bad, dx_grid = read_hierarchy(path + '/moments/', i_nb)
# # x_grid = np.arange(0, 1, dx_grid)
# # M2 *= M0
# # C2 = M2 - M1**2 / M0
# nbody_file = np.genfromtxt(path + 'output_{0:04d}.txt'.format(i_nb))
# v_nbody = nbody_file[:,2]
# x_nbody = nbody_file[:,-1]
#
# # dk_par, a, dx = read_density(path, i_nb)
# # x_grid = np.arange(0, 1, dx)
# # M0 = np.real(np.fft.ifft(dk_par))
# # M0 /= np.mean(M0)
# dk_par, a, dx = read_density(path, i_nb)
# x0 = 0.0
# xn = 1.0001
# x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
# M0_par = np.real(np.fft.ifft(dk_par))
# M0_par /= np.mean(M0_par)
# f_M0 = interp1d(x_grid, M0_par)
# M0_par = f_M0(x_cell)
#
# M0 = M0_par# / np.mean(M0_nbody) / 1.2
# M1 = M1_nbody
# M2 = M2_nbody
# C1 = C1_nbody / a
# C2 = C2_nbody
# dx_grid = 1e-5
# x_grid = np.arange(0, 1, dx_grid)
#
# def assign_weight(pos, x_grid):
#    assert pos >= 0
#    dx_grid = x_grid[1] - x_grid[0]
#    diff = np.abs(pos - x_grid)
#    ngp_i = int(np.where(diff == np.min(diff))[0])
#    ngp = float(x_grid[ngp_i])
#    W1 = 1 - (diff[ngp_i] / dx_grid)
#    W2 = 1 - W1
#    return ngp_i, W1, W2
#
# M0 = np.zeros(x_grid.size)
# for j in range(x_nbody.size):
#    i, W1, W2 = assign_weight(x_nbody[j], x_grid)
#    M0[i] += W1
#    try:
#       M0[i+1] += W2
#    except IndexError:
#       print(i)
#       M0[0] += W2
# M0 /= np.mean(M0)
# x_grid = x_grid[2:-2]
# M0 = M0[2:-2]


# j = 1
# passes = 1
# cell_left, cell_right = 0, 0
# M0, C1, M2, cells = [], [], [], []
#
# while cell_right < 1:
#    cell_right = cell_left + j * dx_cell
#    cell_cen = (cell_left + cell_right) / 2
#    print("Cell {}, bounds: [{}, {}]\n".format(passes, cell_left, cell_right))
#    s = is_between(x_nbody, cell_left, cell_right)
#    try:
#       vels = v_nbody[s[0]]
#       M0.append(s[1].size)
#       v_temp = sum(vels) / len(vels)
#       C1.append(v_temp)
#       M2.append(-((sum(vels**2)  / v_nbody.size) - (v_temp**2)))
#       cells.append(cell_cen)
#       j = 1
#       cell_left = cell_right
#       passes += 1
#
#    except:
#       print("No particles found, increasing cell size\n")
#       j += 1
#       pass

# cells = np.array(cells)
# M0 /= np.mean(M0)
# M0 = np.array(M0)
# C1 = np.array(C1)
# M2 = np.array(M2)
# # C1[-1] = C1[0]
# # M2[0] = M2[1]
# # M2[-1] = M2[-2]
# M0 = M0[5:-5]
# C1 = C1[5:-5]
# M2 = M2[5:-5]
# M1 = C1 * M0
# C2 = M2 - (M1**2 / M0)
# cells = cells[5:-5]
# from scipy.interpolate import interp1d
# x_grid = np.arange(0.05, 0.95, 1e-5)
# f_M0 = interp1d(x_cell, M0_nbody)
# f_C1 = interp1d(cells, C1)
# f_M2 = interp1d(cells, M2)
# M0 = f_M0(x_grid)
# C1 = f_C1(x_grid)
# M2 = f_M2(x_grid)
# M2 *= M0
# M1 = C1 * M0
# C2_nbody = M2_nbody - (M1_nbody**2 / M0_nbody)
# C2 = M2 - (M1**2 / M0)
