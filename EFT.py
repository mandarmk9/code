#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import h5py
import matplotlib.pyplot as plt

from SPT import *
from functions import spectral_calc, Psi_q_finder, kde_gaussian

loc2 = '/vol/aibn31/data1/mandar/'
run = '/sch_hfix_run10/'
filename_SPT = loc2 + 'data' + run + 'spt_kern.hdf5'
# j = 75659
mean_MH2, a_list, tau = [], [], []
for j in range(500, 560, 20):
   with h5py.File(loc2 + 'data' + run + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, m, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))
      print(a)
      
#    # gadget_files = '/vol/aibn31/data1/mandar/code/N420/'
#    # file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
#    # pos = np.array(file['/Positions'])
#    # header = file['/Header']
#    # a = header.attrs.get('a')
#    # vel = np.array(file['/Velocities']) / a
#    # N = int(pos.size)
#    # file.close()
#
#    Nx = psi.size
#    dx = L / Nx
#    x = np.arange(0, L, dx)
#    k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
#
#    sigma_x = 25 * dx
#    sigma_p = h / (2 * sigma_x)
#    sm = 1 / (4 * (sigma_x**2))
#
#    W_k_an = np.exp(- (k ** 2) / (4 * sm))
#
#    # q = np.arange(0, L, L/N)
#    # k_nbody = np.fft.fftfreq(q.size, q[1]-q[0]) * 2.0 * np.pi
#    # W_k_nbody = norm * np.exp(- (k_nbody ** 2) / (4 * sm))
#
#    psi_star = np.conj(psi)
#    grad_psi = spectral_calc(psi, k, o=1, d=0)
#    grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
#    lap_psi = spectral_calc(psi, k, o=2, d=0)
#    lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)
#    MW_0 = np.abs(psi ** 2)
#    MW_00 = np.abs(psi ** 2) - 1
#    MW_1 = (1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi))
#    MW_2 = - ((h**2) / 4) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))
#
#    MH_0_k = np.fft.fft(MW_0) * W_k_an
#    MH_0 = np.real(np.fft.ifft(MH_0_k))
#
#    MH_00_k = np.fft.fft(MW_00) * W_k_an
#    MH_00 = np.real(np.fft.ifft(MH_00_k))
#
#    MH_1_k = np.fft.fft(MW_1) * W_k_an
#    MH_1 = np.real(np.fft.ifft(MH_1_k))
#
#    MH_2_k = np.fft.fft(MW_2) * W_k_an
#    MH_2 = np.real(np.fft.ifft(MH_2_k))
#
#    CH_1 = MH_1 / MH_0
#    v_pec = CH_1 / (m * a)
#
#    d_l = MH_00
#    v_l = v_pec #((MH_1 / (MH_0)) / a ) * (2 * norm)
#    dv_l = spectral_calc(v_l, k, o=1, d=0) * np.sqrt(a) / H0
#
#    # d_l = kde_gaussian(q, pos, sm, L)
#    # v_l = np.real(np.fft.ifft(np.fft.fft(vel) * W_k_an))
#    # dv_l = spectral_calc(v_l, k_nbody, o=1, d=0) * np.sqrt(a) / H0
#
#    break
#    # print(v_l[int(psi.size / 2)])
#    # tau.append(v_l[int(psi.size / 2)])
#    # a_list.append(a)
#
# from scipy.optimize import curve_fit
# from scipy.interpolate import interp1d
# def fitting_function(X, a0, a1, a2):
#    x1, x2 = X
#    return a0 + a1*x1 + a2*x2
#
# a0, a1, a2 = curve_fit(fitting_function, (d_l, dv_l), MH_2)[0]
# fit = fitting_function((d_l, dv_l), a0, a1, a2)
#
# # interp1d(x, tau)
# print(a0, a1, a2)
#
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.set_title(r'a = {}'.format(str(np.round(a, 6))))
# ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
# ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
# # ax.set_xlabel('a', fontsize=12)
# # ax.set_ylabel(r'$[\tau]_{\Lambda} \; [\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\left(\frac{\mathrm{km}}{\mathrm{s}}\right)^{2}] $', fontsize=12)
# # ax.set_ylabel(r'$M^{(0)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}]$', fontsize=12)
# # ax.set_ylabel(r'$M^{(1)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=12)
# # ax.set_ylabel(r'$M^{(2)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\left(\mathrm{km}\;\mathrm{s}^{-1}\right)^{2}]$', fontsize=12)
# # ax.set_ylabel(r'$\frac{M^{(1)}_{H}}{M^{(0)}_{H}}\,[\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=12)
#
# # ax.scatter(x, fit, s=10, c='r', label='fit')
# # ax.plot(x, MH_2, c='b', ls='dashed', lw=2, label='Sch')
# ax.plot(x, (d_l) / np.max(d_l), c='r', label='den')
# ax.plot(x, dv_l / np.max(dv_l), c='k', label='vel')
# ax.plot(x, (MH_2 - np.mean(MH_2)) / np.max(MH_2), c='b', label=r'$\tau$')
# # ax.plot(x, v_l, c='cyan', label=r'$v_{l}$')
# # ax.plot(a_list, tau, c='b', label=r'$v_{l}(x=0)$')
# plt.legend()
#
# print('saving...')
# plt.savefig(loc2 + 'plots/EFT/3fields_hfix_10.png', bbox_inches='tight', dpi=120)
# plt.close()
# tn = time.time()
# print('This took {}s'.format(np.round(tn-t0, 3)))
#

## Old SPT code
# n = 3
# F = spt_read_from_hdf5(filename_SPT)
# G = np.empty(F.shape)
# for j in range(len(F)):
#    G[j] = spectral_calc(F[j], k, o=1, d=1)

# den_spt = SPT_final(F, a)[n-1]
# dk_spt = np.fft.fft(den_spt)
# dk_spt *= W_k_an

# d_l = np.real(np.fft.ifft(dk_spt))
# cs2 = 1e-5
# cbv2 = 1e-5
# csv2 = 1e-5
# rho_b = 27.755
# gamma = 2.8 #5 / 3

# v_spt = SPT_final(G, a)[n-1]
# v_spt *= -H0 / np.sqrt(a)
# vk_spt = np.fft.fft(v_spt) * W_k_an
# v_l = np.real(np.fft.ifft(vk_spt))
# dv_l = spectral_calc(v_l, k, o=1, d=0) * np.sqrt(a) / H0

# def fitting_function(X, gamma, cs2, cbv2, csv2):
#    x1, x2 = X
#    a0 = cs2 / gamma
#    a1 = cs2
#    a2 = (cbv2 - (csv2/4))
#    return a0 + a1*x1 + a2*x2
#
# gamma, cs2, cbv2, csv2 = curve_fit(fitting_function, (d_l, dv_l), MH_2)[0]
# fit = fitting_function((d_l, dv_l), gamma, cs2, cbv2, csv2)
#
# print(gamma, cs2, cbv2, csv2)
