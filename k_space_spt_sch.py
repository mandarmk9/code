#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, dn
from EFT_solver import EFT_sm_kern, smoothing
from SPT import *

# plt.style.use('clean_1d')

loc = '../'
run = '/sch_hfix_run17/'
Nfiles = 379

Lambda = 5
j = 0
with h5py.File(loc + 'data' + run + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
   ls = list(hdf.keys())
   A = np.array(hdf.get(str(ls[0])))
   a = np.array(hdf.get(str(ls[1])))
   L, h, m, H0 = np.array(hdf.get(str(ls[2])))
   psi = np.array(hdf.get(str(ls[3])))
   print('a = ', a)

Nx = psi.size
dx = L / Nx
x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
rho_0 = 27.755
sigma_x = 2.5 * dx
sm = 1 / (4 * (sigma_x**2))
W_k_an = np.exp(- (k ** 2) / (4 * sm))
W_EFT = EFT_sm_kern(k, Lambda)

from zel import eulerian_sampling as es
nd = es(x, a, A)
q = nd[0]
dc = nd[1]

d0 = (A[0] * np.cos(A[1]*x)) + (A[2] * np.cos(A[3]*x))
d0_bar = d0#smoothing(d0, W_EFT*W_k_an)

# n = 3
# F = SPT_agg(n, x, s=1)
# # write_to_hdf5(filename_SPT, F)
# F = np.real(np.fft.fft(F)) / Nx


# print(F)

F = dn(5, k, d0_bar)
F = np.real(np.fft.fft(F)) / Nx

# spt1 = F[0][2]
# spt2 = F[1][2]
# spt3 = F[2][2]
# spt4 = F[3][2]
# spt5 = F[4][2]
# # spt6 = F[5][2]

# plt.scatter(k, F[2], c='cyan', s=20, ls='dashed')
# plt.scatter(k, F[3], c='violet', s=20, ls='dashed')
# plt.scatter(k, F[4], c='orange', s=20, ls='dashed')

plt.scatter(k, F[0], c='b', s=50)
plt.scatter(k, F[1], c='r', s=50)
plt.scatter(k, F[2], c='k', s=50)
plt.scatter(k, F[3], c='brown', s=50)
plt.scatter(k, F[4], c='cyan', s=50)
plt.show()
# print("/n")
# print('spt1 = ', spt1)
# print('spt2 = ', spt2)
# print('spt3 = ', spt3)
# print('spt4 = ', spt4)
# print('spt5 = ', spt5)
# print('spt6 = ', spt6)


# dk_spt1 = np.fft.fft(spt1) / Nx
# dk_spt2 = np.fft.fft(spt2) / Nx
# dk_spt3 = np.fft.fft(spt3) / Nx
# dk_spt4 = np.fft.fft(spt4) / Nx
#
#
# # G = SPT_final(F, a)
# # dk_spt1 = np.fft.fft(G[0]) / Nx
# # dk_spt2 = np.fft.fft(G[1]) / Nx
# # dk_spt3 = np.fft.fft(G[2]) / Nx
# # dk_spt4 = np.fft.fft(G[3]) / Nx
# #
#
# print(np.real(dk_spt1[1:5]))
# print(np.real(dk_spt2[1:5]))
# print(np.real(dk_spt3[1:5]))
# print(np.real(dk_spt4[1:5]))


#naive SPT computation for n = 2, based on discretisation in Fourier space
#first, the necessary fields
# d1 = np.fft.fft(d0_bar) / Nx
# d2 = np.zeros(d1.size)
#
# # for i in range(1, Nx):
# #    q_i = int(k[i])
# #    for j in range(1, Nx):
# #       q_j = int(k[j])
# #       if -Nx/2 - 1 < q_i + q_j < Nx/2:
# #          rhs = (1 + (q_i/(2*q_j)) + (q_j/(2*q_i))) * (d1[q_i] * d1[q_j])
# #          d2[q_i + q_j] += rhs
#
# dk_spt2_grid = (d1 * a) + (d2 * a**2)
# dk2_spt2_grid = np.abs(dk_spt2_grid)**2
#
# F = dn(3, k, d0_bar)
# spt_sol = SPT_final(F, a)
#
# F_ns = dn(3, k, d0)
# spt_sol_ns = SPT_final(F_ns, a)
#
# # d1 = d0_bar
# # d2 = -(spectral_calc(d1, k, d=0, o=1) * spectral_calc(d1, k, d=1, o=1)) + d1**2
# # dx_2 = (a * d1) + (a**2 * d2)
# # dk_2spt_new = np.fft.fft(dx_2) / Nx
# # dk2_2spt_new = np.abs(dk_2spt_new)**2
#
# # dk_zel = np.fft.fft(nd[1]) / Nx #* W_k_an * W_EFT
# # dk_1spt = np.fft.fft(spt_sol[0]) / Nx
# dk_2spt = np.fft.fft(spt_sol[1]) / Nx
#
# dk_2spt_ns = np.fft.fft(spt_sol_ns[1]) / Nx
#
# # dk_3spt = np.fft.fft(spt_sol[2]) / Nx
# # dk_6spt = np.fft.fft(spt_sol[5]) / Nx
#
# # zel = np.abs(dk_zel)**2
# # spt1 = np.abs(dk_1spt)**2
# spt2 = np.abs(dk_2spt)**2
# spt2_ns = np.abs(dk_2spt_ns)**2
#
# # spt3 = np.abs(dk_3spt)**2
# # spt6 = np.abs(dk_6spt)**2
#
# # d1 = d0_bar
# # d2 = d1**2 + (A[0]*A[1] * np.sin(A[1] * x) + A[2]*A[3] * np.sin(A[3]*x)) * ((A[0]/A[1])*np.sin(A[1]*x) + (A[2]/A[3]*np.sin(A[3]*x)))
# # spt_new = (d1 * a) + (d2 * (a**2))
# # dk2_spt_new2 = np.abs(np.fft.fft(spt_new) * dx)**2
#
# MW_0 = np.abs(psi ** 2)
# dk_sch = np.fft.fft(MW_0 - 1) / Nx * W_k_an * W_EFT
# dk_sch_ns = np.fft.fft(MW_0 - 1) / Nx #* W_k_an * W_EFT
#
# sch = np.abs(dk_sch)**2
# sch_ns = np.abs(dk_sch_ns)**2
#
# mode = 2
# err_sm = (spt2[mode] - sch[mode]) * 100 / sch[mode]
# err_ns = (spt2_ns[mode] - sch_ns[mode]) * 100 / sch_ns[mode]
#
# print(err_sm)
# print(err_ns)
#
# fig, ax = plt.subplots()
# ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
# ax.set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
# # ax.scatter(k, sch, label='Sch', s=50, c='k')
# # ax.scatter(k, sch_ns, label='Sch_ns', s=25, c='b')
# ax.scatter(k, spt2, label='2SPT', s=50, c='r', ls='dashed')
# ax.scatter(k, spt2_ns, label='2SPT_ns', s=25, c='cyan', ls='dashed')
# # ax.scatter(k, dk2_2spt_new, label='2SPT_grid', s=40, c='brown')
# # ax.scatter(k, err_sm, c='k', s=50)
# # ax.scatter(k, err_ns, c='b', s=20)
#
# # ax.scatter(k, spt6, label='6SPT', s=10, c='green')
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
#
# ax.legend(fontsize=14, loc=2, bbox_to_anchor=(1,1))
# # ax.set_ylim(-1e-9, 6e-9)
# # ax.set_xlim(-0.5, 4.5)
# plt.show()
#
# # err_spt = (spt2[2] - zel[2]) * 100 / zel[2]
# # print(err_spt)
# #
# # plt.savefig('plots/run17/modes/SPT_k_{}_l_{}.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# #
