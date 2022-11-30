#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, spectral_calc
from scipy.optimize import curve_fit
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 4 * (2*np.pi)

kind = 'sharp'
kind_txt = 'sharp cutoff'

kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
mode = 1
j = 0

# a_list, tau_list, param_list, ctot_list = [], [], [], []
#
# for j in range(51):
#     path = path[:-2] + '{}/'.format(1)
#
#     a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, '')
#     taus = []
#     taus.append(tau_l_0)
#     for run in range(2, 9):
#         ind = -2
#         path = path[:ind] + '{}/'.format(run)
#         taus.append(read_sim_data(path, Lambda, kind, j, '')[1])
#
#     taus = np.array(taus)
#     Nt = len(taus)
#
#     tau_l = sum(np.array(taus)) / Nt
#
#     print('a = ', a)
#
#     rho_0 = 27.755
#     H0 = 100
#     rho_b = rho_0 / a**3
#
#     # M&W Estimator
#     Lambda_int = int(Lambda / (2*np.pi))
#     num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
#     denom = ((d1k * np.conj(d1k)) * (a**2))
#     ntrunc = int(num.size-Lambda_int)
#     ctot2_2 = np.real(sum(num[:Lambda_int+1]) / sum(denom[:Lambda_int+1])) / rho_b
#
#     a_list.append(a)
#     tau_k = np.real(sum(np.fft.fft(tau_l)[:4])/tau_l.size)
#     tau_list.append(tau_k)
#     ctot_list.append(ctot2_2)
#
#     if j > 4:
#         tlist = np.array(tau_list)
#         alist = np.array(a_list)
#         X = alist
#         def fitting_function(X, a0, a1):
#             return a0*(X**(a1))
#
#         guesses = 1, 1
#         C, cov = curve_fit(fitting_function, X, tlist, sigma=np.ones(X.size), method='lm', absolute_sigma=True)
#         fit = fitting_function(X, C[0], C[1])
#         # param_list.append(C[1])
#         param_list.append(fit)
#
#
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
#
# # plt.plot(a_list, tau_list, c='b', lw=2, label='measured')
# # plt.plot(a_list, fit, c='r', ls='dashed', lw=2, label='fit, index = {}'.format(C[1]))
# ax.plot(a_list[5:], param_list, lw=2, c='b', label='index', marker='o')
# ax.plot(a_list, ctot_list, lw=2, c='r', marker='+')
#
# # plt.axhline(-1, lw=1, ls='dashed', c='k')
# ax.axvline(1.81818, lw=1, ls='dashed', c='k', label=r'$a_{\mathrm{sc}}$')
# ax.axvline(4.35, lw=1, ls='dashed', c='k')
#
# # ax.plot(a_list, np.array(tau_list), c='b', lw=2, label='measured')
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.yaxis.set_ticks_position('both')
# ax.set_xlabel(r'$a$', fontsize=14)
# ax.set_ylabel(r'$\lambda(a)$', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
#
# # plt.savefig('../plots/test/new_paper_plots/slope_ens.png', bbox_inches='tight', dpi=120)
# # plt.close()

a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, '')
rho_0 = 27.755
H0 = 100
rho_b = rho_0 / a**3

# Baumann estimator
def Power(f1, f2):
    f1_k = np.fft.fft(f1)
    f2_k = np.fft.fft(f2)

    corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    return np.real(np.fft.ifft(corr))

A = spectral_calc(tau_l, 1, o=2, d=0) / rho_b / (a**2)
T = -dv_l / (H0 / (a**(1/2)))
P_AT = Power(A, T)
P_dT = Power(dc_l, T)
P_Ad = Power(A, dc_l)
P_TT = Power(T, T)
P_dd = Power(dc_l, dc_l)

num_cs2 = (P_AT * spectral_calc(P_dT, 1, o=2, d=0)) - (P_Ad * spectral_calc(P_TT, 1, o=2, d=0))
den_cs2 = ((spectral_calc(P_dT, 1, o=2, d=0))**2 / (a**2)) - (spectral_calc(P_dd, 1, o=2, d=0) * spectral_calc(P_TT, 1, o=2, d=0) / a**2)

num_cv2 = (P_Ad * spectral_calc(P_dT, 1, o=2, d=0)) - (P_AT * spectral_calc(P_dd, 1, o=2, d=0))
cs2_3 = num_cs2 / den_cs2
cv2_3 = num_cv2 / den_cs2
# ctot2_3 = np.median(np.real(cs2_3 + cv2_3))
ctot2_3 = np.real(cs2_3 + cv2_3)

inds = np.where(np.abs(den_cs2) > 50000)
print(inds[0].size)
ctot = ctot2_3[inds]

plt.plot(x[inds], ctot)#/(np.std(T)**2))
plt.show()
#
#
# def Power_fou(f1, f2, mode):
#     f1_k = np.fft.fft(f1)
#     f2_k = np.fft.fft(f2)
#     corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
#     return corr[mode]
#
# ctot2 = np.real(Power_fou(tau_l/rho_b, dc_l, mode) / Power_fou(dc_l, T, mode))

# print(ctot2, ctot2_2, ctot2_3)

    # # Baumann estimator
    # def Power(f1_k, f2_k, Lambda_int):
    #   corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    #   ntrunc = corr.size - Lambda_int
    #   corr[Lambda_int+1:ntrunc] = 0
    #   return corr
    #
    # A = np.fft.fft(tau_l) / rho_b / tau_l.size
    # T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
    # d = dc_l#np.fft.fft(dc_l) / dc_l.size
    # # A = tau_l / tho_b
    # # T = dv_l / (H0 / (a**(1/2)))
    #
    # # Ad = Power(A, dc_l, Lambda_int)[mode]
    # # AT = Power(A, T, Lambda_int)[mode]
    # # P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
    # # P_TT = Power(T, T, Lambda_int)[mode]
    # # P_dT = Power(dc_l, T, Lambda_int)[mode]
    #
    #
    # Ad = Power(A, d, Lambda_int)[mode]
    # AT = Power(A, T, Lambda_int)[mode]
    # P_dd = Power(d, d, Lambda_int)[mode]
    # P_TT = Power(T, T, Lambda_int)[mode]
    # P_dT = Power(d, T, Lambda_int)[mode]
    #
    # print('\na = ', a)
    # try:
    #     cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
    #     cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
    #
    #     N1 = ((P_TT * Ad) - (P_dT * AT))
    #     N2 = ((P_dT * Ad) - (P_dd * AT))
    #     D = (P_dd * P_TT - (P_dT)**2)
    #
    #     print(np.real(N1+N2), np.real(D))
    #     ctot2_3 = np.real(cs2_3 + cv2_3)
    #     print(ctot2_2, ctot2_3)
    #
    # except:
    #     print(ctot2_2, 0)
    #     pass
