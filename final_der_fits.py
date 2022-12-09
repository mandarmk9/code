#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 20
a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
dv_l *= -np.sqrt(a) / 100
tau_l -= np.mean(tau_l)


def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
    def dir_der_o1(X, tau_l, ind):
        """Calculates the first-order directional derivative of tau_l along the vector X."""
        x1 = np.array([X[0][ind], X[1][ind]])
        x2 = np.array([X[0][ind+1], X[1][ind+1]])
        v = (x2 - x1)
        D_v_tau = (tau_l[ind+1] - tau_l[ind]) / v[0]
        return v, D_v_tau

    def dir_der_o2(X, tau_l, ind):
        """Calculates the second-order directional derivative of tau_l along the vector X."""
        v0, D_v_tau0 = dir_der_o1(X, tau_l, ind-2)
        v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
        v2, D_v_tau2 = dir_der_o1(X, tau_l, ind+2)
        x0 = np.array([X[0][ind-2], X[1][ind-2]])
        x1 = np.array([X[0][ind], X[1][ind]])
        x2 = np.array([X[0][ind+2], X[1][ind+2]])
        v = (x2 - x1)
        D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
        return v, D2_v_tau

    X = np.array([dc_l, dv_l])
    params_list = []
    for j in range(-dist//2, dist//2 + 1):
        v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
        v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
        dc_0, dv_0 = dc_l[ind], dv_l[ind]
        C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
        # C_ = [(tau_l[ind]), dtau1, dtau1_o2/2]

        params_list.append(C_)

    params_list = np.array(params_list)
    C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
    C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
    C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))

    C_ = [C0_, C1_, C2_]
    return C_

# fig, ax = plt.subplots()
dist = 50
n_sub = 1
C_list = []
# j = 4
start, thresh = [5000, 2000]
N = x.size
sub = np.linspace(start, N-start+1, n_sub, dtype=int)
del_ind = np.argmax(tau_l)
for point in sub:
    if del_ind-thresh < point < del_ind+thresh:
        sub = np.delete(sub, np.where(sub==point)[0][0])
    else:
        pass

ind_0 = np.argmin(dc_l**2 + dv_l**2)
print(tau_l[ind_0])
sub = np.array([ind_0])#, ind_0-10000])
n_sub = sub.size
# delta = 0.0001
# fac = 2
# sub_dc = np.arange(dc_l.min()/fac, dc_l.max()/fac, delta)
# sub_dv = np.arange(dv_l.min()/fac, dv_l.max()/fac, delta)
# n_sub = np.minimum(sub_dc.size, sub_dv.size)


for j in range(n_sub):
    # dc_0, dv_0 = sub_dc[j], sub_dv[j]
    # ind = np.argmin((dc_l - dc_0)**2 + (dv_l - dv_0)**2)
    tau_val = tau_l[sub[j]]
    tau_diff = np.abs(tau_l - tau_val)
    ind_tau = np.argmin(tau_diff)
    dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
    ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
    C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
    C_list.append(C_)
    # ax.scatter(x[sub[j]], tau_l[sub[j]], color='seagreen', s=20)
    # ax.scatter(x[ind], tau_l[ind], color='seagreen', s=20)

guesses = 1, 1, 1

def fitting_function(X, a0, a1, a2):
    x1 = X
    return a0 + a1*x1 + a2*(x1**2)
C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)


fit = C[0] + C[1]*dc_l + C[2]*dc_l**2

C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
C_ = [C0_, C1_, C2_]
fit_fde = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

print('from fit: ', C)
print('from FDE: ', C_)


def calc_fit(params, dc_l, dv_l, tau_l):
    start, thresh = params
    dist = 50
    n_sub = 1
    C_list = []
    N = dc_l.size
    sub = np.linspace(start, N-start+1, n_sub, dtype=int)
    del_ind = np.argmax(tau_l)
    for point in sub:
        if del_ind-thresh < point < del_ind+thresh:
            sub = np.delete(sub, np.where(sub==point)[0][0])
        else:
            pass
    n_sub = sub.size

    for j in range(n_sub):
        tau_val = tau_l[sub[j]]
        tau_diff = np.abs(tau_l - tau_val)
        ind_tau = np.argmin(tau_diff)
        dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
        ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
        C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
        C_list.append(C_)

    C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
    C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
    C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
    C_ = [C0_, C1_, C2_]
    fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

    return C_, fit

# C_, l = calc_fit((start,thresh), dc_l, dv_l, tau_l)
# fit2 = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots()
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')
# ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)

ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)

plt.plot(x, tau_l, c='b', label=r'measured')
plt.plot(x, fit, c='r', ls='dashed', label='fit')
plt.plot(x, fit_fde, c='k', ls='dashed', label='FDE')
plt.scatter(x[ind], tau_l[ind], s=20, c='seagreen')
# plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')


# plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
plt.show()

# guesses = 1, 1, 1
#
# def fitting_function(X, a0, a1, a2):
#     x1 = X
#     return a0 + a1*x1 + a2*(x1**2)
# C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
#
#
# fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
#
# print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])
# print('C1_deriv = ', C_[1], 'C1_fit = ', C[1])
# print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])
# fit2 = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
# # ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
#
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
#
# plt.plot(x, tau_l, c='b', label=r'measured')
# plt.plot(x, fit, c='r', ls='dashed', label='fit')
# plt.plot(x, fit2, c='k', ls='dashed', label='using derivatives')
# # plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')
#
# point = ind #sub[j]
# plt.scatter(x[point], tau_l[point], color='seagreen', s=20)
#
# # for point in sub:
# #     plt.scatter(x[point], tau_l[point], color='seagreen', s=20)
#
# # plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
# plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/test.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.close()
# # plt.close()



# C_list = []
# # sub = sub_find(n_sub, dc_l.size)
#
# # # n_sub = 5
# X = 40
# delta = 0.01
# sub = np.arange(dc_l.min()/2.5, dc_l.max()/2.5, delta)
# n_sub = sub.size
#
# delta = 0.001
# sub_dc = np.arange(dc_l.min(), dc_l.max(), delta)
# sub_dv = np.arange(dv_l.min(), dv_l.max(), delta)
# n_sub = np.minimum(sub_dc.size, sub_dv.size)
#
# # for j in range(n_sub):
# #     print(j)
# #     ind = np.argmin((dc_l - dc_0)**2 + (dv_l - dv_0)**2)
#
# # j = 0
# fits, fits_fit = [], []
# for j in range(2, n_sub-2):
#     # dc_0, dv_0 = np.repeat(sub[j], 2)
#     dc_0, dv_0 = sub_dc[j], sub_dv[j]
#     ind = np.argmin((dc_l - dc_0)**2 + (dv_l - dv_0)**2)
#     C_list.append(new_param_calc(dc_l, dv_l, tau_l, dist, ind))
#
# guesses = 1, 1, 1
#
# def fitting_function(X, a0, a1, a2):
#     x1 = X
#     return a0 + a1*x1 + a2*(x1**2)
# C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
#
#
# fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
#
# C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
# C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
# C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
# C_ = [C0_, C1_, C2_]
# fit2 = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
#
# print(C)
# print(C_)
#
# plt.plot(x, tau_l, c='b')
# plt.plot(x, fit, c='k', ls='dashdot')
# plt.plot(x, fit2, c='r', ls='dashed')
# plt.savefig('../plots/test/new_paper_plots/der_fits.png', bbox_inches='tight', dpi=150)
# plt.close()
# # plt.show()


# sub = [np.where(dc_l == points[j])[0] for j in range(points.size)]
# n_sub = len(sub)

# sub = (np.arange(X, 62500-X, 1))[::62500//10]#np.arange(0, x.size, 1)
# n_sub = sub.size
# print(n_sub)
#
# # sub = [0, x.size//2, x.size//3]
# # n_sub = len(sub)
# # for j in range(n_sub):
# #     tau_val = tau_l[j]
# #     print(tau_val)
# #     tau_diff = np.abs(tau_l - tau_val)
# #     ind_tau = np.argmin(tau_diff)
# #     dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
# #     ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
# #     # C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
# #     C_list.append(new_param_calc(dc_l, dv_l, tau_l, dist, ind))
# #
# # C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
# # C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
# # C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
# # C_ = [C0_, C1_, C2_]
#
# ind_0 = np.argmin(np.abs(tau_l))
# ind_peak = np.argmax(tau_l)
# sub = [2500, 12500, 20000, 30000, 40000, 50000]#[ind_0-5000, ind_0, ind_peak-500]
# n_sub = len(sub)


    # tau_val = tau_l[sub[j]]
    # # print(tau_val)
    # tau_diff = np.abs(tau_l - tau_val)
    # ind_tau = np.argmin(tau_diff)
    # dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
    # ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
    # C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)

    # guesses = 1, 1, 1
    #
    # def fitting_function(X, a0, a1, a2):
    #     x1 = X
    #     return a0 + a1*x1 + a2*(x1**2)
    # C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    #
    #
    # fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
    #
    # print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])
    # print('C1_deriv = ', C_[1], 'C1_fit = ', C[1])
    # print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])
    # fit2 = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
    #
    # fits.append(fit2)
    # fits_fit.append(fit)

    # plt.rcParams.update({"text.usetex": True})
    # plt.rcParams.update({"font.family": "serif"})
    # fig, ax = plt.subplots()
    # ax.minorticks_on()
    # ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    # ax.yaxis.set_ticks_position('both')
    # # ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
    # ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
    #
    # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
    # ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)
    #
    # plt.plot(x, tau_l, c='b', label=r'measured')
    # plt.plot(x, fit, c='r', ls='dashed', label='fit')
    # plt.plot(x, fit2, c='k', ls='dashed', label='using derivatives')
    # # plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')
    #
    # point = ind #sub[j]
    # plt.scatter(x[point], tau_l[point], color='seagreen', s=20)
    #
    # # for point in sub:
    # #     plt.scatter(x[point], tau_l[point], color='seagreen', s=20)
    #
    # # plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
    # plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
    # # plt.show()
    # plt.close()
    # # plt.savefig('../plots/test/new_paper_plots/test.png'.format(kind), bbox_inches='tight', dpi=150)
    # # plt.close()

# fit2 = sum(np.array(fits)) / len(fits)
# fit = sum(np.array(fits_fit)) / len(fits_fit)
