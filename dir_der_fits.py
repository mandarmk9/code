#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data, param_calc_ens
from scipy.interpolate import interp1d, interp2d, griddata
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 27
for j in range(j, j+1):
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
            # print(D_v_tau)
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
            params_list.append(C_)

        params_list = np.array(params_list)
        dist = params_list.shape[0]
        if dist != 0:
            C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
            C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
            C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))
            C_ = [C0_, C1_, C2_]
        else:
            C_ = [0, 0, 0]
        return C_

    dist = 1
    ind = np.argmin(dc_l**2 + dv_l**2)
    distance = np.sqrt(dc_l**2 + dv_l**2)
    per = np.percentile(distance, 1)
    indices = np.where(distance < per)[0]
    # # new_distance = distance[distance < per]
    # # plt.hist(distance, color='r')
    # # plt.hist(new_distance, color='b')
    # #
    # # plt.axvline(per, color='k')
    # plt.show()
    # C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)

    params_list = []
    for ind in indices:
        C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
        params_list.append(C_)


    C0_ = np.median([params_list[j][0] for j in range(len(params_list))])
    C1_ = np.median([params_list[j][1] for j in range(len(params_list))])
    C2_ = np.median([params_list[j][2] for j in range(len(params_list))])
    C_ = [C0_, C1_, C2_]
    ind = indices

    # npoints = 50
    # inds = []
    # dtau = np.abs(tau_l-0.4)
    # ind = np.argmin(dtau)
    # dc_0, dv_0 = dc_l[ind], dv_l[ind]#np.repeat(dc_l[np.argmax(tau_l)-1000], 2)
    # ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
    # # # ind = np.argmax(tau_l) - 500 #np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)

    # # print(dc_l[ind], tau_l[ind])
    # ind = np.argmin((dc_l)**2 + (dv_l)**2)
    # for gap in range(npoints):
    #     gap *= dc_l.size//npoints
    #     inds.append(int(ind-gap))
    # # inds = [ind]
    #
    # for ind in inds:
    #     C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
    #     params_list.append(C_)
    #
    #
    # print(points)

    # ind = np.argmax(dc_l)
    # print(dc_l.max(), dc_l[ind-5:ind+5])

    # # C0_, C1_, C2_, C3_, C4_, C5_ = C_

    # tau_l = 11*dc_l - 7*dc_l + 15*dc_l**2 + dv_l**3 + 2*dv_l**2 + 1.55 #dc_l**2 + dv_l**2
    # C_ = new_param_calc(dc_l, dv_l, tau_l, dist)
    # # print(C_)

    # ind = np.argmin(dc_l**2 + dv_l**2)
    # # print(dc_l[ind], dv_l[ind])
    # v1, dir1 = dir_der_o1((dc_l, dv_l), tau_l, ind, h=1)
    # v2, dir2 = dir_der_o1((dc_l, dv_l), tau_l, ind+1, h=1)
    #
    # print(v1, dir1)
    # print(v2, dir2)

    # v, dir2 = dir_der_o2((dc_l, dv_l), tau_l, ind, h=1)
    # print(dir2)


    # print(new_param_calc(dc_l, dv_l, tau_l, dist))
    # ind = np.argmin(dc_l**2 + dv_l**2)
    # interp_list = []
    # for j in range(-dist, dist):
    #     interp_list.append([dc_l[ind+j], dv_l[ind+j], tau_l[ind+j]])
    #
    # dc_list = np.array([interp_list[j][0] for j in range(len(interp_list))])
    # dv_list = np.array([interp_list[j][1] for j in range(len(interp_list))])
    # tau_list = np.array([interp_list[j][2] for j in range(len(interp_list))])
    # points = (dc_list, dv_list)
    # values = tau_list
    # dc_grid = np.arange(-0.01, 0.01, 1e-3)
    # dv_grid = np.arange(-0.01, 0.01, 1e-3)
    # xi = (dc_grid, dv_grid)
    # tau_grid = griddata(points, values, xi, method='nearest')
    # # print(dc_grid, dv_grid, tau_grid)
    # C_ = new_param_calc(dc_grid, dv_grid, tau_grid, 1)
    # C0_, C1_, C2_, C3_, C4_, C5_ = C_

    # tau_func = interp2d(dc_list, dv_list, tau_list, fill_value='extrapolate')

    # C0_ = tau_func(0, 0)

    # ind = np.argmin(dc_l**2 + dv_l**2)
    # j = 0
    # X = np.array([dc_l, dv_l])
    # h = 1e-3
    # v1_o2, dir1_o2 = dir_der_o1(X, tau_l, ind+j, h)
    # v2_o2, dir2_o2 = dir_der_o2(X, tau_l, ind+j+1, h)
    # v3_o2, dir3_o2 = dir_der_o2(X, tau_l, ind+j-1, h)
    #
    # f1 = (v1_o2[0]**2 * v3_o2[1]**2) - (v1_o2[1]**2 * v3_o2[0]**2)
    # f2 = (v1_o2[0]**2 * v2_o2[1]**2) - (v1_o2[1]**2 * v2_o2[0]**2)
    # g1 = (v1_o2[0]**2 * v2_o2[0] * v2_o2[1]) - (v2_o2[0]**2 * v1_o2[0] * v1_o2[1])
    # g2 = (v1_o2[0]**2 * v3_o2[0] * v3_o2[1]) - (v3_o2[0]**2 * v1_o2[0] * v1_o2[1])
    #
    # tau_xy = (((f1 * v1_o2[0]**2 * dir2_o2) - (f2 * v1_o2[0]**2 * dir3_o2)) - (dir1_o2 * (f1*v2_o2[0]**2 - f2*v3_o2[0]**2))) / (f1*g1 - f2*g2)
    # tau_yy = (((v1_o2[0]**2 * dir2_o2) - (v2_o2[0]**2 * dir1_o2)) - tau_xy*g1) / f1
    # tau_xx = (dir1_o2 - (v1_o2[1]**2 * tau_yy) - (v1_o2[0]*v1_o2[1]*tau_xy)) / (v1_o2[0]**2)
    # C3_, C4_, C5_ = tau_xx, tau_yy, tau_xy


    # guesses = 1, 1, 1
    # def fitting_function(X, a0, a1, a2):
    #     x1, x2 = X
    #     return a0 + a1*x1 + a2*x2
    #
    # C, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    # C0, C1, C2 = C
    # fit = fitting_function((dc_l, dv_l), C0, C1, C2)

    # guesses = 1, 1, 1, 1, 1, 1
    # guesses = 0, 0, 0, 0, 0, 0
    #
    # def fitting_function(X, a0, a1, a2, a3, a4, a5):
    #     x1, x2 = X
    #     return a0 + a1*x1 + a2*x2 + a3*(x1**2) + a4*(x2**2) + a5*(x1*x2)
    #
    # C_calc, cov = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    # C = np.array(C)
    guesses = 1, 1, 1

    def fitting_function(X, a0, a1, a2):
        x1 = X
        return a0 + a1*x1 + a2*(x1**2)
    C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)

    # C = [C_calc[0], C_calc[1]+C_calc[2], C_calc[3]+C_calc[4]+C_calc[5]]
    # fit = fitting_function((dc_l, dv_l), C0, C1, C2, C3, C4, C5)

    fit = C[0] + C[1]*dc_l + C[2]*dc_l**2


    # est = fitting_function((dc_l, dv_l), C0_, C1_, C2_, C3_, C4_, C5_)


    # # C1_ /= 1.5
    # # C2_ /= 1.5
    # def fitting_function(X, a0, a1, a2):
    #     x1, x2 = X
    #     return a0 + a1*x1 + a2*x2
    #
    # # fit = fitting_function((dc_l, dv_l), C0, C1, C2)
    # # est = fitting_function((dc_l, dv_l), C0_, C1_, C2_)


    # # C_[2] *= -0.5
    # fac = (C[3]+C[4]+(2*C[5])) / C_[2]
    # C = [C[0], C[1]+C[2], C[3]+C[4]+(2*C[5])]
    # C_[2] *= fac
    # print('C_fit = ', C)
    # print('C_der = ', C_)
    # est = C_[0] + C_[1]*(dc_l) #+ C_[2]*(dc_l**2)
    # fit = C[0] + C[1]*(dc_l) #+ C[2]*(dc_l**2)

    # fit = C[0] + C[1]*dc_l + C[2]*dv_l + C[3]*dc_l**2 + C[4]*dv_l**2 + C[5]*(dc_l*dv_l)
    # fit2 = C[0] + (C[1]+C[2])*(dc_l) + (C[3]+C[4]+(2*C[5]))*(dc_l**2)
    print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])

    # print('C1 = ', C[1]+C[2])
    print('C1_deriv = ', C_[1], 'C1_fit = ', C[1])

    # print('C2 = ', C[3]+C[4]+C[5])
    print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])

    fit_fde = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
    # fit = dc_l**2 + dv_l**2 + 2*dc_l*dv_l #dc_l + dv_l
    # fit2 = 4*dc_l**2 #2*dc_l

    # del_tau = fit2-tau_l
    # plt.rcParams.update({"text.usetex": True})
    # fig, ax = plt.subplots()
    # ax.scatter(tau_l, del_tau, c='b', s=2)
    # ax.set_xlabel(r'$\tau_{l}$')
    # ax.set_ylabel(r'$\Delta \tau_{l}$')
    # plt.show()
    # # plt.savefig('../plots/test/new_paper_plots/tau_diff.png', bbox_inches='tight', dpi=150)
    # # plt.close()

    # plt.plot(x, dc_l)
    # plt.show()

    # ind = np.argmin(dc_l**2 + dv_l**2)
    # print(fit[ind], fit2[ind])
    print(sum((tau_l - fit_fde)**2))

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
    # plt.scatter(x[indices], tau_l[indices], s=20, c='seagreen')
    plt.scatter(x[ind], tau_l[ind], s=20, c='seagreen')

    # plt.plot(x, fit3, c='cyan', ls='dotted', label='using derivatives 2')


    # plt.plot(x, est, c='k', ls='dashed', label='using derivatives')
    plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
    plt.show()
    # plt.savefig('../plots/test/new_paper_plots/fde_vs_fit_one_point/{}_tau_{}.png'.format(kind, j), bbox_inches='tight', dpi=150)
    # plt.close()
