#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

from EFT_nbody_solver import *
# from EFT_ens_solver import * #EFT_solve
# from EFT_solver_gauss import *

from scipy.optimize import curve_fit
from SPT import SPT_final
from functions import plotter, SPT_real_sm, SPT_real_tr, alpha_to_corr, read_sim_data, initial_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def param_calc(j, Lambda, path, A, mode, kind, nruns=4):
    a, x, d1k, dc_l, dv_l, tau_l_0 = read_sim_data(path, kind, j)

    taus = []
    taus.append(tau_l_0)
    for run in range(2, nruns+1):
        path = path[:-2] + '{}/'.format(run)
        taus.append(read_sim_data(path, kind, j)[1])

    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt
    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    def fitting_function(X, a0, a1, a2):
      x1, x2 = X
      return a0 + a1*x1 + a2*x2

    diff = np.array([(taus[i] - taus[0])**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / ((Nt-1)))
    # yerr = np.ones(tau_l.size) * 1e-15

    guesses = 1, 1, 1
    FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=yerr, method='lm', absolute_sigma=True)
    C0, C1, C2 = FF[0]
    cov = FF[1]
    err0, err1, err2 = np.sqrt(np.diag(cov))

    fit = fitting_function((dc_l, dv_l), C0, C1, C2)
    C = [C0, C1, C2]

    cs2 = np.real(C1 / rho_b)
    cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
    ctot2 = (cs2 + cv2)

    resid = fit - tau_l
    chisq = sum((resid / yerr)**2)
    red_chi = chisq / (x.size - 3)
    C = [C0, C1, C2]

    # print('\na = ', a)
    # print('chi_sq = ', chisq)
    # print('reduced chi_sq = ', red_chi)

    #to propagate the errors from C_i to c^{2}, we must divide by rho_b (the minus sign doesn't matter as we square the sigmas)
    err1 /= rho_b
    err2 /= rho_b

    total_err = np.sqrt(err1**2 + err2**2)
    # print(total_err)

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))[:Lambda_int]
    denom = ((d1k * np.conj(d1k)) * (a**2))[:Lambda_int]
    ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

    # Baumann estimator
    def Power(f1_k, f2_k):
      corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
      return corr

    A = np.fft.fft(tau_l) / rho_b / tau_l.size
    T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
    d = np.fft.fft(dc_l) / dc_l.size
    Ad = Power(A, dc_l)[mode]
    AT = Power(A, T)[mode]
    P_dd = Power(dc_l, dc_l)[mode]
    P_TT = Power(T, T)[mode]
    P_dT = Power(dc_l, T)[mode]

    num_cs = (P_TT * Ad) - (P_dT * AT)
    num_cv = (P_dT * Ad) - (P_dd * AT)
    den = ((P_dd * P_TT) - (P_dT)**2)
    cs2_3 = num_cs / den
    cv2_3 = num_cv / den
    ctot2_3 = np.real(cs2_3 + cv2_3)

    # print(P_TT, Ad, P_dT, AT, num_cs)
    print(P_TT*Ad, P_dT*AT, num_cs)
    print(num_cs, num_cv, den)
    print(cs2_3, cv2_3)
    print(ctot2, ctot2_2, ctot2_3)


    return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, total_err #cs2, cv2, red_chi



def ctot2_calc(path, Nfiles, Lambda, A, mode, kind):
    zero = 0
    H0 = 100

    #define lists to store the data
    a_list = np.zeros(Nfiles)
    ctot2_list = np.zeros(Nfiles)
    ctot2_list2 = np.zeros(Nfiles)
    ctot2_list3 = np.zeros(Nfiles)
    total_err = np.zeros(Nfiles)
    #initial scalefactor
    a0 = EFT_solve(0, Lambda, path, A, kind)[0]

    for file_num in range(zero, Nfiles):
       # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
       #the function 'EFT_solve' return solutions of all modes + the EFT parameters
       ##the following line is to keep track of 'a' for the numerical integration
       if file_num > 0:
          a0 = a


       a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, terr = param_calc(file_num, Lambda, path, A, mode, kind)#, nruns=8)
       total_err[file_num] = terr

       # a, x, ctot2, ctot2_2, ctot2_3 = param_calc(file_num, Lambda, path, A, mode, kind)


       print('a = {}\n'.format(a))
       # print('fit: ', ctot2)
       # print('MW: ', ctot2_2)
       # print('Bau: ', ctot2_3)

       a_list[file_num] = a
       ctot2_list[file_num] = ctot2
       ctot2_list2[file_num] = ctot2_2
       ctot2_list3[file_num] = ctot2_3

    a_sc = 1 / np.max(initial_density(x, A, L=1.0))
    return a_list, ctot2_list, ctot2_list2, ctot2_list3, a_sc, total_err


path = 'cosmo_sim_1d/final_phase_run1/'
Nfiles = 88
Lambda = 3 * (2 * np.pi)
mode = 1
A = [-0.05, 1, -0.5, 11]#, -0.01, 2, -0.01, 3, -0.01, 4]
kind = 'sharp'
kind_txt = 'sharp cutoff'
a_list, sharp1, sharp2, sharp3, a_sc, yerror_sharp = ctot2_calc(path, Nfiles, Lambda, A, mode, kind)
yaxes_sharp = [sharp3, sharp2, sharp1]

# header = ['a', 'Fit', 'M&W', 'Baumann']
# ctot = pd.DataFrame(np.transpose(np.array([a_list, sharp1, sharp2, sharp3])))
# print(ctot)
# ctot.to_csv('ens_sharp_ctot.csv', header=header, index=False)

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
# a_list, gauss1, gauss2, gauss3, a_sc, yerror_gauss = ctot2_calc(path, Nfiles, Lambda, A, mode, kind)
# yaxes_gauss = [gauss3, gauss2, gauss1]
# header = ['a', 'Fit', 'M&W', 'Baumann']
# ctot = pd.DataFrame(np.transpose(np.array([a_list, gauss1, gauss2, gauss3])))
# ctot.to_csv('ens_gauss_ctot.csv', header=header, index=False)



# def ctot2_read(file_path):
#     file = pd.read_csv(file_path)
#     return np.array(file['a']), np.array(file['Fit']), np.array(file['M&W']), np.array(file['Baumann'])
#
# a_list, sharp1, sharp2, sharp3 = ctot2_read('sharp_ctot.csv')
# a_list, gauss1, gauss2, gauss3 = ctot2_read('gauss_ctot.csv')
# yaxes_sharp = [sharp3, sharp2, sharp1]
# yaxes_gauss = [gauss3, gauss2, gauss1]
#
#
#
# xlabel = r'$a$'
# ylabel = r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$'
# colours = ['orange', 'cyan', 'k']
# labels = [r'$\mathrm{B^{+12}}$', 'M&W', r'from fit to $\tau_{l}$'] #$\left<\tau_{l}\right>$
# linestyles = ['solid', 'solid', 'solid']
# markers = ['v', '*', 'o'] #
#
# xaxis = a_list
# fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})
# ax[0].set_xlabel(xlabel, fontsize=18, x=1)
#
# ax[0].set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (Gaussian)'.format(int(Lambda/(2*np.pi))), fontsize=18)
# ax[1].set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (sharp)'.format(int(Lambda/(2*np.pi))), fontsize=18)
#
# # ax[0].errorbar(xaxis, yaxes_gauss[0], yerr=yerror_gauss, c=colours[0], ls=linestyles[0], lw=2, label=labels[0], marker='o')
#
# for i in range(1, len(yaxes_sharp)):
#     ax[0].plot(xaxis, yaxes_gauss[i], c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker=markers[i])
#     ax[1].plot(xaxis, yaxes_sharp[i], c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker=markers[i])
#
# # ax[0].plot(xaxis, yaxes_gauss[2], c=colours[2], ls=linestyles[2], lw=2, label=labels[2], marker='*')
# # ax[0].set_ylim(0, 1.5)
#
# # ax[1].errorbar(xaxis, yaxes_sharp[0], yerr=yerror_sharp, c=colours[0], ls=linestyles[0], lw=2, label=labels[0], marker='o')
# # ax[1].plot(xaxis, yaxes_sharp[1], c=colours[1], ls=linestyles[1], lw=2, label=labels[1], marker='v')
# # ax[1].plot(xaxis, yaxes_sharp[1], c=colours[2], ls=linestyles[2], lw=2, label=labels[2], marker='*')
#
#
# for i in range(2):
#     ax[i].set_ylabel(ylabel, fontsize=18)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13)
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].yaxis.set_ticks_position('both')
#     # ax[i].axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
#
# ax[1].yaxis.set_label_position('right')
#
# plt.subplots_adjust(wspace=0)
# ax[1].tick_params(labelleft=False, labelright=True)
# ax[1].legend(fontsize=13)#, loc=2, bbox_to_anchor=(1,1))
# # plt.savefig('../plots/test/ctot.png', bbox_inches='tight', dpi=150)
# # plt.show()
#
# plots_folder = 'test/final/'#/paper_plots'
# # plots_folder = 'nbody_gauss_run4/'#/paper_plots'
#
# savename = 'ctot2_ev_L{}'.format(int(Lambda/(2*np.pi)))
# # plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
# plt.close()
# # plt.show()
#


# #for ctot2 plots
# savename = 'ctot2_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, error_plotting=False, title_str=title)


# xlabel = r'$a$'
# ylabel = r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$'
# colours = ['orange', 'cyan', 'k']
# labels = [r'$\mathrm{B^{+12}}$', 'M&W', r'from fit to $\left<\tau_{l}\right>$']
# linestyles = ['solid', 'solid', 'solid']
# markers = ['v', '*', 'o'] #
#
# xaxis = a_list
# fig, ax = plt.subplots()
# ax.set_xlabel(xlabel, fontsize=18)
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (Gaussian)'.format(int(Lambda/(2*np.pi))), fontsize=18)
# # ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (sharp)'.format(int(Lambda/(2*np.pi))), fontsize=18)
#
# for i in range(0, 3):
#     ax.plot(xaxis, yaxes_sharp[i], c=colours[i], ls=linestyles[i], lw=2, marker=markers[i])
#     # ax.errorbar(xaxis, yaxes_sharp[i], yerr=yerror_sharp, c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker='o')
#     # ax.plot(xaxis, yaxes_gauss[i], c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker=markers[i])
#     # ax.errorbar(xaxis, yaxes_gauss[i], yerr=yerror_gauss, c=colours[i], ls=linestyles[i], lw=2, marker='o')
#
# ax.set_ylabel(ylabel, fontsize=18)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=13)
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=13)#, loc=2, bbox_to_anchor=(1,1))
# plt.show()
#
#
#
