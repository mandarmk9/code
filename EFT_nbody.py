#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pickle
import pandas
# from EFT_ens_solver import *
from EFT_nbody_solver import *
#
# # from EFT_nbody_solver import *

from SPT import SPT_final
from functions import plotter, SPT_real_sm, SPT_real_tr, dc_in_finder, dn
from zel import eulerian_sampling
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#run5: k2 = 7; Nfiles = 101
#run2: k2 = 11; Nfiles = 81
#run6: only k1; Nfiles = 51

path = 'cosmo_sim_1d/sim_k_1/run1/'
# path = 'cosmo_sim_1d/final_phase_run1/'
# plots_folder = 'test/spec'
# plots_folder = 'test'

zero = 0
Nfiles = 51
mode = 1
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
H0 = 100
A = [-0.05, 1, -0.0, 11]#, -0.01, 2, -0.01, 3, -0.01, 4]

#define lists to store the data
a_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)
ctot2_list2 = np.zeros(Nfiles)
ctot2_list3 = np.zeros(Nfiles)
cs2_list = np.zeros(Nfiles)
cv2_list = np.zeros(Nfiles)
fit_list = np.zeros(Nfiles)
tau_list = np.zeros(Nfiles)

#An and Bn for the integral over the Green's function
An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
Pn = np.zeros(Nfiles)
Qn = np.zeros(Nfiles)

An2 = np.zeros(Nfiles)
Bn2 = np.zeros(Nfiles)
Pn2 = np.zeros(Nfiles)
Qn2 = np.zeros(Nfiles)

An3 = np.zeros(Nfiles)
Bn3 = np.zeros(Nfiles)
Pn3 = np.zeros(Nfiles)
Qn3 = np.zeros(Nfiles)

err_A = np.zeros(Nfiles)
err_B = np.zeros(Nfiles)

terr_list = np.zeros(Nfiles)

#the densitites
P_nb = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l_sm = np.zeros(Nfiles)
P_1l_tr = np.zeros(Nfiles)
P_2l_sm = np.zeros(Nfiles)
P_2l_tr = np.zeros(Nfiles)

dk_stoch = np.zeros(Nfiles)
dk_nb = np.zeros(Nfiles)
dk_eft = np.zeros(Nfiles)


#initial scalefactor
a0 = EFT_solve(0, Lambda, path, A, kind)[0]

for file_num in range(zero, Nfiles):
   # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a

   a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody, d1k, d2k, terr = param_calc(file_num, Lambda, path, A, mode, kind)
   a_list[file_num] = a
   ctot2_list[file_num] = ctot2
   ctot2_list2[file_num] = ctot2_2
   ctot2_list3[file_num] = ctot2_3
   print('tau_l sum = ', np.sum(tau_l))
   Nx = x.size

   cs2_list[file_num] = cs2
   cv2_list[file_num] = cv2

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0

      #for α_c using c^2 from fit to τ_l
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      Qn[file_num] = ctot2
      # An[file_num] = da * Pn[file_num]
      # Bn[file_num] = da * Qn[file_num]

      #for α_c using τ_l directly (M&W)
      Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
      Qn2[file_num] = ctot2_2
      # An2[file_num] = da * Pn2[file_num]
      # Bn2[file_num] = da * Qn2[file_num]

      #for α_c using correlations (Baumann)
      Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
      Qn3[file_num] = ctot2_3
      # An3[file_num] = da * Pn3[file_num]
      # Bn3[file_num] = da * Qn3[file_num]


   #we now extract the solutions for a specific mode
   P_nb[file_num] = P_nb_a[mode]
   P_lin[file_num] = P_lin_a[mode]
   P_1l_sm[file_num] = P_1l_a_sm[mode]
   P_2l_sm[file_num] = P_2l_a_sm[mode]
   P_1l_tr[file_num] = P_1l_a_tr[mode]
   P_2l_tr[file_num] = P_2l_a_tr[mode]
   terr_list[file_num] = terr

   dk_nb[file_num] = (np.fft.fft(M0_nbody) / M0_nbody.size)[mode]

   print('a = ', a, '\n')

#A second loop for the integration
for j in range(1, Nfiles):
    An[j] = np.trapz(Pn[:j], a_list[:j])
    Bn[j] = np.trapz(Qn[:j], a_list[:j])

    An2[j] = np.trapz(Pn2[:j], a_list[:j])
    Bn2[j] = np.trapz(Qn2[:j], a_list[:j])

    An3[j] = np.trapz(Pn3[:j], a_list[:j])
    Bn3[j] = np.trapz(Qn3[:j], a_list[:j])

    err_A[j] = np.trapz((a_list**(5/2))[:j], a_list[:j])
    err_B[j] = np.trapz((np.ones(a_list.size))[:j], a_list[:j])


#calculation of the Green's function integral
C = 2 / (5 * H0**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))
An3 /= (a_list**(5/2))

err_A /= a_list**(5/2)
err_Int = C * terr_list * (err_A - err_B)
# print(err_A, err_B)
# print(terr_list)

alpha_c_true = (P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2)
sign = 1#-alpha_c_true / np.abs(alpha_c_true)
alpha_c_naive = sign * C * (An - Bn)
alpha_c_naive2 = sign * C * (An2 - Bn2)
alpha_c_naive3 = sign * C * (An3 - Bn3)

P_eft_tr = P_1l_tr + ((2 * alpha_c_naive) * (k[mode]**2) * P_lin)
P_eft2_tr = P_1l_tr + ((2 * alpha_c_naive2) * (k[mode]**2) * P_lin)
P_eft3_tr = P_1l_tr + ((2 * alpha_c_naive3) * (k[mode]**2) * P_lin)
P_eft_fit = P_1l_tr + ((2 * alpha_c_true) * (k[mode]**2) * P_lin)

a_sc = 1 / np.max(initial_density(x, A, 1))

#for plotting the spectra
xaxis = a_list
yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft3_tr / a_list**2, P_eft_fit / a_list**2] #, P_eft_w_stoch / a_list**2]
for spec in yaxes:
    spec /= 1e-4

# $\left<[\tau]_{l}\right>$
colours = ['b', 'brown', 'k', 'cyan', 'orange', 'g', 'violet']
labels = [r'$N$-body', 'SPT-4', r'EFT: from fit to $[\tau]_{l}$',  'EFT: M&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{\mathrm{N-body}}$']
linestyles = ['solid', 'dashdot', 'dashed',  'dashed', 'dashed', 'dotted']
savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
xlabel = r'$a$'
ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes]
plots_folder = '/sim_k_1/'

# df = pandas.DataFrame(data=[mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title, err_Int])
# pickle.dump(df, open("spec_plot.p", "wb"))

# [mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title, err_Int] = pickle.load(open("spec_plot.p", "rb" ))[0]
a_sc = 0
plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title, terr=err_Int, save=True)

fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
ax[0].set_title(title, fontsize=16)
ax[1].set_xlabel(xlabel, fontsize=16)
ax[0].set_ylabel(ylabel, fontsize=16)
for i in range(len(yaxes)):
    # if i==2:
    #     ax[0].errorbar(xaxis, yaxes[2], yerr=err_Int, c=colours[2], ls=linestyles[2], lw=2.5, label=labels[2])
    # else:
    ax[0].plot(xaxis, yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5, label=labels[i])

    if i == 0:
        ax[1].axhline(0, c=colours[0])
    else:
        ax[1].plot(xaxis, errors[i], ls=linestyles[i], lw=2.5, c=colours[i])
for i in range(2):
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in')
    # ax[i].axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11)
ax[1].set_ylabel('% err', fontsize=16)

plt.subplots_adjust(hspace=0)

# plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()

plt.show()

# #for ctot2 plots
# plots_folder = 'sim_k_1'
# xaxis = a_list
# yaxes = [ctot2_list, ctot2_list2, ctot2_list3]
# colours = ['k', 'cyan', 'orange']
# labels = [r'from fit to $\tau_{l}$', 'M&W', 'Baumann']
# linestyles = ['dashdot', 'dashdot', 'dashdot']
# savename = 'ctot2_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# ylabel = r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# save = False
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, error_plotting=False, title_str=title)

# #for alpha_c plots
# xaxis = a_list
# yaxes = [alpha_c_true, alpha_c_naive, alpha_c_naive2, alpha_c_naive3]
# colours = ['g', 'k', 'cyan', 'orange']
# labels = [r'from matching $P^{\mathrm{N-body}}_{\mathrm{NL}}$', r'from fit to $\tau_{l}$', 'M&W', 'Baumann']
# linestyles = ['solid', 'dashdot', 'dashdot', 'dashdot']
# savename = 'alpha_c_k{}_L{}'.format(mode, int(Lambda/(2*np.pi)))
# xlabel = r'$a$'
# ylabel = r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$'
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename)
#
# #for fit-as-a-function-of-time plot
# xaxis = a_list
# yaxes = [np.log(tau_list), np.log(fit_list)]
# colours = ['b', 'k']
# labels = [r'$\tau_{l}$', r'fit to $tau_{l}$']
# linestyles = ['solid', 'dashed']
# savename = 'fit_k{}_L{}'.format(mode, int(Lambda/(2*np.pi)))
# xlabel = r'$a$'
# ylabel = r'$|\tau_{l}(k)|^{2}$'
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename)


# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
# moments_file = np.genfromtxt(path + moments_filename)
# x = moments_file[:,0]
# k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
# dc_in = dc_in_finder(path, x)
# dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
# F = dn(3, k, 1.0, dc_in)
# d1k = (np.fft.fft(F[0]) / k.size)
# d2k = (np.fft.fft(F[1]) / k.size)
# d3k = (np.fft.fft(F[2]) / k.size)
#
# for j in range(Nfiles):
#     a = a_list[j]
#     d3k_corr = alpha_c_naive[j] * (k**2) * d1k * a
#     dk_eft[j] = (a*d1k + (a**2)*d2k + (a**3)*(d3k) + d3k_corr)[mode]
#     dk_stoch[j] = dk_nb[j] - dk_eft[j]
#
#     P11 = (d1k * np.conj(d1k)) * (a**2)
#     P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
#     P22 = (d2k * np.run1/'
# path = 'cosmo_sim_1d/final_phase_run1/'
# plots_folder = 'test/spec'
# plots_folder = 'test'

# zero = 0
# Nfiles = 51
# mode = 1
# Lambda = 3 * (2 * np.pi)
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'
# H0 = 100
# A = [-0.05, 1, -0.0, 11]#, -0.01, 2, -0.01, 3, -0.01, 4]
#
# #define lists to store the data
# a_list = np.zeros(Nfiles)
# ctot2_list = np.zeros(Nfiles)
# ctot2_list2 = np.zeros(Nfiles)
# ctot2_list3 = np.zeros(Nfiles)
# cs2_list = np.zeros(Nfiles)
# cv2_list = np.zeros(Nfiles)
# fit_list = np.zeros(Nfiles)
# tau_list = np.zeros(Nfiles)
#
# #An and Bn for the integral over the Green's function
# An = np.zeros(Nfiles)
# Bn = np.zeros(Nfiles)
# Pn = np.zeros(Nfiles)
# Qn = np.zeros(Nfiles)
#
# An2 = np.zeros(Nfiles)
# Bn2 = np.zeros(Nfiles)
# Pn2 = np.zeros(Nfiles)
# Qn2 = np.zeros(Nfiles)
#
# An3 = np.zeros(Nfiles)
# Bn3 = np.zeros(Nfiles)
# Pn3 = np.zeros(Nfiles)
# Qn3 = np.zeros(Nfiles)
#
# err_A = np.zeros(Nfiles)
# err_B = np.zeros(Nfiles)
#
# terr_list = np.zeros(Nfiles)
#
# #the densitites
# P_nb = np.zeros(Nfiles)
# P_lin = np.zeros(Nfiles)
# P_1l_sm = np.zeros(Nfiles)
# P_1l_tr = np.zeros(Nfiles)
# P_2l_sm = np.zeros(Nfiles)
# P_2l_tr = np.zeros(Nfiles)
#
# dk_stoch = np.zeros(Nfiles)
# dk_nb = np.zeros(Nfiles)
# dk_eft = np.zeros(Nfiles)
#
#
# #initial scalefactor
# a0 = EFT_solve(0, Lambda, path, A, kind)[0]
#
# for file_num in range(zero, Nfiles):
#    # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
#    #the function 'EFT_solve' return solutions of all modes + the EFT parameters
#    ##the following line is to keep track of 'a' for the numerical integration
#    if file_num > 0:
#       a0 = a
#
#    a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody, d1k, d2k, terr = param_calc(file_num, Lambda, path, A, mode, kind)
#    a_list[file_num] = a
#    ctot2_list[file_num] = ctot2
#    ctot2_list2[file_num] = ctot2_2
#    ctot2_list3[file_num] = ctot2_3
#
#    Nx = x.size
#
#    cs2_list[file_num] = cs2
#    cv2_list[file_num] = cv2
#
#    ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
#    if file_num > 0:
#       da = a - a0
#
#       #for α_c using c^2 from fit to τ_l
#       Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
#       Qn[file_num] = ctot2
#       # An[file_num] = da * Pn[file_num]
#       # Bn[file_num] = da * Qn[file_num]
#
#       #for α_c using τ_l directly (M&W)
#       Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
#       Qn2[file_num] = ctot2_2
#       # An2[file_num] = da * Pn2[file_num]
#       # Bn2[file_num] = da * Qn2[file_num]
#
#       #for α_c using correlations (Baumann)
#       Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
#       Qn3[file_num] = ctot2_3
#       # An3[file_num] = da * Pn3[file_num]
#       # Bn3[file_num] = da * Qn3[file_num]
#
#
#    #we now extract the solutions for a specific mode
#    P_nb[file_num] = P_nb_a[mode]
#    P_lin[file_num] = P_lin_a[mode]
#    P_1l_sm[file_num] = P_1l_a_sm[mode]
#    P_2l_sm[file_num] = P_2l_a_sm[mode]
#    P_1l_tr[file_num] = P_1l_a_tr[mode]
#    P_2l_tr[file_num] = P_2l_a_tr[mode]
#    terr_list[file_num] = terr
#
#    dk_nb[file_num] = (np.fft.fft(M0_nbody) / M0_nbody.size)[mode]
#
#    print('a = ', a, '\n')
#
# #A second loop for the integration
# for j in range(1, Nfiles):
#     An[j] = np.trapz(Pn[:j], a_list[:j])
#     Bn[j] = np.trapz(Qn[:j], a_list[:j])
#
#     An2[j] = np.trapz(Pn2[:j], a_list[:j])
#     Bn2[j] = np.trapz(Qn2[:j], a_list[:j])
#
#     An3[j] = np.trapz(Pn3[:j], a_list[:j])
#     Bn3[j] = np.trapz(Qn3[:j], a_list[:j])
#
#     err_A[j] = np.trapz((a_list**(5/2))[:j], a_list[:j])
#     err_B[j] = np.trapz((np.ones(a_list.size))[:j], a_list[:j])
#
#
# #calculation of the Green's function integral
# C = 2 / (5 * H0**2)
# An /= (a_list**(5/2))
# An2 /= (a_list**(5/2))
# An3 /= (a_list**(5/2))
#
# err_A /= a_list**(5/2)
# err_Int = C * terr_list * (err_A - err_B)
# # print(err_A, err_B)
# # print(terr_list)
#
# alpha_c_true = (P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2)
# sign = -alpha_c_true / np.abs(alpha_c_true)
# alpha_c_naive = sign * C * (An - Bn)
# alpha_c_naive2 = sign * C * (An2 - Bn2)
# alpha_c_naive3 = sign * C * (An3 - Bn3)

# moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
# moments_file = np.genfromtxt(path + moments_filename)
# x = moments_file[:,0]
# k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
# dc_in = dc_in_finder(path, x)
# dc_in = smoothing(dc_in, k, Lambda, kind) #truncating the initial overdensity
# F = dn(3, k, 1.0, dc_in)
# d1k = (np.fft.fft(F[0]) / k.size)
# d2k = (np.fft.fft(F[1]) / k.size)
# d3k = (np.fft.fft(F[2]) / k.size)
#
# for j in range(Nfiles):
#     a = a_list[j]
#     d3k_corr = alpha_c_naive[j] * (k**2) * d1k * a
#     dk_eft[j] = (a*d1k + (a**2)*d2k + (a**3)*(d3k) + d3k_corr)[mode]
#     dk_stoch[j] = dk_nb[j] - dk_eft[j]
#
#     P11 = (d1k * np.conj(d1k)) * (a**2)
#     P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
#     P22 = (d2k * np.conj(d2k)) * (a**4)
#     P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
#     Pc1 = ((d1k * np.conj(d3k_corr)) + (d3k_corr * np.conj(d1k))) * (a)
#
# P_jj = dk_stoch * np.conj(dk_stoch)
# P_ej = ((dk_stoch * np.conj(dk_eft)) + (dk_eft * np.conj(dk_stoch)))
# P_ee = dk_eft * np.conj(dk_eft)
#
# #for plotting the spectra
# xaxis = a_list
# yaxes = [P_nb, P_jj, P_ej, P_ee]
# # for spec in yaxes:
# #     spec /= 1e-4
# colours = ['b', 'k', 'cyan', 'r']
# labels = [r'$P_{nb}$', '$P_{JJ}$', r'$P_{EJ}$', r'$P_{EE}$']
# linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
# savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# ylabel = r'$P(k=1, a)$'
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# a_sc = 1 / np.max(initial_density(x, A, 1))
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title)



# P_eft_tr = P_1l_tr + ((2 * alpha_c_naive) * (k[mode]**2) * P_lin)
# P_eft2_tr = P_1l_tr + ((2 * alpha_c_naive2) * (k[mode]**2) * P_lin)
# P_eft3_tr = P_1l_tr + ((2 * alpha_c_naive3) * (k[mode]**2) * P_lin)
# P_eft_fit = P_1l_tr + ((2 * alpha_c_true) * (k[mode]**2) * P_lin)
#
# a_sc = 1 / np.max(initial_density(x, A, 1))
#
# #for plotting the spectra
# xaxis = a_list
# yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft3_tr / a_list**2, P_eft_fit / a_list**2] #, P_eft_w_stoch / a_list**2]
# for spec in yaxes:
#     spec /= 1e-4
#
# # $\left<[\tau]_{l}\right>$
# colours = ['b', 'brown', 'k', 'cyan', 'orange', 'g', 'violet']
# labels = [r'$N$-body', 'SPT-4', r'EFT: from fit to $[\tau]_{l}$',  'EFT: M&W', 'EFT: $B^{+12}$', r'EFT: from matching $P_{\mathrm{N-body}}$', r'EFT + stochastic'] #$\left<[\tau]_{l}\right>$
# linestyles = ['solid', 'dashdot', 'dashed',  'dashed', 'dashed', 'dashed', 'dashed']
# savename = 'final_spec' #'err_eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# errors = [(yaxis - yaxes[0]) * 100 / yaxes[0] for yaxis in yaxes]
# plots_folder = '/sim_k_1/'

# df = pandas.DataFrame(data=[mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title, err_Int])
# pickle.dump(df, open("spec_plot.p", "wconj(d2k)) * (a**4)
#     P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
#     Pc1 = ((d1k * np.conj(d3k_corr)) + (d3k_corr * np.conj(d1k))) * (a)
#
# P_jj = dk_stoch * np.conj(dk_stoch)
# P_ej = ((dk_stoch * np.conj(dk_eft)) + (dk_eft * np.conj(dk_stoch)))
# P_ee = dk_eft * np.conj(dk_eft)
#
# #for plotting the spectra
# xaxis = a_list
# yaxes = [P_nb, P_jj, P_ej, P_ee]
# # for spec in yaxes:
# #     spec /= 1e-4
# colours = ['b', 'k', 'cyan', 'r']
# labels = [r'$P_{nb}$', '$P_{JJ}$', r'$P_{EJ}$', r'$P_{EE}$']
# linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
# savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
# xlabel = r'$a$'
# ylabel = r'$P(k=1, a)$'
# # ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
# title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)
# a_sc = 1 / np.max(initial_density(x, A, 1))
# plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title)
