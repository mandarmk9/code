#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from EFT_solver_gauss import *
from SPT import SPT_final
from functions import plotter, SPT_real_sm, SPT_real_tr, alpha_to_corr
from zel import eulerian_sampling
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/nbody_gauss_run4/'
plots_folder = 'nbody_gauss_run4'
# plots_folder = 'test'

zero = 0
Nfiles = 51
mode = 1
Lambda = 3 * (2 * np.pi)
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
H0 = 100

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

#the densitites
P_nb = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l_sm = np.zeros(Nfiles)
P_1l_tr = np.zeros(Nfiles)
P_2l_sm = np.zeros(Nfiles)
P_2l_tr = np.zeros(Nfiles)
P_zel = np.zeros(Nfiles)

#initial scalefactor
a0 = EFT_solve(0, Lambda, path, kind)[0]

for file_num in range(zero, Nfiles):
   # filename = '/output_hierarchy_{0:03d}.txt'.format(file_num)
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a

   a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody = param_calc(file_num, Lambda, path, mode, kind)

   a_list[file_num] = a
   ctot2_list[file_num] = ctot2
   ctot2_list2[file_num] = ctot2_2
   ctot2_list3[file_num] = ctot2_3

   Nx = x.size
   tau_k = np.fft.fft(tau_l) / Nx
   fit_k = np.fft.fft(fit) / Nx

   tau_2k = np.abs(tau_k * np.conj(tau_k))
   fit_2k = np.abs(fit_k * np.conj(fit_k))
   fit_list[file_num] = fit_2k[mode]
   tau_list[file_num] = tau_2k[mode]

   cs2_list[file_num] = cs2
   cv2_list[file_num] = cv2

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0

      #for α_c using c^2 from fit to τ_l
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      Qn[file_num] = ctot2

      #for α_c using τ_l directly (M&W)
      Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
      Qn2[file_num] = ctot2_2

      #for α_c using correlations (Baumann)
      Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
      Qn3[file_num] = ctot2_3

   #we now extract the solutions for a specific mode
   P_nb[file_num] = P_nb_a[mode]
   P_lin[file_num] = P_lin_a[mode]
   P_1l_sm[file_num] = P_1l_a_sm[mode]
   P_2l_sm[file_num] = P_2l_a_sm[mode]
   P_1l_tr[file_num] = P_1l_a_tr[mode]
   P_2l_tr[file_num] = P_2l_a_tr[mode]

   print('a = ', a, '\n')

#A second loop for the integration
for j in range(1, Nfiles):
    An[j] = np.trapz(Pn[:j], a_list[:j])
    Bn[j] = np.trapz(Qn[:j], a_list[:j])

    An2[j] = np.trapz(Pn2[:j], a_list[:j])
    Bn2[j] = np.trapz(Qn2[:j], a_list[:j])

    An3[j] = np.trapz(Pn3[:j], a_list[:j])
    Bn3[j] = np.trapz(Qn3[:j], a_list[:j])

#calculation of the Green's function integral
C = 2 / (5 * H0**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))
An3 /= (a_list**(5/2))


alpha_c_true = (P_nb - P_1l_tr) / (2 * P_lin * k[mode]**2)
sign = -alpha_c_true / np.abs(alpha_c_true)
alpha_c_naive = sign * C * (An - Bn)
alpha_c_naive2 = sign * C * (An2 - Bn2)
alpha_c_naive3 = sign * C * (An3 - Bn3)

P_eft_tr = P_1l_tr + ((2 * alpha_c_naive) * (k[mode]**2) * P_lin)
P_eft2_tr = P_1l_tr + ((2 * alpha_c_naive2) * (k[mode]**2) * P_lin)
P_eft3_tr = P_1l_tr + ((2 * alpha_c_naive3) * (k[mode]**2) * P_lin)
P_eft_fit = P_1l_tr + ((2 * alpha_c_true) * (k[mode]**2) * P_lin)

a_sc = 1 #/ np.max(initial_density(x, A, 1))

#for plotting the spectra
xaxis = a_list
yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft_fit / a_list**2]
# for spec in yaxes:
#     spec /= 1e-4
colours = ['b', 'brown', 'k', 'cyan', 'g']
labels = [r'$N$-body', 'SPT-4', r'EFT: from fit to $\tau_{l}$',  'EFT: M&W', r'EFT: from matching $P^{\mathrm{N-body}}_{\mathrm{NL}}$']
linestyles = ['solid', 'dashdot', 'dashed', 'dashed', 'dashed']
savename = 'spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
xlabel = r'$a$'
ylabel = r'$a^{-2}P(k=1, a)$' # \times 10^{4}$'
# ylabel = r'$|\tilde{\delta}(k=1, a)|^{2}\; / a^{2}$'
title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)

plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title)
