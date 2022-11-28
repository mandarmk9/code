#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from EFT_solver_av import *
from SPT import SPT_final

#define directories, file parameteres
path = '../data/sch_multi_k/'
# plots_folder = 'sch_new_run5_L3/'

Nfiles = 2
mode = 1
Lambda = 7 #this is not scaled by 2*pi because the k values are not either.
H0 = 100

#define lists to store the data
a_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)
ctot2_list2 = np.zeros(Nfiles)
# ctot2_list3 = np.zeros(Nfiles)
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

# An3 = np.zeros(Nfiles)
# Bn3 = np.zeros(Nfiles)
# Pn3 = np.zeros(Nfiles)
# Qn3 = np.zeros(Nfiles)

#the densitites
P_sch = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l_sm = np.zeros(Nfiles)
P_1l_tr = np.zeros(Nfiles)
P_2l_tr = np.zeros(Nfiles)

#initial scalefactor
a0 = EFT_solve(0, Lambda, path)[0]
m = 1 #this is the index in ctot2 = a^(m); for analytical tests

for file_num in range(Nfiles):
   # filename = '/output_hierarchy_{0:04d}.txt'.format(file_num)
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a

   a, x, k, P_sch_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2 = param_calc(file_num, Lambda, path)

   a_list[file_num] = a
   ctot2_list[file_num] = ctot2
   ctot2_list2[file_num] = ctot2_2
   # ctot2_list3[file_num] = ctot2_3

   Nx = x.size
   tau_k = np.fft.fft(tau_l) / Nx
   fit_k = np.fft.fft(fit) / Nx

   tau_2k = np.abs(tau_k * np.conj(tau_k))
   fit_2k = np.abs(fit_k * np.conj(fit_k))
   fit_list[file_num] = fit_2k[1]
   tau_list[file_num] = tau_2k[1]

   cs2_list[file_num] = cs2
   cv2_list[file_num] = cv2

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0

      #for α_c using c^2 from fittting τ_l
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      Qn[file_num] = ctot2
      An[file_num] = da * Pn[file_num]
      Bn[file_num] = da * Qn[file_num]

      #for α_c using τ_l (M&W)
      Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
      Qn2[file_num] = ctot2_2
      An2[file_num] = da * Pn2[file_num]
      Bn2[file_num] = da * Qn2[file_num]

      # #for α_c using τ_l (Baumann)
      # Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
      # Qn3[file_num] = ctot2_3
      # An3[file_num] = da * Pn3[file_num]
      # Bn3[file_num] = da * Qn3[file_num]


   #we now extract the solutions for a specific mode
   P_sch[file_num] = P_sch_a[mode]
   P_lin[file_num] = P_lin_a[mode]
   P_1l_sm[file_num] = P_1l_a_sm[mode]
   P_1l_tr[file_num] = P_1l_a_tr[mode]
   P_2l_tr[file_num] = P_2l_a_tr[mode]

   print('a = ', a, '\n')

#A second loop for the integration
for j in range(1, Nfiles):
   An[j] += An[j-1]
   Bn[j] += Bn[j-1]

   An2[j] += An2[j-1]
   Bn2[j] += Bn2[j-1]

   # An3[j] += An3[j-1]
   # Bn3[j] += Bn3[j-1]


#calculation of the Green's function integral
C = 2 / (5 * H0**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))
# An3 /= (a_list**(5/2))


alpha_c_naive = C * (An - Bn)
alpha_c_naive2 = C * (An2 - Bn2)
# alpha_c_naive3 = C * (An3 - Bn3)
# alpha_c_true = (P_sch - P_1l_tr) / (2 * P_lin * k[mode]**2)

# P_eft_sm = P_1l_sm + ((2 * alpha_c_naive) * (k[mode]**2) * P_lin)
# P_eft2_sm = P_1l_sm + ((2 * alpha_c_naive2) * (k[mode]**2) * P_lin)
# # P_eft3_sm = P_1l_sm + ((2 * alpha_c_naive3) * (k[mode]**2) * P_lin)
#
# P_eft_tr = P_1l_tr + ((2 * alpha_c_naive) * (k[mode]**2) * P_lin)
# P_eft2_tr = P_1l_tr + ((2 * alpha_c_naive2) * (k[mode]**2) * P_lin)
# # P_eft3_tr = P_1l_tr + ((2 * alpha_c_naive3) * (k[mode]**2) * P_lin)
#
# P_eft_fit = P_1l_tr + ((2 * alpha_c_true) * (k[mode]**2) * P_lin)


P_tau = P_1l_a_tr + ((2 * alpha_c_naive[-1]) * (k**2) * P_lin_a)
P_MW = P_1l_a_tr + ((2 * alpha_c_naive2[-1]) * (k**2) * P_lin_a)

fig, ax = plt.subplots()
ax.set_title('a = {}'.format(np.round(a, 4)))
#
# ax.scatter(k, P_ZA, c='k', s=30, label='Zel')
ax.scatter(k, P_sch_a, c='b', s=55, label=r'Sch')
ax.scatter(k, P_lin_a, c='r', s=45, label=r'SPT: lin')
ax.scatter(k, P_1l_a_tr, c='magenta', s=35, label=r'SPT: 1-loop')
ax.scatter(k, P_2l_a_tr, c='cyan', s=25, label=r'SPT: 2-loop')
ax.scatter(k, P_MW, c='orange', s=15, label=r'EFT: M&W')
ax.scatter(k, P_tau, c='k', s=5, label=r'EFT: from fit to $\tau_{l}$')

ax.set_xlim(-0.5, 15.5)
# ax.set_ylim(1e-7, 1)

ax.set_xlabel(r'k', fontsize=14)
ax.set_ylabel(r'$P(k)$', fontsize=14)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
ax.yaxis.set_ticks_position('both')
plt.yscale('log')
# print(P_1l[11], P_2l[11], P_nb[11])
# ax.plot(x, d0, c='k', lw=2, label='ZA: direct')
# ax.plot(x, M0_par-1, c='b', lw=2, ls='dashdot', label=r'$N$-body')
# ax.plot(q_nbody, dc_nb, c='r', lw=2, ls='dashed', label=r'$N$-body: from $\Psi$')

# plt.savefig('../plots/test/spec_sch_multi/PS_{0:03d}.png'.format(j), bbox_inches='tight', dpi=150)
# plt.close()
plt.show()


# #for plotting the spectra
# xaxis = a_list
# yaxes = [P_sch / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft2_tr / a_list**2, P_eft_fit / a_list**2]
# colours = ['b', 'brown', 'k', 'cyan', 'g']
# labels = [r'Sch', 'SPT: 1-loop', r'EFT: from fit to $\tau_{l}$', 'EFT: M&W', r'EFT: from matching $P^{\mathrm{N-body}}_{\mathrm{NL}}$']
# linestyles = ['solid', 'dashed', 'dashdot', 'dashdot', 'dashdot']
# savename = 'spectra'
# ylabel = r'$|\tilde{\delta}(k=1, a)|^{2} / a^{2}$'
# plotter(mode, Lambda, xaxis, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, which='Sch')
#
# #for ctot2 plots
# xaxis = a_list
# yaxes = [ctot2_list, ctot2_list2]
# colours = ['k', 'cyan']
# labels = [r'EFT: from fit to $\tau_{l}$', 'EFT: M&W']
# linestyles = ['dashdot', 'dashdot']
# savename = 'ctot2'
# ylabel = r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$'
# plotter(mode, Lambda, xaxis, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, error_plotting=False, which='Sch')
#
# #for alpha_c plots
# xaxis = a_list
# yaxes = [alpha_c_true, alpha_c_naive, alpha_c_naive2]
# colours = ['g', 'k', 'cyan', 'orange']
# labels = [r'from matching $P^{\mathrm{Sch}}_{\mathrm{NL}}$', r'from fit to $\tau_{l}$', 'M&W']
# linestyles = ['solid', 'dashdot', 'dashdot']
# savename = 'alpha_c'
# ylabel = r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$'
# plotter(mode, Lambda, xaxis, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, which='Sch')
#
# #for fit-as-a-function-of-time plot
# xaxis = a_list
# yaxes = [np.log(tau_list), np.log(fit_list)]
# colours = ['b', 'k']
# labels = [r'$\tau_{l}$', r'fit to $tau_{l}$']
# linestyles = ['solid', 'dashed']
# savename = 'fit'
# ylabel = r'$|\tau_{l}(k)|^{2}$'
# plotter(mode, Lambda, xaxis, yaxes, ylabel, colours, labels, linestyles, plots_folder, savename, which='Sch')
