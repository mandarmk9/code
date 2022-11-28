#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

from EFT_nbody_solver import *

#define directories, file parameteres
path = 'cosmo_sim_1d/nbody_hier/'

Nfiles = 25
Lambda = 5
H0 = 100
#define lists to store the data
a_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)
ctot2_list2 = np.zeros(Nfiles)
cs2_list = np.zeros(Nfiles)
cv2_list = np.zeros(Nfiles)

#An and Bn for the integral over the Green's function
An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
Pn = np.zeros(Nfiles)
Qn = np.zeros(Nfiles)

An2 = np.zeros(Nfiles)
Bn2 = np.zeros(Nfiles)
Pn2 = np.zeros(Nfiles)
Qn2 = np.zeros(Nfiles)

a0 = EFT_solve(0, Lambda, path)[0]

for file_num in range(Nfiles):
   if file_num > 0:
      a0 = a

   a, x, k, P_nb, P_lin, P_1l_sm, P_1l, tau_l, fit, ctot2, ctot2_2, cs2, cv2 = param_calc(file_num, Lambda, path)

   a_list[file_num] = a

   Nx = x.size

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0

      #for α_c using c^2 from fittting τ_l
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      Qn[file_num] = ctot2
      An[file_num] = da * Pn[file_num]
      Bn[file_num] = da * Qn[file_num]

      #for α_c using τ_l directly
      Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
      Qn2[file_num] = ctot2_2
      An2[file_num] = da * Pn2[file_num]
      Bn2[file_num] = da * Qn2[file_num]

   print('a = ', a, '\n')

#A second loop for the integration
for j in range(1, Nfiles):
   An[j] += An[j-1]
   Bn[j] += Bn[j-1]

#calculation of the Green's function integral
C = 2 / (5 * H0**2)
An /= (a_list**(5/2))
An2 /= (a_list**(5/2))

alpha_c_naive = C * (An - Bn)
alpha_c_naive2 = C * (An2 - Bn2)

P_eft = P_1l + ((2 * alpha_c_naive[-1]) * (k**2) * P_lin)
P_eft2 = P_1l + ((2 * alpha_c_naive2[-1]) * (k**2) * P_lin)

# from scipy.optimize import curve_fit
# def fitting_function(X, c, n):
#    P_spt, P11, a, k = X
#    return P_spt + ((c * (a**n)) * (k**2) * P11)
#
# guesses = 1, 1
# FF = curve_fit(fitting_function, (P_1l, P_lin, a_list, k), P_nb, guesses, sigma=1e-5*np.ones(a_list.size), method='lm')
# c, n = FF[0]
# cov = FF[1]
# err_c, err_n = np.sqrt(np.diag(cov))
# fit = fitting_function((P_1l, P_11, a_list, k), c, n)
#
# alpha_c_fit = (fit - P_1l) / (2 * P_lin)
# P_eft_fit = P_1l_tr + ((2 * alpha_c_fit) * (mode**2) * P_lin)

fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
ax[0].set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
ax[0].set_ylabel(r'$P(k)$', fontsize=14)
ax[1].set_xlabel(r'$k$', fontsize=14)

ax[0].scatter(k, P_nb, label=r'$P^{\mathrm{N-body}}_{\mathrm{NL}}$', s=35, c='b')
ax[0].scatter(k, P_lin, label=r'SPT: $P^{\mathrm{SPT}}_{\mathrm{lin}}$', s=25, c='r')

# truncated spectra
ax[0].scatter(k, P_1l, label=r'$P^{\mathrm{SPT}}_{\mathrm{1-loop}}$', s=20, c='brown')
ax[0].scatter(k, P_eft, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$: via $c^{2}_{\mathrm{tot}}$', s=20, c='k')
# ax[0].plot(k, P_eft2, label=r'$P_{\mathrm{EFT}}$: directly from $\tau_{l}$, ls='dashed', lw=3, c='green')
# ax[0].plot(k, P_eft_fit, label=r'EFT: fit to $P_{\mathrm{NL}}$', ls='dashed', lw=3, c='green')

# #bottom panel; errors
ax[1].axhline(0, color='b')

# #error on the linear PS
# err_lin = (P_lin - P_nb) * 100 / P_nb
# ax[1].plot(a_list, err_lin, lw=2.5, c='r')
#
# #errors on the truncated spectra
# err_1l = (P_1l - P_nb) * 100 / P_nb
# err_eft = (P_eft - P_nb) * 100 / P_nb
# err_eft2 = (P_eft2 - P_nb) * 100 / P_nb
# ax[1].plot(k, err_1l, lw=2.5, c='brown')
# ax[1].plot(k, err_eft, ls='dashed', lw=3, c='k')
# ax[1].plot(k, err_eft2, ls='dashed', lw=3, c='green')
# ax[1].plot(k, err_fit, ls='dotted', lw=3, c='orange')
#

ax[0].set_ylim(-0.01, 0.05)
ax[1].set_ylabel('% err', fontsize=14)
ax[1].set_xlim(0, 5)

ax[1].minorticks_on()

for i in range(2):
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].grid(lw=0.2, ls='dashed', color='grey')
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

plt.show()
# plt.savefig('../plots/EFT_nbody/PS.png'.format(Nfiles), bbox_inches='tight', dpi=120)
# plt.close()

# print('c = ', c)
# print('n = ', n)
