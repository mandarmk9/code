#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

from EFT_solver_av import *
from SPT import SPT_final
from functions import dn

#define directories, file parameteres
path = '../data/new_run1/'
index = 324
Nfiles = index
mode = 1
# Lambda = 6
H0 = 100

#define lists to store the data
alpha_c_fit_list, alpha_c_true_list, alpha_c_naive_list, alpha_c_naive2_list, alpha_c_naive3_list = [], [], [], [], []

a_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)
ctot2_list2 = np.zeros(Nfiles)
ctot2_list3 = np.zeros(Nfiles)

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

An3 = np.zeros(Nfiles)
Bn3 = np.zeros(Nfiles)
Pn3 = np.zeros(Nfiles)
Qn3 = np.zeros(Nfiles)

#the densitites
P_nb = np.zeros(Nfiles)
P_1l = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)

Lambdas = np.array([2, 3, 4, 5, 7])#, 9, 11])
for l in range(len(Lambdas)):
   Lambda = Lambdas[l]
   print('Lambda = {}'.format(Lambda))
   #initial scalefactor
   a0 = EFT_solve(0, Lambda, path)[0]
   m = 1 #this is the index in ctot2 = a^(m); for analytical tests

   for file_num in range(0, index):
      #the function 'EFT_solve' return solutions of all modes + the EFT parameters
      ##the following line is to keep track of 'a' for the numerical integration
      if file_num > 0:
         a0 = a

      a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_1l_a, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2 = param_calc(file_num, Lambda, path)

      a_list[file_num] = a
      ctot2_list[file_num] = ctot2
      ctot2_list2[file_num] = ctot2_2
      # ctot2_list3[file_num] = ctot2_3

      ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
      if file_num > 0:
         da = a - a0
         Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
         Qn[file_num] = ctot2

         Pn2[file_num] = ctot2_2 * (a**(5/2)) #for calculation of alpha_c
         Qn2[file_num] = ctot2_2

         # Pn3[file_num] = ctot2_3 * (a**(5/2)) #for calculation of alpha_c
         # Qn3[file_num] = ctot2_3

         An[file_num] = da * Pn[file_num]
         Bn[file_num] = da * Qn[file_num]

         An2[file_num] = da * Pn2[file_num]
         Bn2[file_num] = da * Qn2[file_num]

         # An3[file_num] = da * Pn3[file_num]
         # Bn3[file_num] = da * Qn3[file_num]

      #we now extract the solutions for a specific mode
      P_nb[file_num] = P_nb_a[mode]
      P_lin[file_num] = P_lin_a[mode]
      P_1l[file_num] = P_1l_a[mode]
      # P_1l_sm[file_num] = P_1l_a_sm[mode]

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

   alpha_c_guess = -(ctot2_list2 * a_list) / (9 * H0**2)

   #for 3 parameters a0, a1, a2 such that τ_l = a0 + a1 × (1 + δ_l) + a2 × dv_l
   from scipy.optimize import curve_fit
   def fitting_function(X, c, n):
      P_1l, P_lin, a, mode = X
      return P_1l + ((c * (a**n)) * (mode[0]**2) * P_lin)

   guesses = 1, 1
   FF = curve_fit(fitting_function, (P_1l, P_lin, a_list, mode*np.ones(a_list.size)), P_nb, guesses, sigma=1e-5*np.ones(a_list.size), method='lm')
   c, n = FF[0]
   cov = FF[1]
   err_c, err_n = np.sqrt(np.diag(cov))
   fit = fitting_function((P_1l, P_lin, a_list, mode*np.ones(a_list.size)), c, n)

   alpha_c_fit = (fit - P_1l) / (2 * P_lin * k[mode]**2)
   alpha_c_true = (P_nb - P_1l) / (2 * P_lin * k[mode]**2)

   alpha_c_fit_list.append(alpha_c_fit[-1])
   alpha_c_naive_list.append(alpha_c_naive[-1])
   alpha_c_naive2_list.append(alpha_c_naive2[-1])
   # alpha_c_naive3_list.append(alpha_c_naive3[-1])
   alpha_c_true_list.append(alpha_c_true[-1])


   a_final = a_list[-1]

Lambdas_inv = 1 / Lambdas
alpha_c_naive_list = np.array(alpha_c_naive_list) * 2 / (0.7**2) #to convert from h^{-2}Mpc^2 to Mpc^2 and get 2α_c
alpha_c_naive2_list = np.array(alpha_c_naive2_list) * 2 / (0.7**2)
# alpha_c_naive3_list = np.array(alpha_c_naive3_list) * 2 / (0.7**2)

alpha_c_fit_list = np.array(alpha_c_fit_list) * 2 / (0.7**2)
alpha_c_true_list = np.array(alpha_c_true_list) * 2 / (0.7**2)

Lambdas_inv /= 0.7 #to convert from h^{-1}Mpc to Mpc

fig, ax = plt.subplots(dpi=150)
# ax.set_title(r'$k_{2} = 15\; [h\;\mathrm{Mpc}^{-1}]$')
ax.set_title(r'$a = {}$'.format(a_final))

ax.set_ylabel(r'$2\alpha_c\;[\mathrm{Mpc}^{2}]$', fontsize=14)
ax.set_xlabel(r'$\Lambda^{-1} [\mathrm{Mpc}]$', fontsize=14)

ax.plot(Lambdas_inv, alpha_c_naive_list, c='k', lw=2, marker='+', label=r'from fit to $\tau_{l}$')
ax.plot(Lambdas_inv, alpha_c_naive2_list, c='cyan', marker='o', label=r'M&W')
# ax.plot(Lambdas_inv, alpha_c_naive3_list, c='orange', marker='*', label=r'Baumann')
# ax.plot(Lambdas_inv, alpha_c_fit_list, c='green', lw=2, label=r'from fit to $P^{N\mathrm{-body}}$')
ax.plot(Lambdas_inv, alpha_c_true_list, c='blue', marker='x', lw=2, label=r'from matching $P^{N\mathrm{-body}}$')

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('bot1')

ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))


plt.savefig('../plots/sch_new_run1_L5/alpha_c_scale_dep_s10.png'.format(Lambda), bbox_inches='tight', dpi=120)
plt.close()
