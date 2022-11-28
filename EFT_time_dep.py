#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

from EFT_solver import *
from SPT import SPT_final
from functions import dn, interp_green

#define directories, file parameteres
loc = '../'
run = '/sch_hfix_mix_run1/'
Nfiles = 851 #len(os.listdir('/vol/aibn31/data1/mandar/data' + run))
mode = 1
Lambda = 5
H0 = 100

# art_a_list = np.arange(0.01, 10, 0.1)
# Nfiles = len(art_a_list)

#define lists to store the data

a_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)
cs2_list = np.zeros(Nfiles)
cv2_list = np.zeros(Nfiles)

#An and Bn for the integral over the Green's function
An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
Pn = np.zeros(Nfiles)
Qn = np.zeros(Nfiles)


#the densitites
dk_sch_list = np.zeros(Nfiles)
dk_spt_list = np.zeros(Nfiles)
dk_order2_list = np.zeros(Nfiles)
dk_order3_list = np.zeros(Nfiles)
dk_order4_list = np.zeros(Nfiles)
dk_order5_list = np.zeros(Nfiles)
dk_order6_list = np.zeros(Nfiles)

#initial scalefactor
a0 = EFT_solve(0, Lambda, loc, run, EFT=1)[0]
m = 1 #this is the index in ctot2 = a^(m); for analytical tests

for file_num in range(0, Nfiles):
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a

   a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, tau_l, fit, sch_k, spt_k, ctot2, cs2, cv2, d_l, dv_l= EFT_solve(file_num, Lambda, loc, run, EFT=1)
   # a = art_a_list[file_num]
   # ctot2 = a**(m) #this is a test case for analytical verification; comment out for the real solution
   a_list[file_num] = a
   ctot2_list[file_num] = ctot2
   cs2_list[file_num] = cs2
   cv2_list[file_num] = cv2

   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0
      Pn[file_num] = ctot2 * (a**(5/2)) #for calculation of alpha_c
      # Pn[file_num] = ctot2 * (a**(-1/2)) #for the 1D green's function as in Pajer & Zaldarriaga
      Qn[file_num] = ctot2

      An[file_num] = da * Pn[file_num]
      Bn[file_num] = da * Qn[file_num]

   #we now extract the solutions for a specific mode
   dk_sch_list[file_num] = sch_k[mode]
   dk_spt_list[file_num] = spt_k[mode]

   dk_order2_list[file_num] = np.real(order_2)[mode]
   dk_order3_list[file_num] = np.real(order_3)[mode]
   dk_order4_list[file_num] = np.real(order_4)[mode]
   dk_order5_list[file_num] = np.real(order_5)[mode]
   dk_order6_list[file_num] = np.real(order_6)[mode]
   # dk_eft_hert_list[file_num] = np.real(eft_hert)[mode]

   print('a = ', a, '\n')

#define the full spt solution as the sum over all orders
dk_spt_sum_list = dk_order2_list + dk_order3_list + dk_order4_list #+ dk_order5_list + dk_order6_list

#A second loop for the integration
for j in range(1, Nfiles):
   An[j] += An[j-1]
   Bn[j] += Bn[j-1]

#calculation of the Green's function integral
C = 2 / (5 * H0**2)
An /= (a_list**(5/2))

# #for the 1D green's integral
# C = -2 / (H0**2)
# An /= (a_list**(3/2))

alpha_c_naive = C * (An - Bn)
alpha_c_guess = - (ctot2_list * a_list) / (9 * H0**2)

term1 = - (5 * C * a_list**(m+1)) / ((2*m + 7) * (m+1))
term2 = - C * (a_list[0]**(((2*m) + 7) / 2)) / (a_list**(5/2))
term3 = C * (a_list[0]**(m+1)) / (m+1)
alpha_c_an = term1 + term2 + term3
alpha_c_interp = interp_green(Pn, Qn, a_list, 0.01, C, simpsons=True)[1]

P_eft = dk_spt_sum_list + (2 * alpha_c_naive) * (mode**2) * dk_order2_list
P_eft_guess = dk_spt_sum_list + (2 * alpha_c_guess) * (mode**2) * dk_order2_list

P_eft_interp = dk_spt_sum_list[1:-1] + (2 * alpha_c_interp) * (mode**2) * dk_order2_list[1:-1]

#for 3 parameters a0, a1, a2 such that τ_l = a0 + a1 × (1 + δ_l) + a2 × dv_l
from scipy.optimize import curve_fit
def fitting_function(X, c, n):
   P_spt, P11, a = X
   return P_spt + ((c * (a**n)) * P11)

guesses = 1, 1
FF = curve_fit(fitting_function, (dk_spt_list, dk_order2_list, a_list), dk_sch_list, guesses, sigma=1e-5*np.ones(a_list.size), method='lm')
c, n = FF[0]
cov = FF[1]
err_c, err_n = np.sqrt(np.diag(cov))
fit = fitting_function((dk_spt_list, dk_order2_list, a_list), c, n)

# alpha_c_fit = (fit - dk_spt_sum_list) / (2 * dk_order2_list)

fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
ax[0].set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
# ax[0].set_title(r'$k = {}$'.format(mode))
# ax[0].set_ylabel(r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$', fontsize=14) # / a^{2}$')
# ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
ax[0].set_ylabel(r'$c_{{\mathrm{{tot}}}}^{{2}}\;[\mathrm{{km}}^{{2}}\mathrm{{s}}^{{-2}}]$', fontsize=14) # / a^{2}$')

ax[1].set_xlabel(r'$a$', fontsize=14)

# ax[0].plot(a_list, alpha_c_fit, label='fit', c='k', lw=2)
# ax[0].plot(a_list, alpha_c_naive, label='measured', ls='dashed', c='r', lw=2)

# ax[0].plot(a_list, cs2_list, c='b', ls='dashed', lw=2, label=r'$c^{2}_{\mathrm{s}}$')
# ax[0].plot(a_list, cv2_list, c='r', ls='dotted', lw=2, label=r'$c^{2}_{\mathrm{v}}$')
ax[0].plot(a_list, ctot2_list, c='k', lw=2, label=r'$c^{2}_{\mathrm{tot}}$')
# # # ax[0].plot(a_list, -10 * a_list, c='r', lw=2)


# ax[0].plot(a_list, dk_sch_list, label='Sch', lw=2.5, c='b')
# ax[0].plot(a_list, P_eft, label='EFT', ls='dashed', lw=2.5, c='k')
# ax[0].plot(a_list, dk_order2_list, label=r'$\mathcal{O}(2)$', ls='dashed', lw=2.5, c='r')
# ax[0].plot(a_list, dk_spt_sum_list, label=r'$\sum \mathcal{O}(6)$', ls='dotted', lw=2.5, c='brown')
# ax[0].plot(a_list, fit, label=r'fit', ls='dashed', lw=3, c='green')
# # ax[0].plot(a_list[1:-1], P_eft_interp, label='EFT interpolated', ls='dotted', lw=2.5, c='orange')
# # # ax[0].plot(a_list, P_eft_guess, label=r'guess', ls='dashed', lw=3, c='cyan')

#
# # ax[0].plot(a_list, dk_spt_list, label=r'SPT', ls='dashed', lw=2.5, c='yellow')
# # ax[0].plot(a_list, dk_order3_list, label=r'$\mathcal{O}(3)$', ls='dashed', lw=2.5, c='magenta')
# # ax[0].plot(a_list, dk_order4_list, label=r'$\mathcal{O}(4)$', ls='dashdot', lw=3, c='orange')
# # ax[0].plot(a_list, dk_order5_list, label=r'$\mathcal{O}(5)$', ls='dashed', lw=3, c='cyan')
# # ax[0].plot(a_list, dk_eft_bald_list, label='EFT Baldauf', ls='dashdot', lw=2.5, c='k')
# # ax[0].plot(a_list, cs2_list, c='b', lw=2)
# #
# # ax[0].plot(a_list, dk_order7_list, label=r'$\mathcal{O}(7)$', ls='dashed', lw=3, c='violet')
# # ax[0].plot(a_list, dk_order8_list, label=r'$\mathcal{O}(8)$', ls='dashdot', lw=3, c='magenta')
# # ax[0].plot(a_list, dk_3spt_list, label=r'$\delta_{3SPT}$', ls='dashed', lw=2.5, c='magenta')
#

#bottom panel; errors
err_order_2 = (dk_order2_list - dk_sch_list) * 100 / dk_sch_list
err_spt_all = (dk_spt_sum_list - dk_sch_list) * 100 / dk_sch_list
err_eft = (P_eft - dk_sch_list) * 100 / dk_sch_list
err_fit = (fit - dk_sch_list) * 100 / dk_sch_list

# err_eft_guess = (P_eft_guess - dk_sch_list) * 100 / dk_sch_list
#
# err_alpha_fit = (alpha_c_naive - alpha_c_fit) * 100 / alpha_c_fit
# err_eft_interp = (P_eft_interp - dk_sch_list[1:-1]) * 100 / dk_sch_list[1:-1]
#
# err_alpha_c = (alpha_c_naive[20:-1] - alpha_c_interp[19:]) * 100 / alpha_c_interp[19:]
# # err_order_3 = (dk_order3_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_4 = (dk_order4_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_5 = (dk_order5_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_6 = (dk_order6_list - dk_sch_list) * 100 / dk_sch_list

ax[1].axhline(0, color='b')
if mode == 1:
   ax[1].plot(a_list, err_order_2, ls='dashed', lw=2.5, c='r')

ax[1].plot(a_list, err_eft, ls='dashed', lw=2.5, c='k')
ax[1].plot(a_list, err_spt_all, ls='dotted', lw=2.5, c='brown')
ax[1].plot(a_list, fit, ls='dashed', lw=2.5, c='green')
# ax[1].plot(a_list, err_eft_guess, ls='dashed', lw=2.5, c='cyan')

# ax[1].plot(a_list[1:-1], err_eft_interp, ls='dotted', lw=2.5, c='orange')

# ax[1].plot(a_list, err_alpha_fit, ls='dashed', lw=2.5, c='red')
# ax[1].plot(a_list, err_order_4, ls='dashdot', lw=2.5, c='orange')
# ax[1].plot(a_list, err_order_5, ls='dashed', lw=2.5, c='cyan')
# # ax[1].plot(a_list, err_dk2_spt, ls='dashed', lw=2.5, c='yellow')
# ax[1].plot(a_list[100:-1], err_alpha_c[80:], ls='dashed', lw=2.5, c='k')

ax[1].set_ylabel('% err', fontsize=14)

ax[1].minorticks_on()

for i in range(2):
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].grid(lw=0.2, ls='dashed', color='grey')
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

# plt.savefig('../plots/sch_hfix_mix/eft_k{}_l{}.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# plt.savefig('../plots/sch_hfix_run19/alpha_c_l{}.png'.format(Lambda), bbox_inches='tight', dpi=120)
plt.savefig('../plots/sch_hfix_run19/ctot2_l{}_test_av.png'.format(Lambda), bbox_inches='tight', dpi=120)
# plt.show()
plt.close()

print('c = ', c)
print('n = ', n)
