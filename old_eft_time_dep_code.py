#!/usr/bin/env python3
##saved on Nov 28, 14:21
import matplotlib.pyplot as plt
import h5py
import numpy as np
from EFT_solver import *
from SPT import SPT_final
# plt.style.use('clean_1d')

loc = '../'
run = '/sch_hfix_run19_new/'
Nfiles = 580
mode = 1
Lambda = 5
a_list = np.zeros(Nfiles)
cs2_list = np.zeros(Nfiles)

dk_sch_list = np.zeros(Nfiles)
dk_spt_list = np.zeros(Nfiles)
dk_order2_list = np.zeros(Nfiles)
dk_order3_list = np.zeros(Nfiles)
dk_order4_list = np.zeros(Nfiles)
dk_order5_list = np.zeros(Nfiles)
dk_order6_list = np.zeros(Nfiles)
integrand_list = np.zeros(Nfiles)
ctot2_list = np.zeros(Nfiles)

# dk_order6_list = np.zeros(Nfiles)
# dk_order7_list = np.zeros(Nfiles)
# dk_order8_list = np.zeros(Nfiles)
# dk_spt_sum_list = np.zeros(Nfiles)
# dk_3spt_list = np.zeros(Nfiles)
# o2 = np.zeros(Nfiles)

# dk_1spt_list = np.zeros(Nfiles)
# dk_3spt_list = np.zeros(Nfiles)
dk_eft_bald_list = np.zeros(Nfiles)
dk_eft_hert_list = np.zeros(Nfiles)
# dk_5spt_list = np.zeros(Nfiles)

def dot(f1, f2):
   fk1 = np.fft.fft(f1)
   fk2 = np.fft.fft(f2)
   return np.abs(fk1 * fk2) / (fk1.size * fk2.size) #np.abs(np.fft.fft(f1) * np.conj(np.fft.fft(f2)) / f1.size / f2.size)

from functions import dn

a0 = EFT_solve(0, Lambda, loc, run, EFT=1)[0]
# an = EFT_solve(380, Lambda, loc, run, EFT=1)[0]
# print(an)

An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
greens_list = np.zeros(Nfiles)
for file_num in range(0, Nfiles):
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   if file_num > 0:
      a0 = a
   try:
      a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, eft_hert, tau_l, fit, sch_k, spt_k, ctot2, C1, C2, d3k = EFT_solve(file_num, Lambda, loc, run, EFT=1)
   except:
      pass

   ctot2 = a #this is a test case for analytical verification; comment out for the real solution
   ##here, we perform the numerical integration over the Green's function (see Baldauf's review eq. 7.157, or eq. 2.48 in Mcquinn & White)
   if file_num > 0:
      da = a - a0
      An[file_num] = da * ctot2 * (a**(5/2))
      # Bn[file_num] = Bn[file_num-1] + (da * ctot2 * a)
   # print('An = ', An)
   # print('Bn = ', Bn)
   # order_2[2:] = 0
   # order_3[2:] = 0
   # order_4[2:] = 0
   # order_5[2:] = 0
   # order_6[2:] = 0
   # sum_246 = order_2 + order_4 + order_6
   # dk2_sch[2:] = 0
   # eft_hert[2:] = 0
   #
   # o2_x = np.real(np.fft.ifft(order_2)) * (Nx**2)
   # o3_x = np.real(np.fft.ifft(order_3)) * (Nx**2)
   # o4_x = np.real(np.fft.ifft(order_4)) * (Nx**2)
   # o5_x = np.real(np.fft.ifft(order_5)) * (Nx**2)
   # o6_x = np.real(np.fft.ifft(order_6)) * (Nx**2)
   # eft_x = np.real(np.fft.ifft(eft_hert) * (Nx**2))
   # sch = np.real(np.fft.ifft(dk2_sch) * (Nx**2))
   # all_x = o2_x + o3_x + o4_x + o5_x + o6_x
   #
   # fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
   # # ax[0].set_title('a = {}, k = {}'.format(np.round(a,3), mode), fontsize=12)
   # ax[0].set_title('a = {}'.format(np.round(a,3), mode), fontsize=12)
   # # ax[0].set_ylabel(r'$|\delta(x)|^{2}$', fontsize=14)
   # ax[0].set_ylabel(r'$M^{(2)}_{H}$', fontsize=14)
   # # # ax[0].set_ylabel(r'$M^{(0)}_{H}$', fontsize=14)
   # #
   # ax[0].plot(x, tau_l, label=r'$M_{H}^{(2)}$', lw=2.5, c='b')
   # ax[0].plot(x, fit, label=r'fit to $M_{H}^{(2)}$', lw=2.5, c='k', ls='dashed')
   # # # ax[0].plot(x, MH0, label=r'$M_{H}^{(0)}$', lw=2.5, c='b')
   # # # ax[0].plot(x, MH0_spt, label=r'$M_{H}^{(0)_{\mathrm{SPT}}}$', ls='dashed', lw=2.5, c='r')
   #
   #
   # # ax[0].plot(x, sch, label='Sch', lw=2.5, c='b')
   # # ax[0].plot(x, o2_x, label=r'SPT; $\mathcal{O}(2)$', ls='dashed', lw=2.5, c='r')
   # # ax[0].plot(x, o4_x, label=r'$\mathcal{O}(4)$', ls='dashed', lw=3, c='cyan')
   # # ax[0].plot(x, o6_x, label=r'$\mathcal{O}(6)$', ls='dotted', lw=3, c='green')
   # # ax[0].plot(x, eft_x, label=r'EFT', ls='dashed', lw=2.5, c='k')
   # # ax[0].plot(x, all_x, label=r'SPT; $\sum \mathcal{O}(6)$', ls='dashdot', lw=2.5, c='brown')
   # # # ax[0].plot(x, o3_x, label=r'$\mathcal{O}(3)$', ls='dotted', lw=2.5, c='cyan')
   # # # ax[0].plot(x, o5_x, label=r'$\mathcal{O}(5)$', ls='dashed', lw=3, c='brown')
   # # err_o2 = (o2_x - sch) #* 100 / sch
   # # err_o4 = (o4_x - sch) #* 100 / sch
   # # err_o6 = (o6_x - sch) #* 100 / sch
   # # err_eft = (eft_x - sch) #* 100 / sch
   # # err_spt = (all_x - sch) #* 100 / sch
   #
   # err_eft = (fit - tau_l) * 100 / tau_l
   # # # ax[0].set_ylim(-10, -2)
   # # err_spt_den = (MH0_spt - MH0) * 100 / MH0
   #
   # #bottom panel; errors
   # ax[1].axhline(0, color='b')
   # # ax[1].plot(x, err_o2, ls='dashed', lw=2.5, c='r')
   # # ax[1].plot(x, err_o4, ls='dashed', lw=2.5, c='cyan')
   # # ax[1].plot(x, err_o6, ls='dotted', lw=2.5, c='green')
   #
   # ax[1].plot(x, err_eft, ls='dashed', lw=2.5, c='k')
   # # ax[1].plot(x, err_spt, ls='dashdot', lw=2.5, c='brown')
   # # ax[1].plot(x, err_spt_den, lw=2.5, ls='dashed', c='r')
   #
   #
   # ax[1].set_xlabel(r'$x$', fontsize=14)
   # # ax[1].set_ylabel('difference', fontsize=14)
   # ax[1].set_ylabel('% err', fontsize=14)
   #
   # ax[1].minorticks_on()
   #
   # for i in range(2):
   #     ax[i].tick_params(axis='both', which='both', direction='in')
   #     ax[i].ticklabel_format(scilimits=(-2, 3))
   #     ax[i].grid(lw=0.2, ls='dashed', color='grey')
   #     ax[i].yaxis.set_ticks_position('both')
   #
   # ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #
   # plt.savefig('../plots/sch_hfix_run19/MH2/MH2_{}.png'.format(file_num), bbox_inches='tight', dpi=120)
   # # plt.savefig('../plots/sch_hfix_run19/mode_ev/den_{}.png'.format(file_num), bbox_inches='tight', dpi=120)
   #
   # plt.close()
   # # break

   # #we now extract the solutions for a specific mode
   # dk_sch_list[file_num] = (np.abs(dk_sch)**2)[mode]
   # dk_1spt_list[file_num] = dk_1spt[mode] #np.abs(dk_1spt[mode])**2 #this is a*δ_{1}
   # dk_2spt_list[file_num] = dk_2spt[mode] #np.abs(dk_2spt[mode])**2 #this is a*δ_{1} + a^{2}*δ_{2}
   # dk_3spt_list[file_num] =  dk_3spt[mode] #np.abs(dk_3spt[mode])**2
   # dk_4spt_list[file_num] = dk_4spt[mode] #np.abs(dk_4spt[mode])**2
   # # dk_5spt_list[file_num] =  dk_5spt[mode] #np.abs(dk_5spt[mode])**2
   # dk_eft_list[file_num] = dk_eft[mode]#(np.abs(dk_eft)**2)[mode]
   # a_list[file_num] = a
   # print('a = ', a)

   #, dk_4spt, dk_5spt, dk_eft
   # a, x, k, dk_sch, d1, d2, d3, d4, F = EFT_solve(file_num, Lambda, loc, run, EFT=1)
   # Nx = x.size
   # #we now extract the SPT solutions for a specific order
   # order_2 = dot(d1, d1) * (a**2)
   # order_3 = (2 * dot(d1, d2)) * (a**3)
   # order_4 = (dot(d2, d2) + (2 * dot(d1, d3))) * (a**4)
   # order_5 = (2 * dot(d2, d3)) * (a**5) # + 2 * dot(d1, d4) #δ_1 * δ_4 term doesn't exist in (δ_3SPT)^2
   # order_6 = (dot(d3, d3)) * (a**6) #+ (2 * dot(d2, d4))
   # order_7 = (2 * dot(d3, d4)) * (a**7)
   # order_8 = dot(d4, d4) * (a**8)
   # sum_all = order_2 + order_3 + order_4 + order_5 + order_6 #+ order_7 + order_8
   #
   # d1k = np.fft.fft(d1) / Nx
   # d2k = np.fft.fft(d2) / Nx
   # d3k = np.fft.fft(d3) / Nx
   #
   # sum_orders = np.abs(d1k * np.conj(d1k)) * (a**2) + np.abs(2 * d1k * np.conj(d2k)) * (a**3) + np.abs((d2k * np.conj(d2k)) + np.abs(2 * d1k * np.conj(d3k))) * (a**4) + np.abs(2 * d2k * d3k) * (a**5) + np.abs(d3k * np.conj(d3k)) * (a**6)
   # # sum_orders = (a * d1k) + ((a**2) * d2k) + ((a**3) * d3k)
   # o2[file_num] = sum_orders[mode] #(np.abs((sum_orders))**2)[mode]

   # G = SPT_final(F, a)
   # o2[file_num] = (np.abs(np.fft.fft(G[2]) / Nx)**2)[mode]

   #we now extract the solutions for a specific mode
   dk_sch_list[file_num] = sch_k[mode]
   dk_spt_list[file_num] = spt_k[mode]

   dk_order2_list[file_num] = np.real(order_2)[mode]
   dk_order3_list[file_num] = np.real(order_3)[mode]
   dk_order4_list[file_num] = np.real(order_4)[mode]
   dk_order5_list[file_num] = np.real(order_5)[mode]
   dk_order6_list[file_num] = np.real(order_6)[mode]
   # dk_eft_hert_list[file_num] = np.real(eft_hert)[mode]
   # dk_eft_bald_list[file_num] = np.real(eft_bald)[mode]

   a_list[file_num] = a
   cs2_list[file_num] = ctot2
   # C1_list[file_num] = C1
   # C2_list[file_num] = C2

   print('a = ', a, '\n')

##calculation of the green's function integral
dk_spt_sum_list = dk_order2_list + dk_order3_list + dk_order4_list + dk_order5_list + dk_order6_list
# H0 = 100
# C = 2 / (5 * H0**2)
# alpha_c = C * ((An / a_list**(5/2)) - Bn)
# G = C * ((a_list**(3/2) / a) - (a /(a_list)))
An /= (a_list**(5/2))
# alpha_c = - a_list * cs2_list / (9 * (H0**2)) #approximation
# alpha_c = 2.15e-4 * a_list**2 / 2 #fit
# P_corr = (2 * alpha_c) * (mode**2) * dk_order2_list
# P_eft = dk_spt_sum_list + (2 * alpha_c) * (mode**2) * dk_order2_list
from scipy.interpolate import interp1d
from scipy.integrate import simpson

da = (30 - 0.1) / 10000 #a_list.size
a_new = np.arange(0.1, 30, da)

lin_func = interp1d(x=a_list, y=integrand_list, kind='linear')
lin_interp = lin_func(a_new)

simpson(lin_interp)

fit_to_An = (2/9) * (a_list**2)

plt.plot(a_list, integrand_list, 'o', c='k', label='data')
plt.plot(a_new, lin_interp, c='b', lw=2, label='linear interpolation')

plt.legend()
plt.savefig('../plots/sch_hfix_run19/integral.png', bbox_inches='tight', dpi=120)

# plt.plot(a_list, greens_list, c='g', label='green')
# plt.plot(a_list, ctot2_list, c='b', ls='dashed', label='ctot2')
# plt.legend()

# An_err = (An - fit_to_An)
#
# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
# ax[0].set_title('Accuracy of numerical integration')
# ax[0].plot(a_list, fit_to_An, c='k', lw=2, label=r'exact')
# ax[0].plot(a_list, An, c='r', lw=2, ls='dashed', label=r'first order integration')
# ax[1].plot(a_list, An_err, ls='dashed', lw=2, c='r')
# ax[1].plot(a_list, np.zeros(a_list.size), c='k', lw=2)
#
# ax[0].set_ylabel(r'$A_{n}$', fontsize=14)
# ax[1].set_xlabel('a', fontsize=14)
# ax[1].set_ylabel('difference', fontsize=14)
#
# ax[1].minorticks_on()
#
# for i in range(2):
#     ax[i].tick_params(axis='both', which='both', direction='in')
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].grid(lw=0.2, ls='dashed', color='grey')
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/sch_hfix_run32/An.png', bbox_inches='tight', dpi=120)

# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
# ax[0].set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
# # ax[0].set_title(r'$k = {}$'.format(mode))
# ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$', fontsize=14) # / a^{2}$')
#
# ax[1].set_xlabel(r'$a$', fontsize=14)
#
# # ax[0].plot(a_list, cs2_list / a_list**2, c='k', lw=2)
# # ax[0].plot(a_list, alpha_c, c='b', lw=2)
#
# ax[0].plot(a_list, dk_sch_list, label='Sch', lw=2.5, c='b')
# # ax[0].plot(a_list, P_eft, label='EFT', ls='dashed', lw=2.5, c='k')
# ax[0].plot(a_list, dk_order2_list, label=r'$\mathcal{O}(2)$', ls='dashed', lw=2.5, c='r')
# ax[0].plot(a_list, dk_spt_sum_list, label=r'$\sum \mathcal{O}(6)$', ls='dotted', lw=2.5, c='brown')
#
# # ax[0].plot(a_list, dk_spt_list, label=r'SPT', ls='dashed', lw=2.5, c='yellow')
# # ax[0].plot(a_list, dk_order3_list, label=r'$\mathcal{O}(3)$', ls='dashed', lw=2.5, c='magenta')
# # ax[0].plot(a_list, dk_order4_list, label=r'$\mathcal{O}(4)$', ls='dashdot', lw=3, c='orange')
# # ax[0].plot(a_list, dk_order5_list, label=r'$\mathcal{O}(5)$', ls='dashed', lw=3, c='cyan')
# # ax[0].plot(a_list, dk_order6_list, label=r'$\mathcal{O}(6)$', ls='dotted', lw=3, c='green')
# # ax[0].plot(a_list, dk_eft_bald_list, label='EFT Baldauf', ls='dashdot', lw=2.5, c='k')
# # ax[0].plot(a_list, cs2_list, c='b', lw=2)
# #
# # ax[0].plot(a_list, dk_order7_list, label=r'$\mathcal{O}(7)$', ls='dashed', lw=3, c='violet')
# # ax[0].plot(a_list, dk_order8_list, label=r'$\mathcal{O}(8)$', ls='dashdot', lw=3, c='magenta')
# # ax[0].plot(a_list, dk_3spt_list, label=r'$\delta_{3SPT}$', ls='dashed', lw=2.5, c='magenta')
#
# # err_eft = (P_eft - dk_sch_list) * 100 / dk_sch_list
# err_order_2 = (dk_order2_list - dk_sch_list) * 100 / dk_sch_list
# err_spt_all = (dk_spt_sum_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_3 = (dk_order3_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_4 = (dk_order4_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_5 = (dk_order5_list - dk_sch_list) * 100 / dk_sch_list
# # err_order_6 = (dk_order6_list - dk_sch_list) * 100 / dk_sch_list
# #
# # # err_dk2_spt = (dk_spt_list - dk_sch_list) * 100 / dk_sch_list
# #
# # # err_eft = (dk_eft_bald_list - dk_eft_hert_list)
#
# #bottom panel; errors
# ax[1].axhline(0, color='b')
# if mode == 1:
#    ax[1].plot(a_list, err_order_2, ls='dashed', lw=2.5, c='r')
#
# # ax[1].plot(a_list, err_eft, ls='dashdot', lw=2.5, c='k')
# ax[1].plot(a_list, err_spt_all, ls='dotted', lw=2.5, c='brown')
# # ax[1].plot(a_list, err_order_3, ls='dashed', lw=2.5, c='magenta')
# # ax[1].plot(a_list, err_order_4, ls='dashdot', lw=2.5, c='orange')
# # ax[1].plot(a_list, err_order_5, ls='dashed', lw=2.5, c='cyan')
# # ax[1].plot(a_list, err_order_6, ls='dotted', lw=2.5, c='green')
#
# # ax[1].plot(a_list, err_dk2_spt, ls='dashed', lw=2.5, c='yellow')
#
#
# ax[1].set_ylabel('% err', fontsize=14)
#
# ax[1].minorticks_on()
#
# for i in range(2):
#     ax[i].tick_params(axis='both', which='both', direction='in')
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].grid(lw=0.2, ls='dashed', color='grey')
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/sch_hfix_run32/eft_k{}_l{}.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# # plt.savefig('../plots/sch_hfix_run19/alpha_c.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
#
# plt.close()

# fig, ax = plt.subplots()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.set_title(r'$k = {}$'.format(mode))
# ax.set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
#
# ax.set_xlabel(r'$a$', fontsize=14)
# # ax.plot(a_list, (spt1), c='r', label=r'$\delta_{1}$', lw=2, ls='dotted')
# # ax.plot(a_list, (spt_diff_21), c='cyan', label='$\delta_{2}$', lw=2, ls='dashed')
# # ax.plot(a_list, (spt_diff_32), c='k', label='$\delta_{3}$', lw=2, ls='dashdot')
# # ax.plot(a_list, (spt_diff_43), c='brown', label='$\delta_{4}$', lw=2, ls='dashed')
# # ax.plot(a_list, (spt_diff_54), c='green', label='$\delta_{5}$', lw=2, ls='dotted')
#
# # ax.plot(a_list, order_, c='r', label=r'$\delta_{1}$', lw=2, ls='dotted')
# ax.plot(a_list, np.log(P11), c='r', label='$P_{11}$', lw=2)#, ls='dashed')
# # ax.plot(a_list, P12, c='k', label='$P_{12}$', lw=2)#, ls='dashdot')
# ax.plot(a_list, np.log(P22 + P31), c='b', label='$P_{22} + P_{31}$', lw=2)#, ls='dashed')
#
#
# plt.legend()
# plt.savefig('./plots/sch_hfix_run17/orders_log.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)

# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
# ax[0].set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
# # ax[0].set_title(r'$k = {}$'.format(mode))
# ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
#
# ax[0].set_yscale('log')
# ax[0].plot(a_list, dk_sch_list, label='Sch', lw=2.5, c='b')
# ax[0].plot(a_list, dk_1spt_list, label=r'1SPT', ls='dashed', lw=2.5, c='r')
# ax[0].plot(a_list, dk_2spt_list, label=r'2SPT', ls='dotted', lw=2.5, c='cyan')
# ax[0].plot(a_list, dk_3spt_list, label=r'3SPT', ls='dashdot', lw=3, c='k')
# ax[0].plot(a_list, dk_eft_list, label=r'EFT', ls='dashed', lw=3, c='orange')
#
# # ax[0].plot(a_list, dk_order5_list, label=r'$\mathcal{O}(5)$', ls='dashed', lw=3, c='brown')
# # ax[0].plot(a_list, dk_order6_list, label=r'$\mathcal{O}(6)$', ls='dotted', lw=3, c='green')
# # ax[0].plot(a_list, dk_order7_list, label=r'$\mathcal{O}(7)$', ls='dashed', lw=3, c='violet')
# # ax[0].plot(a_list, dk_order8_list, label=r'$\mathcal{O}(8)$', ls='dashdot', lw=3, c='magenta')
#
#
# err_1spt = (dk_1spt_list - dk_sch_list) * 100 / dk_sch_list
# err_2spt = (dk_2spt_list - dk_sch_list) * 100 / dk_sch_list
# err_3spt = (dk_3spt_list - dk_sch_list) * 100 / dk_sch_list
# err_eft = (dk_eft_list - dk_sch_list) * 100 / dk_sch_list
# # err_4spt = (dk_4spt_list - dk_sch_list) * 100 / dk_sch_list
# # err_5spt = (dk_5spt_list - dk_sch_list) * 100 / dk_sch_list
#
# # ax[0].set_ylim(-10, -2)
#
# #bottom panel; errors
# ax[1].axhline(0, color='b')
# # ax[1].plot(a_list, err_1spt, ls='dotted', lw=2.5, c='r')
# ax[1].plot(a_list, err_2spt, ls='dashed', lw=2.5, c='cyan')
# ax[1].plot(a_list, err_3spt, ls='dashdot', lw=2.5, c='k')
# ax[1].plot(a_list, err_eft, ls='dashed', lw=2.5, c='orange')
# # ax[1].plot(a_list, err_4spt, ls='dashed', lw=2.5, c='brown')
# # ax[1].plot(a_list, err_5spt, ls='dotted', lw=2.5, c='green')
#
# ax[1].set_xlabel(r'$a$', fontsize=14)
# ax[1].set_ylabel('% err', fontsize=14)
#
# ax[1].minorticks_on()
#
# for i in range(2):
#     ax[i].tick_params(axis='both', which='both', direction='in')
#     # ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].grid(lw=0.2, ls='dashed', color='grey')
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[0].legend(fontsize=14, loc=2, bbox_to_anchor=(1,1))
# plt.savefig('./plots/sch_hfix_run17/eft_k{}_l{}.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# plt.show()


##pasting here on Nov 29, 11:11
##a snippet of code for testing numerical integration
# da_s = [0.1, 0.01] #, 1e-4]
# colours = ['orange', 'brown', 'cyan']
# colours_simp = ['r', 'cyan', 'cyan']
#
# fig, ax = plt.subplots()
# ax.set_title(r"$\alpha_{{c}}$ integral; test case: $c^{{2}}_{{\mathrm{{tot}}}} = (a')^{{m}}$, $m = {}$".format(m))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.tick_params(axis='both', which='both', direction='in')
# ax.minorticks_on()
#
# ax.plot(a_list, alpha_c_an, lw=2, c='k', label='analytical')
# # ax.plot(a_list, error_term, lw=2, c='r', ls='dotted', label='error term')
# ax.plot(a_list, alpha_c_naive, lw=2, c='b', ls='dashed', label='naive approximation')
# for j in range(len(da_s)):
#    alpha_c_interp = interp_green(Pn, Qn, a_list, da_s[j], C, simpsons=False)
#    ax.plot(alpha_c_interp[0], alpha_c_interp[1], lw=2, c=colours[j], ls='dashed', label=r'interpolated naive; $\Delta a = {}$'.format(da_s[j]))
#
#    alpha_c_interp = interp_green(Pn, Qn, a_list, da_s[j], C, simpsons=True)
#    ax.plot(alpha_c_interp[0], alpha_c_interp[1], lw=2, c=colours_simp[j], ls='dotted', label=r"interpolated Simpson's; $\Delta a = {}$".format(da_s[j]))
#
#
# ax.set_xlabel(r'$a$', fontsize=14)
# ax.set_ylabel(r'$\alpha_{c}(a)$', fontsize=14)
# plt.legend()
# plt.savefig('../plots/sch_hfix_run19/alpha_c.png', bbox_inches='tight', dpi=120)

# 
# psi_star = np.conj(psi)
# grad_psi = spectral_calc(psi, k, o=1, d=0)
# grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
# lap_psi = spectral_calc(psi, k, o=2, d=0)
# lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)
#
# sigma_x = 20 * dx
# sigma_p = h / (2 * sigma_x)
# sm = 1 / (4 * (sigma_x**2))
# W_k_an = np.exp(- (k ** 2) / (4 * sm))
#
# #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
# MW_0 = np.abs(psi ** 2) #* (m / a**3) #this is now ρ(x) due to multiplication by a**3 / m
# MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi))) #/ (a**4)
# MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))) #/ (m * (a**5))
#
# MH_0_k = np.fft.fft(MW_0) * W_k_an
# MH_0 = np.real(np.fft.ifft(MH_0_k))
#
# MH_1_k = np.fft.fft(MW_1) * W_k_an
# MH_1 = np.real(np.fft.ifft(MH_1_k))
#
# MH_2_k = np.fft.fft(MW_2) * W_k_an
# MH_2 = np.real(np.fft.ifft(np.fft.fft(MW_2) * W_k_an)) + ((sigma_p**2) * MH_0)
#
#
# p_max = 0.1
# dv = 2 * p_max / (Nx)
# p = np.arange(-p_max, p_max, dv) #np.sort(k * h)
#
# print(sigma_x, sigma_p)
# print(dx, dv)
#
# from functions import husimi
# [X, P] = np.meshgrid(x, p)
# F = husimi(psi, X, P, sigma_x, h, L)
# mom = MW_1 / MW_0
# vel = mom / (m * a0)
#
# v_pec = MH_1 / MH_0 / (m * a0)
# #
# # fig, ax = plt.subplots()
# # ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=20)
# # ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=20)
# # title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a0, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
# # plot2d_2 = ax.pcolormesh(x, p, F, shading='auto', cmap='inferno')
# #
# # ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
# # c = fig.colorbar(plot2d_2, fraction=0.15)
# # c.set_label(r'$f_{H}$', fontsize=20)
# # ax.plot(x, v_zel * (m * a0), c='k', label='v_zel')
# # ax.plot(x, p_pec, c='r', ls='dashed', label='v_pec')
# #
# # # ax.set_ylim(-20, 20)
# # ax.legend(loc='upper right')
# # legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
# # frame = legend.get_frame()
# # plt.tight_layout()
# # frame.set_edgecolor('white')
# # frame.set_facecolor('black')
# # for text in legend.get_texts():
# #     plt.setp(text, color = 'w')
#
# F /= np.mean(np.sum(F, axis=0))
# M0 = np.sum(F, axis=0) #* (m * (a0**3))
# M1 = np.sum(F * P, axis=0) #this produces a 'velocity density' as opposed to a momentum density
# M2 = np.sum(F * (P ** 2), axis=0) #this is the second moment, i.e., the Schrodinger stress
#
# print(np.mean(M0))
# print(np.mean(M2))
# div = 5000 #np.mean(M2)
# # M0 /= np.mean(M0)
# # M1 /= np.mean(M1)
# # M2 /= np.mean(M0)
#
# # v_f = M1 / M0 / (m * a0)
# # # # plt.plot(x, nd, c='k')
# # # plt.plot(x, v_f, c='k')
# # # plt.plot(x, v_pec, c='brown', ls='dashdot')
# # # plt.plot(x, v_zel, c='b', ls='dashed')
# # # plt.plot(x, vel, c='r', ls='dotted')
# # # plt.scatter(k, W_k_an, c='b', s=50)
# # # plt.scatter(k, W_k_an_1, c='k', s=20)
# #
# # C2 = M2 - (M1**2 / M0)
# # CH_2 = MH_2 - (MH_1**2 / MH_0)
# #
# # plt.plot(x, CH_2, c='k')
# # plt.plot(x, C2, c='brown', ls='dashdot')
# #
# # plt.savefig('/vol/aibn31/data1/mandar/plots/ps.png')
# # plt.close()
#
#
# # M0 /= np.mean(M0)
# # M1 /= np.mean(M0)
# # M1 *= (m * (a0**3))
#
# # M2 /= np.mean(M0)
# C1 = M1 / M0 #* (a0**4)
# C2 = (M2 - (M1**2 / M0))
# CH_1 = MH_1 / MH_0
# CH_2 = (MH_2 - (MH_1**2 / MH_0)) * (np.mean(MH_0))
#
# # #
# plt.plot(x, MH_0, c='k', lw=2, label=r'from $\Psi$')
# plt.plot(x, M0, c='b', ls='dashed', lw=2, label=r'from $f$')
# plt.legend()
# plt.savefig('/vol/aibn31/data1/mandar/plots/M0.png')
# plt.close()
#
# plt.plot(x, MH_1, c='k', lw=2, label=r'from $\Psi$')
# plt.plot(x, M1, c='b', ls='dashed', lw=2, label=r'from $f$')
# plt.legend()
# plt.savefig('/vol/aibn31/data1/mandar/plots/M1.png')
# plt.close()
# #
# plt.plot(x, MH_2, c='k', lw=2, label=r'from $\Psi$')
# plt.plot(x, M2, c='b', ls='dashed', lw=2, label=r'from $f$')
# plt.legend()
# plt.savefig('/vol/aibn31/data1/mandar/plots/M2.png')
#
