#!/usr/bin/env python3
##saved on Nov 28, 14:21
import matplotlib.pyplot as plt
import h5py
import numpy as np
from EFT_solver_av import *
from functions import dn
from SPT import SPT_final

loc = '../'
run = '/sch_no_m_run5/'
index = 0
Nfiles = 764 - index
mode = 1
Lambda = 5
fit_list = np.zeros(Nfiles)
tau_list = np.zeros(Nfiles)
a_list = np.zeros(Nfiles)
dl_list = np.zeros(Nfiles)
dvl_list = np.zeros(Nfiles)

for file_num in range(0, Nfiles):
   #the function 'EFT_solve' return solutions of all modes + the EFT parameters
   ##the following line is to keep track of 'a' for the numerical integration
   a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, tau_l, fit, sch_k, spt_k, ctot2, ctot2_2, C1, C2, dl, dv_l = param_calc(file_num+index, Lambda, loc, run)
   print('a = ', a)
   Nx = x.size

   tau_k = np.fft.fft(tau_l) / Nx
   fit_k = np.fft.fft(fit) / Nx

   tau_2k = np.abs(tau_k * np.conj(tau_k))
   fit_2k = np.abs(fit_k * np.conj(fit_k))
   fit_list[file_num] = fit_2k[1]
   tau_list[file_num] = tau_2k[1]
   a_list[file_num] = a

   # # dl_list[file_num] = np.fft.fft(dl)[1] / Nx
   # # dvl_list[file_num] = np.fft.fft(dv_l)[1] / Nx
   #
   # # plt.title(r'$a = {}, \Lambda = {}$'.format(a, Lambda), fontsize=12)
   # # plt.grid(lw=0.2, color='grey', ls='dashed')
   # # plt.tick_params(axis='both', which='both', direction='in')
   # # plt.minorticks_on()
   # # plt.scatter(k, tau_k, label=r'Sch', s=30, c='b')
   # # plt.scatter(k, fit_k, label='fit', s=15, c='k')
   # # plt.xlim(-0.5, 12)
   # # plt.ylim(-200, 800)
   # #
   # # plt.legend(fontsize=11, loc=(1,1))
   # # plt.xlabel('k', fontsize=12)
   # # plt.ylabel(r'$\tau_{l}(k)$', fontsize=12)
   # #
   # # plt.savefig('../plots/sch_hfix_run19/k_MH2/MH2_{}.png'.format(file_num), bbox_inches='tight', dpi=120)
   # # break
   # # order_2[2:] = 0
   # # order_3[2:] = 0
   # # order_4[2:] = 0
   # # order_5[2:] = 0
   # # order_6[2:] = 0
   # # sum_246 = order_2 + order_4 + order_6
   # # dk2_sch[2:] = 0
   # # eft_hert[2:] = 0
   # #
   # # o2_x = np.real(np.fft.ifft(order_2)) * (Nx**2)
   # # o3_x = np.real(np.fft.ifft(order_3)) * (Nx**2)
   # # o4_x = np.real(np.fft.ifft(order_4)) * (Nx**2)
   # # o5_x = np.real(np.fft.ifft(order_5)) * (Nx**2)
   # # o6_x = np.real(np.fft.ifft(order_6)) * (Nx**2)
   # # eft_x = np.real(np.fft.ifft(eft_hert) * (Nx**2))
   # # sch = np.real(np.fft.ifft(dk2_sch) * (Nx**2))
   # # all_x = o2_x + o3_x + o4_x + o5_x + o6_x
   # lap_tau_l = spectral_calc(tau_l, k, o=2, d=0)
   #
   # from scipy.optimize import curve_fit
   # def fitting_function(X, c):
   #    x1, x2 = X
   #    #x1 is the original fit, x2 is lap_tau
   #    return x1 + c*x2
   #
   # guesses = 1e-5
   # FF = curve_fit(fitting_function, (fit, lap_tau_l), tau_l, guesses, sigma=1e-5*np.ones(x.size), method='lm')
   # c = FF[0]
   # cov = FF[1]
   # err_c = np.sqrt(np.diag(cov))
   # fit_new = fitting_function((fit, lap_tau_l), c)
   #
   # fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
   # # ax[0].set_title('a = {}, k = {}'.format(np.round(a,3), mode), fontsize=12)
   # ax[0].set_title('a = {}'.format(np.round(a,3), mode), fontsize=12)
   # # ax[0].set_ylabel(r'$|\delta(x)|^{2}$', fontsize=14)
   # # ax[0].set_ylabel(r'$|\tau_{l}(k)|^{2}$', fontsize=14)
   # ax[0].set_ylabel(r'$\tau_{l}(x)$', fontsize=14)
   #
   # # # ax[0].set_ylabel(r'$M^{(0)}_{H}$', fontsize=14)
   # # ax[0].set_xlim(0.5, 12.5)
   # # ax[0].set_ylim(-0.1e-6, 1e-6)
   #
   # ax[0].plot(x, tau_l, label=r'$M_{H}^{(2)}$', lw=2.5, c='b')
   # ax[0].plot(x, fit, label=r'fit to $M_{H}^{(2)}$; w/o $\Delta \tau$', lw=2.5, c='k', ls='dashed')
   # # ax[0].plot(x, tau_l - fit,  label=r'$\Theta$', lw=2.5, c='k')#, ls='dashed')
   #
   # # ax[0].plot(x, fit_new, label=r'fit with $\Delta \tau$', ls='dotted', lw=2.5, c='r')
   # # # # # ax[0].plot(x, MH0_spt, label=r'$M_{H}^{(0)_{\mathrm{SPT}}}$', ls='dashed', lw=2.5, c='r')
   #
   # # ax[0].scatter(k[1:], tau_2k[1:], label=r'$\tau_{l}$ from Sch', s=35, c='b')
   # # ax[0].scatter(k[1:], fit_2k[1:], label=r'fit to $\tau_{l}$', s=20, c='k')
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
   # err_new = (fit_new - tau_l) * 100 / tau_l
   #
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
   # # ax[1].plot(x, err_new, ls='dotted', lw=2.5, c='r')
   #
   # # ax[1].plot(x, -lap_tau_l / np.max(lap_tau_l), ls='dashed', lw=2.5, c='r')
   #
   # # ax[1].plot(x, err_spt, ls='dashdot', lw=2.5, c='brown')
   # # ax[1].plot(x, err_spt_den, lw=2.5, ls='dashed', c='r')
   #
   # ax[1].set_xlabel(r'$x$', fontsize=14)
   # ax[1].set_ylabel('% err', fontsize=14)
   #
   # # ax[1].set_ylabel('difference', fontsize=14)
   #
   # # err_fit_k = (tau_2k - fit_2k) #* 100 / tau_2k
   # # ax[1].scatter(k, err_fit_k, s=20, c='k')
   # # ax[1].set_xlabel(r'$k$', fontsize=14)
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
   # plt.savefig('../plots/sch_hfix_run19/theta/Theta_l{}_{}.png'.format(Lambda, file_num), bbox_inches='tight', dpi=120)
   # # plt.savefig('../plots/sch_hfix_run19/mode_ev/den_{}.png'.format(file_num), bbox_inches='tight', dpi=120)
   # # plt.show()
   # plt.close()
   # # break


fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
ax[0].set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
# ax[0].set_title(r'$k = {}$'.format(mode))
# ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
# ax[0].set_ylabel(r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$', fontsize=14) # / a^{2}$')
# ax[0].set_ylabel(r'$c_{{\mathrm{{tot}}}}^{{2}}\;[\mathrm{{km}}^{{2}}\mathrm{{s}}^{{-2}}]$', fontsize=14) # / a^{2}$')
ax[0].set_ylabel(r'$|\tau_{l}(k)|^{2}$', fontsize=14)

ax[1].set_xlabel(r'$a$', fontsize=14)

# ax[0].plot(a_list, alpha_c_fit, label='fit', c='k', lw=2)
# ax[0].plot(a_list, alpha_c_naive, label='measured', ls='dashed', c='r', lw=2)

# ax[0].plot(a_list, ctot2_list, c='k', lw=2)

ax[0].plot(a_list, np.log(tau_list), label='Sch', lw=2.5, c='b')
ax[0].plot(a_list, np.log(fit_list), label='fit', ls='dashed', lw=2.5, c='k')
# ax[0].plot(a_list, fit_k, label='fit in k-space', ls='dashed', lw=2.5, c='r')

err_fit = (fit_list - tau_list) * 100 / tau_list
# err_k_fit = (fit_k - tau_list) * 100 / tau_list

#bottom panel; errors
ax[1].axhline(0, color='b')

ax[1].plot(a_list, err_fit, ls='dashed', lw=2.5, c='k')
# ax[1].plot(a_list, err_k_fit, ls='dashed', lw=2.5, c='r')

ax[1].set_ylabel('% err', fontsize=14)
# ax[1].set_ylim(-100, 500)
ax[1].minorticks_on()

for i in range(2):
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].grid(lw=0.2, ls='dashed', color='grey')
    ax[i].yaxis.set_ticks_position('both')

ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

plt.savefig('../plots/sch_no_m_run5/tau_fit_k1_l{}.png'.format(Lambda), bbox_inches='tight', dpi=120)
# print(err_fit)
# plt.savefig('../plots/sch_hfix_run19/eft_k{}_l{}_fit.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# plt.savefig('../plots/sch_hfix_run19/alpha_c_l{}.png'.format(Lambda), bbox_inches='tight', dpi=120)
# plt.savefig('../plots/sch_hfix_run19/ctot2_l{}_test.png'.format(Lambda), bbox_inches='tight', dpi=120)

plt.close()
