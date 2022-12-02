#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import * #read_sim_data, spectral_calc, AIC, BIC, binning
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

pars_list = [3, 6]
# pars_list = [1, 3, 5, 6, 8, 12]
fits, aics = [], []
nbins_x, nbins_y = 15, 15

for npars in pars_list:
    file_nums = [0, 13, 23]
    sol_0 = binning(file_nums[0], path, Lambda, kind, nbins_x, nbins_y, npars)
    sol_1 = binning(file_nums[1], path, Lambda, kind, nbins_x, nbins_y, npars)
    sol_2 = binning(file_nums[2], path, Lambda, kind, nbins_x, nbins_y, npars)

    x = sol_0[1]
    a_list = [sol_0[0], sol_1[0], sol_2[0]]
    tau_list = [sol_0[2], sol_1[2], sol_2[2]]
    aic_list = [sol_0[12], sol_1[12], sol_2[12]]
    fit_list = [sol_0[15], sol_1[15], sol_2[15]]

    # print('npars = {}'.format(npars))
    fits.append(fit_list)
    aics.append(aic_list)
    # a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned

fde = []
for j in file_nums:
    dc_l, dv_l, tau_l = read_sim_data(path, Lambda, kind, j)[3:6]
    C_ = deriv_param_calc(dc_l, dv_l, tau_l)[0]
    fde.append(C_[0] + C_[1]*dc_l + C_[2]*dc_l**2)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
ax[0].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
ax[2].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
ax[2].yaxis.set_label_position('right')
# ax[2].tick_params(labelleft=False, labelright=True)
ax[1].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
for j in range(3):
    ax[j].set_title(r'$a = {}$'.format(np.round(a_list[j], 3)), x=0.15, y=0.9, fontsize=18)
    # ax[j].plot(x, tau_list[j], c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
    ax[j].plot(x, tau_list[j], c='b', lw=1.5, label=r'$[\tau]_{\Lambda}$')
    ax[j].plot(x, fde[j], c='cyan', lw=1.5, label=r'FDE')



    colors = iter(['k', 'r', 'seagreen', 'cyan'])

    for l in range(len(pars_list)):
        ax[j].plot(x, fits[l][j], c=next(colors), lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{{\Lambda}}\right>, N_{{\mathrm{{p}}}} = {}$'.format(pars_list[l]))
        # aic_str = r'$AIC = {}$'.format(np.round(aics[l][j], 5))
        # ax[j].text(0.35, 0.05, chi_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[j].transAxes)
        print('a = {}, npars = {}, AIC = {}'.format(np.round(a_list[j], 3), pars_list[l], np.round(aics[l][j], 5)))

    ax[j].minorticks_on()
    ax[j].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[j].yaxis.set_ticks_position('both')

plt.legend(fontsize=18, bbox_to_anchor=(1, 1.325))
fig.align_labels()
plt.subplots_adjust(wspace=0.17)
plt.show()
# plt.savefig('../plots/test/new_paper_plots/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close()


#     # print('number of parameters used = {}'.format(npars))
#     # print('AIC = {}'.format(aic))
#     # print('BIC = {}\n'.format(bic))
#     # ax.plot(x, fit, next(colors), lw=2, label='fit; N = {}, AIC = {}'.format(npars, np.round(aic, 5)))
#
# # print('a = {}'.format(np.round(a, 3)))
# ax.plot(x, tau_l, c='b', ls='dashed', label=r'$\tau_{l}$')
# ax.set_title('a = {}'.format(np.round(a, 3)))
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.yaxis.set_ticks_position('both')
# ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
# ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
# plt.legend()
#
# # plt.savefig('../plots/test/new_paper_plots/aic_pre_sc.png', bbox_inches='tight', dpi=150)
# # # plt.savefig('../plots/test/new_paper_plots/aic_post_sc.png', bbox_inches='tight', dpi=150)
# # plt.savefig('../plots/test/new_paper_plots/sharp_aic_test.png', bbox_inches='tight', dpi=150)
# # plt.close()
# plt.show()
