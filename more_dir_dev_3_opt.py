#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from functions import read_sim_data, param_calc_ens, deriv_param_calc
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'


# j = 18
for j in range(27, 28):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
    print('\na = {}'.format(np.round(a, 3)))
    dv_l *= -np.sqrt(a) / 100
    tau_l -= np.mean(tau_l)

    guesses = 1, 1, 1

    def fitting_function(X, a0, a1, a2):
        x1 = X
        return a0 + a1*x1 + a2*(x1**2)
    C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
    C_, err_, sub = deriv_param_calc(dc_l, dv_l, tau_l)

    # import pprint
    # pprint.pprint(sub)

    fit = C[0] + C[1]*dc_l + C[2]*dc_l**2
    print('C0_deriv = ', C_[0], 'C0_fit = ', C[0])
    print('C1_deriv = ', C_[1], 'C1_fit = ', C[1])
    print('C2_deriv = ', C_[2], 'C2_fit = ', C[2])

    fit_fde = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
    resid_fde = sum((tau_l - fit_fde)**2)
    resid_fit = sum((tau_l - fit)**2)

    print("residuals: ", resid_fde, resid_fit)
    # rho_b = 27.755 / a**3
    # print('ctot2 = ', C_[1]/rho_b)

    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)

    ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
    ax.set_title(r'$a ={}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a,3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16, y=1.01)

    plt.plot(x, tau_l, c='b', label=r'measured')
    plt.plot(x, fit, c='r', ls='dashed', label='fit')
    plt.plot(x, fit_fde, c='k', ls='dashed', label='FDE')
    plt.scatter(x[sub], tau_l[sub], c='seagreen', s=20)

    plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
    plt.show()
    # plt.savefig('../plots/test/new_paper_plots/optimised_fde_taus/{}_tau_{}.png'.format(kind, j), bbox_inches='tight', dpi=150)
    # plt.close()
