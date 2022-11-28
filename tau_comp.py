#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import os

from EFT_nbody_solver import *
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# file_num = 27
Lambda = 3 * (2 * np.pi)
mode = 1
kind = 'sharp'
A = [-0.05, 1, -0.5, 11]

def tau_ext(file_num, Lambda, path, A, mode, kind):
    sol = param_calc(file_num, Lambda, path, A, mode, kind)
    a = sol[0]
    x = sol[1]
    tau_l = sol[9]
    dc_l = sol[-2]
    dv_l = sol[-1]
    return a, x, tau_l, dc_l, dv_l

def fitting_function(X, a0, a1, a2):
    x1, x2 = X
    return a0 + a1*x1 + a2*x2

plots_folder = 'phase_full_run1/phases'
colors = ['violet', 'b', 'r', 'k']
labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
linestyles = ['solid', 'dashdot', 'dashed', 'dotted']

for file_num in range(12, 51):
    sols = []
    for run in range(1, 5):
        path = 'cosmo_sim_1d/phase_full_run{}/'.format(run)
        sols.append(tau_ext(file_num, Lambda, path, A, mode, kind))
        # sol = tau_ext(file_num, Lambda, path, A, mode, kind)
        # sols.append(sol[2])
        # x = sol[1]
        a = sols[0][0]
        # if run == 1:
        #     dc_l = sol[3]
        #     dv_l = sol[4]
    
    #ensemble average?
    # tau_l = (sols[0] + sols[1] + sols[2] + sols[3]) / 4
    # tau_l_0 = sols[0]
    #
    # #fit to one realisation
    # guesses = 1, 1, 1
    # FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l_0, guesses, sigma=1e-15*np.ones(x.size), method='lm')
    # C0, C1, C2 = FF[0]
    # cov = FF[1]
    # err0, err1, err2 = np.sqrt(np.diag(cov))
    # fit_0 = fitting_function((dc_l, dv_l), C0, C1, C2)
    # C = [C0, C1, C2]
    #
    # #fit to the mean
    # guesses = 1, 1, 1
    # FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=1e-15*np.ones(x.size), method='lm')
    # C0, C1, C2 = FF[0]
    # cov = FF[1]
    # err0, err1, err2 = np.sqrt(np.diag(cov))
    # fit = fitting_function((dc_l, dv_l), C0, C1, C2)
    # C = [C0, C1, C2]
    #


    print('a = ', a)
    fig, ax = plt.subplots()
    ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=12)
    for i in range(4):
        ax.plot(sols[i][1], sols[i][2], c=colors[i], lw=2, ls=linestyles[i], label=labels[i])

    # ax.plot(x, tau_l, c='b', lw=2, label=r'$\left<[\tau]_{\Lambda}\right>$')
    # ax.plot(x, tau_l_0, c='red', ls='dashed', lw=2, label=r'$[\tau]_{\Lambda}$')

    # ax.plot(x, fit_0, c='k', ls='dashdot', lw=2, label=r'fit to $[\tau]_{\Lambda}$')
    # ax.plot(x, fit, c='red', ls='dashed', lw=2, label=r'fit to $\left<[\tau]_{\Lambda}\right>$')


    ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
    ax.ticklabel_format(scilimits=(-2, 3))
    # ax.grid(lw=0.2, ls='dashed', color='grey')
    ax.legend(fontsize=12, bbox_to_anchor=(1,1))#, title='Phase shift')
    ax.yaxis.set_ticks_position('both')
    plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, file_num), bbox_inches='tight', dpi=150)
    plt.close()
    # # plt.savefig('../plots/{}/tau_{}.pdf'.format(plots_folder, j), bbox_inches='tight', dpi=300)

    # plt.show()
