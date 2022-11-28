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
    P_lin = sol[4]
    ctot2 = sol[11]
    return a, x, tau_l, dc_l, dv_l, P_lin, ctot2

def fitting_function(X, a0, a1, a2):
    x1, x2 = X
    return a0 + a1*x1 + a2*x2

plots_folder = 'test'
colors = ['violet', 'b', 'r', 'k']
labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
linestyles = ['solid', 'dashdot', 'dashed', 'dotted']

rho_0 = 27.755
H0 = 100

Nfiles = 51
P_tt = np.zeros(Nfiles)
P_dd = np.zeros(Nfiles)
P_jj = np.zeros(Nfiles)
a_list = np.zeros(Nfiles)


for file_num in range(Nfiles):
    sols = []
    for run in range(1, 5):
        path = 'cosmo_sim_1d/nbody_phase_run{}/'.format(run)
        # sols.append(tau_ext(file_num, Lambda, path, A, mode, kind))
        sol = tau_ext(file_num, Lambda, path, A, mode, kind)
        sols.append(sol[2])
        x = sol[1]
        a = sol[0]
        rho_b = rho_0 / a**3
        if run == 1:
            dc_l = sol[3]
            dv_l = sol[4]
            P_lin = sol[5]
            ctot2 = sol[6]
            nbody_filename = 'output_{0:04d}.txt'.format(file_num)
            nbody_file = np.genfromtxt(path + nbody_filename)
            Psi = nbody_file[:,1]
            Psi *= (2*np.pi)
            std = np.std([Psi], ddof=1)
            k_NL = 1/np.mean(np.abs(Psi))
            print('k_NL = ', k_NL)

    print('a = ', a)
    a_list[file_num] = a
    H = H0 * (a**(-3/2))
    tau_l = (sols[0] + sols[1] + sols[2] + sols[3]) / 4
    tau_l_0 = sols[0]

    tau_j = tau_l_0 - tau_l
    tau_k = (np.fft.fft(tau_l) / tau_l.size)[mode]
    dk_l = (np.fft.fft(dc_l) / dc_l.size)[mode]
    # tau_j_k = (np.fft.fft(tau_j) / tau_j.size)[mode]
    # P_tt[file_num] = tau_k * np.conj(tau_k) / ((mode/k_NL)**3)
    # P_jj[file_num] = tau_j_k * np.conj(tau_j_k)
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))

    P_dd[file_num] = dk_l * np.conj(dk_l)
    tt = tau_k * np.conj(tau_k)
    P_jj[file_num] = (mode**2 / (H**2 * rho_b))**2 * tt
    # P_tt[file_num] = (1 - (np.sqrt(ctot2) * mode / H)**2) * P_lin[mode]
    P_tt[file_num] = (ctot2) * (mode / H)**2 * P_lin[mode]

# k /= (2*np.pi)
# plt.scatter(k, P_tt, c='k', s=25)
# plt.scatter(k, P_dd, c='b', s=15)
# plt.show()

a_sc = 1.81
fig, ax = plt.subplots()
ax.set_title(r'$k = {}, \Lambda = {} \;[h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode, int(Lambda/(2*np.pi))))
savename = 'stoch'

ax.set_xlabel(r'$a$', fontsize=16)

# ax.set_ylabel(ylabel, fontsize=16)

# ax.plot(a_list, P_dd, c='b', lw=2.5, label=r'$P_{\mathrm{nb}}$')
ax.plot(a_list, P_tt, c='r', ls='dashed', lw=2.5, label=r'$P_{\mathrm{TT}}$')
ax.plot(a_list, P_jj, c='k', ls='dashed', lw=2.5, label=r'$P_{\mathrm{JJ}}$')


# ax.plot(a_list, P_tt, c='k', lw=2.5, label=r'$\left<\tau\right>^{2}\left(\frac{k}{k_{\mathrm{NL}}}\right)^{3}$')
# ax.plot(a_list, P_jj, c='b', ls='dashed', lw=2.5, label=r'$\left<\Delta\tau\right>^{2}$')

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.yaxis.set_ticks_position('both')
ax.axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')
ax.legend(fontsize=11)
plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
plt.close()
