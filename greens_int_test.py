#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

from EFT_nbody_solver import *
from functions import plotter, SPT_real_sm, SPT_real_tr, smoothing
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#run5: k2 = 7; Nfiles = 101
#run2: k2 = 11; Nfiles = 81
#run6: only k1; Nfiles = 51

# path = 'cosmo_sim_1d/nbody_new_run2/'
plots_folder = 'test/'

Nfiles = 51
mode = 1
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

H0 = 100
rho_0 = 27.755
A = [-0.05, 1, -0.5, 11]

#define lists to store the data
a_list = np.zeros(Nfiles)

An = np.zeros(Nfiles)
Bn = np.zeros(Nfiles)
Pn = np.zeros(Nfiles)
Qn = np.zeros(Nfiles)

I1 = np.zeros(Nfiles, dtype='complex')
I2 = np.zeros(Nfiles, dtype='complex')
J1 = np.zeros(Nfiles, dtype='complex')
J2 = np.zeros(Nfiles, dtype='complex')

#the densitites
d1k = np.zeros(Nfiles)
d2k = np.zeros(Nfiles)
P_nb = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l_tr = np.zeros(Nfiles)

#initial scalefactor
path = 'cosmo_sim_1d/phase_full_run1/'
# path = 'cosmo_sim_1d/nbody_phase_run1/'

a0 = EFT_solve(0, Lambda, path, A, kind)[0]

for file_num in range(Nfiles):
    taus = []
    for run in range(1, 5):
        path = 'cosmo_sim_1d/phase_full_run{}/'.format(run)
        # path = 'cosmo_sim_1d/nbody_phase_run{}/'.format(run)

        sol = param_calc(file_num, Lambda, path, A, mode, kind)
        if run == 1:
            a = sol[0]
            x = sol[1]
            k = sol[2]
            P_nb_a = sol[3]
            P_lin_a = sol[4]
            P_1l_a_tr = sol[7]
            tau_l = sol[9]
            ctot2 = sol[11]
            d1k_a = sol[17]
            d2k_a = sol[18]
            taus.append(tau_l)

            a_list[file_num] = a

        else:
            taus.append(sol[9])


    tau_l_mean = sum(np.array(taus), 0) / 4
    rho_b = rho_0 / a**3
    del_tau = tau_l - tau_l_mean

    js = spectral_calc(del_tau, 1.0, o=2, d=0) / rho_b
    js_mode = (np.fft.fft(js) / js.size)[mode]


    Pn[file_num] = ctot2 * (a**(5/2))
    Qn[file_num] = ctot2

    J1[file_num] = js_mode * (a**(3/2))
    J2[file_num] = js_mode / a

    if file_num > 0:

        An[file_num] = np.trapz(Pn[:file_num], a_list[:file_num])
        Bn[file_num] = np.trapz(Qn[:file_num], a_list[:file_num])

        I1[file_num] = np.trapz(J1[:file_num], a_list[:file_num])
        I2[file_num] = np.trapz(J2[:file_num], a_list[:file_num])

    d1k[file_num] = d1k_a[mode]
    d2k[file_num] = d2k_a[mode]
    P_nb[file_num] = P_nb_a[mode]
    P_lin[file_num] = P_lin_a[mode]
    P_1l_tr[file_num] = P_1l_a_tr[mode]
    print('a = ', a)


alpha_c = (2 / (5 * H0**2)) * ((An / a_list**(5/2)) - Bn)

del_J = -(2 / (5 * H0**2)) * ((I1 / a_list**(3/2)) - (I2 * a_list))

d_JJ = np.abs(del_J * np.conj(del_J))
d_J1 = np.abs(del_J * np.conj(d1k) + d1k * np.conj(del_J)) / 2
d_J2 = np.abs(del_J * np.conj(d2k) + d2k * np.conj(del_J)) / 2

print('JJ: ', d_JJ)
print('J1: ', d_J1)
print('J2: ', d_J2)

P_stoch = d_JJ + d_J1 + d_J2

P_eft_tr = P_1l_tr + ((2 * alpha_c) * (k[mode]**2) * P_lin)
print('EFT: ', P_eft_tr)

P_eft_stoch = P_eft_tr + P_stoch

a_sc = 1 / np.max(initial_density(x, A, 1))

#for plotting the spectra
xaxis = a_list
yaxes = [P_nb / a_list**2, P_1l_tr / a_list**2, P_eft_tr / a_list**2, P_eft_stoch / a_list**2]
for spec in yaxes:
    spec /= 1e-4
colours = ['b', 'brown', 'k', 'cyan']
labels = [r'$N$-body', 'SPT-4', r'EFT: from fit to $[\tau]_{l}$',  r'EFT + stochastic']
linestyles = ['solid', 'dashed', 'dotted', 'dotted']
savename = 'eft_spectra_k{}_L{}_{}'.format(mode, int(Lambda/(2*np.pi)), kind)
xlabel = r'$a$'
ylabel = r'$a^{-2}P(k=1, a) \times 10^{4}$'
title = r'$k = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(mode, int(Lambda/(2*np.pi)), kind_txt)

plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title)


# fig, ax = plt.subplots()
# ax.set_xlabel('a')
# ax.plot(a_list[:], f[:], c='b', lw=2)
# plt.savefig('../plots/test/greens_test.png', dpi=150)
# # plt.show()
