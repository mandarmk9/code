#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import dc_in_finder, smoothing, dn, param_calc_ens, alpha_c_finder

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
A = []
mode = 1
Nfiles = 51
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

n_runs = 8
n_use = n_runs-1
fm = '' #fitting method
nbins_x, nbins_y, npars = 20, 20, 6


def P13_finder(path, Nfiles, Lambda, kind, mode):
    print('P13 finder')
    Nx = 2048
    L = 1.0
    dx = L/Nx
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    a_list, P13_list = [], []
    for j in range(Nfiles):
        a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
        dc_in, k = dc_in_finder(path, x, interp=True) #[0]
        dc_in = smoothing(dc_in, k, Lambda, kind)
        Nx = dc_in.size
        F = dn(3, L, dc_in)
        d1k = (np.fft.fft(F[0]) / Nx)
        d2k = (np.fft.fft(F[1]) / Nx)
        d3k = (np.fft.fft(F[2]) / Nx)
        P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
        P13_list.append(np.real(P13)[mode])
        a_list.append(a)
        print('a = ', a)
    return np.array(a_list), np.array(P13_list)

def ctot_finder(Lambda, path, A, mode, kind, n_runs, n_use):
    a_list, ctot2_list, ctot2_2_list, ctot2_3_list = [], [], [], []
    for j in range(Nfiles):
        # sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use)
        sol = param_calc_ens(j, Lambda, path, A, mode, kind, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)

        a_list.append(sol[0])
        ctot2_list.append(sol[2])
        ctot2_2_list.append(sol[3])
        ctot2_3_list.append(sol[4])
    return np.array(a_list), np.array(ctot2_list), np.array(ctot2_2_list), np.array(ctot2_3_list)

# a_list, P13 = P13_finder(path, Nfiles, Lambda, kind, mode)
#
# df = pandas.DataFrame(data=[P13])
# pickle.dump(df, open("./{}/P13_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "wb"))

P13 = np.array(pickle.load(open("./{}/P13_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), "rb")))[0]
# a_list, ctot2, ctot2_2, ctot2_3 = ctot_finder(Lambda, path, A, mode, kind, n_runs, n_use)

# sigma_ctot2 = np.sqrt(sum((ctot2 - np.mean(ctot2))**2))
# sigma_ctot2_2 = np.sqrt(sum((ctot2_2 - np.mean(ctot2_2))**2))
# sigma_ctot2_3 = np.sqrt(sum((ctot2_3 - np.mean(ctot2_3))**2))
#
# sigma_P13 = np.sqrt(sum((P13 - np.mean(P13))**2))
#
# print(sigma_P13, sigma_ctot2, sigma_ctot2_2, sigma_ctot2_3)


a_list, x, ac, ac2, ac3, err = alpha_c_finder(Nfiles, Lambda, path, A, mode, kind, fm=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})


fig, ax = plt.subplots(figsize=(10, 6))
labels=[r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'$P_{13}$']
ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
ax.set_xlabel(r'$a$', fontsize=22)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
ax.set_ylabel(r'$\alpha_{c} / P_{13}$', fontsize=22)


# ax.plot(a_list, ctot2, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# ax.plot(a_list, ctot2_2, c='cyan', lw=1.5, marker='*', label=labels[1])
# ax.plot(a_list, ctot2_3, c='orange', lw=1.5, marker='v', label=labels[2])
# ax.plot(a_list, P13, c='blue', lw=1.5, marker='x', label=labels[3])

ratio = ac / P13
ratio2 = ac2 / P13
ratio3 = ac3 / P13
err /= P13
# ax.set_ylim(-1, 10)
ax.plot(a_list, ratio, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
ax.plot(a_list, ratio2, c='brown', lw=1.5, marker='*', label=labels[1])
ax.plot(a_list, ratio3, c='orange', lw=1.5, marker='v', label=labels[2])
ax.fill_between(a_list, ratio-err, ratio+err, color='darkslategray', alpha=0.55, rasterized=True)
# ax.axvline(1.8181, c='g', lw=1)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
ax.yaxis.set_ticks_position('both')
ax.legend(fontsize=14)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/loops_with_time_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
plt.close()


# fig, ax = plt.subplots(figsize=(10, 6))
# labels=[r'from fit to $\langle[\tau]_{\Lambda}\rangle$', 'M&W', r'$\mathrm{B^{+12}}$', r'$P_{13}$']
#
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18)
# ax.set_xlabel(r'$a$', fontsize=18)
# # ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2} / P_{13}\;[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
#
#
# # ax.plot(a_list, ctot2, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# # ax.plot(a_list, ctot2_2, c='cyan', lw=1.5, marker='*', label=labels[1])
# # ax.plot(a_list, ctot2_3, c='orange', lw=1.5, marker='v', label=labels[2])
# # ax.plot(a_list, P13, c='blue', lw=1.5, marker='x', label=labels[3])
#
# ratio = ctot2 / P13
# ratio2 = ctot2_2 / P13
# ratio3 = ctot2_3 / P13
#
# ax.plot(a_list, ratio, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# ax.plot(a_list, ratio2, c='cyan', lw=1.5, marker='*', label=labels[1])
# ax.plot(a_list, ratio3, c='orange', lw=1.5, marker='v', label=labels[2])
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11)
# plt.show()
# # plt.savefig('../plots/sim_k_1_11/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.close()
