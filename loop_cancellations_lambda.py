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
kind = 'sharp'
kind_txt = 'sharp cutoff'

kind = 'gaussian'
kind_txt = 'Gaussian smoothing'

n_runs = 8
n_use = n_runs-1
j = 24
fm = '' #fitting method
nbins_x, nbins_y, npars = 20, 20, 6

def P13_finder(path, j, Lambdas, kind, mode):
    print('P13 finder')
    Nx = 2048
    L = 1.0
    dx = L/Nx
    x = np.arange(0, L, dx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    a_list, P13_list = [], []
    a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
    dc_in = dc_in_finder(path, x, interp=True)[0]
    Nx = dc_in.size
    for Lambda in Lambdas:
        print('Lambda = ', Lambda)
        Lambda *= (2 * np.pi)
        dc_in = smoothing(dc_in, k, Lambda, kind)
        F = dn(3, L, dc_in)
        d1k = (np.fft.fft(F[0]) / Nx)
        d2k = (np.fft.fft(F[1]) / Nx)
        d3k = (np.fft.fft(F[2]) / Nx)
        P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
        P13 += ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k))) * (a**3)

        P13_list.append(np.real(P13)[mode])

    return np.array(P13_list)


def ctot_finder(Lambdas, path, j, A, mode, kind, n_runs, n_use):
    ctot2_list, ctot2_2_list, ctot2_3_list = [], [], []
    for Lambda in Lambdas:
        print('Lambda = ', Lambda)
        Lambda *= (2 * np.pi)
        sol = param_calc_ens(j, Lambda, path, A, mode, kind, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
        ctot2_list.append(sol[2])
        ctot2_2_list.append(sol[3])
        ctot2_3_list.append(sol[4])
    a = sol[0]
    return a, np.array(ctot2_list), np.array(ctot2_2_list), np.array(ctot2_3_list)

def alpha_c_lambda(Lambdas, path, j, A, mode, kind, n_runs, n_use, fm='', nbins_x=10, nbins_y=10, npars=3):
    ac, ac2, ac3, err = [], [], [], []
    for Lambda in Lambdas:
        print('Lambda = ', Lambda)
        Lambda *= (2 * np.pi)
        sol = alpha_c_finder(j, Lambda, path, A, mode, kind, fm=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
        ac.append(sol[2][-1])
        ac2.append(sol[3][-1])
        ac3.append(sol[4][-1])
        err.append(sol[-1][-1])
    a = sol[0][-1]
    return a, ac, ac2, ac3, err

Lambdas = np.arange(2, 7)

# P13 = P13_finder(path, j, Lambdas, kind, mode)
# df = pandas.DataFrame(data=[P13])
# pickle.dump(df, open("./{}/P13_lambda_{}_{}.p".format(path, kind, j), "wb"))

P13 = np.array(pickle.load(open("./{}/P13_lambda_{}_{}.p".format(path, kind, j), "rb")))[0]
# a, ctot2, ctot2_2, ctot2_3 = ctot_finder(Lambdas, path, j, A, mode, kind, n_runs, n_use)
a, ac, ac2, ac3, err = alpha_c_lambda(Lambdas, path, j, A, mode, kind, n_runs, n_use, fm=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)


# sigma_ctot2_2 = np.sqrt(sum((ctot2_2 - np.mean(ctot2_2))**2))
# sigma_P13 = np.sqrt(sum((P13 - np.mean(P13))**2))
#
# print(sigma_P13, sigma_ctot2_2)
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots(figsize=(10, 6))
labels=[r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$', r'$P_{13}$']

ax.set_title(r'$a = {}$ ({})'.format(a, kind_txt), fontsize=20)
ax.set_xlabel(r'$\Lambda \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=22)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=18)
ax.set_ylabel(r'$\alpha_{c} / P_{13}$', fontsize=22)
# ax.set_ylim(0.0001, 0.0004)
# ax.plot(Lambdas, ctot2, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
# ax.plot(Lambdas, ctot2_2, c='cyan', lw=1.5, marker='*', label=labels[1])
# ax.plot(Lambdas, ctot2_3, c='orange', lw=1.5, marker='v', label=labels[2])
# ax.plot(Lambdas, P13 / normc, c='blue', lw=1.5, marker='x', label=labels[3])

# ratio = ctot2 / P13
# ratio2 = ctot2_2 / P13
# ratio3 = ctot2_3 / P13

ratio = ac / P13
ratio2 = ac2 / P13
ratio3 = ac3 / P13

err /= P13
# print(ratio, err)
ax.plot(Lambdas, ratio, c='k', lw=1.5, zorder=4, marker='o', label=labels[0])
ax.plot(Lambdas, ratio2, c='brown', lw=1.5, marker='*', label=labels[1])
ax.plot(Lambdas, ratio3, c='orange', lw=1.5, marker='v', label=labels[2])
ax.fill_between(Lambdas, ratio-err, ratio+err, color='darkslategray', alpha=0.55, rasterized=True)


ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=14)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/loops_with_Lambda_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
plt.close()
