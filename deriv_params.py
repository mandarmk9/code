#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import *#read_sim_data, spectral_calc, AIC, BIC, binning
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)

# kind = 'sharp'
# kind_txt = 'sharp cutoff'

kind = 'gaussian'
kind_txt = 'Gaussian smoothing'


def binning(j, path, Lambda, kind, bsx, bsy, nbins, npars):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)

    def find_nearest(a, a0):
        """Element in nd array 'a' closest to the scalar value 'a0'."""
        idx = np.abs(a - a0).argmin()
        return idx, a.flat[idx]

    zero_x = 0#find_nearest(dc_l, 0)[1]
    zero_y = 0#find_nearest(dv_l, 0)[1]

    dels, thes, taus, delsq, thesq, delthe, yerr, meds = [], [], [], [], [], [], [], []
    coor_x = [(-((bsx/2) + ((nbins/2)-j+1)*bsx + zero_x), -((bsx/2) + ((nbins/2)-j)*bsx + zero_x)) for j in range(nbins)]
    coor_y = [(-((bsy/2) + ((nbins/2)-j+1)*bsy + zero_y), -((bsy/2) + ((nbins/2)-j)*bsx + zero_y)) for j in range(nbins)]
    for j in range(nbins):
        for l in range(nbins):
            try:
                dc_co = np.logical_and(dc_l>=coor_x[j][0], dc_l<=coor_x[j][1])
                dv_co = np.logical_and(dv_l>=coor_y[l][0], dv_l<=coor_y[l][1])
                idx = np.where(np.logical_and(dc_co, dv_co))[0]


                dels.append(np.mean(dc_l[idx]))
                thes.append(np.mean(dv_l[idx]))
                taus.append(np.mean(tau_l[idx]))
                delsq.append(np.mean((dc_l**2)[idx]))
                thesq.append(np.mean((dv_l**2)[idx]))
                delthe.append(np.mean((dc_l*dv_l)[idx]))
                yerr.append(np.sqrt(sum((tau_l[idx] - np.mean(tau_l[idx]))**2) / (idx.size - 1)))
                meds.append(np.mean(idx))

            except Exception as e: print(e)

    meds, taus = (list(t) for t in zip(*sorted(zip(meds, taus))))
    meds, dels = (list(t) for t in zip(*sorted(zip(meds, dels))))
    meds, thes = (list(t) for t in zip(*sorted(zip(meds, thes))))
    meds, delsq = (list(t) for t in zip(*sorted(zip(meds, delsq))))
    meds, thesq = (list(t) for t in zip(*sorted(zip(meds, thesq))))

    indx = int(np.where(np.abs(dels) == min(np.abs(dels)))[0][0])
    indy = int(np.where(np.abs(thes) == min(np.abs(thes)))[0][0])

    # print(indx, indy)
    hd = (dels[indx] - dels[indx-1]) #+ (dels[indx] - dels[indx-1])
    ddtau = (taus[indx] - taus[indx-1]) / (hd)

    # ht = (thes[indy+1] - thes[indy]) #+ (dels[indx] - dels[indx-1])
    # dttau = (taus[indy+1] - taus[indy]) / (ht)

    ht = (thes[indy] - thes[indy-1])
    dttau = (taus[indy] - taus[indy-1]) / (ht)
    C_ = (taus[indx], ddtau, dttau/2)
    # print(C_)

    if npars == 1: #constant
        def fitting_function(X, a0):
            return a0 * X[0]
        X = (np.ones(len(dels)))
        X_ = (np.ones(len(x)))
        guesses = 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0 = C
        fit_sp = C0 * X
        fit = C0 * X_

    elif npars == 3: #first-order
        def fitting_function(X, a0, a1, a2):
            return a0*X[0] + a1*X[1] + a2*X[2]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes))
        X_ = (np.ones(len(x)), dc_l, dv_l)
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit_sp = fitting_function(X, C0, C1, C2)
        fit = fitting_function(X_, C0, C1, C2)

    elif npars == 5: #derivative terms
        def fitting_function(X, a0, a1, a2, a3, a4):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(dx_del), np.array(dx_the))
        X_ = (np.ones(len(x)), dc_l, dv_l, d_dcl, d_dvl)
        guesses = 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4)
        fit = fitting_function(X_, C0, C1, C2, C3, C4)


    elif npars == 6: #second-order
        def fitting_function(X, a0, a1, a2, a3, a4, a5):
            return a0*X[0] + a1*X[1] + a2*X[2] + a3*X[3] + a4*X[4] + a5*X[5]
        X = (np.ones(len(dels)), np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe))
        X_ = (np.ones(len(x)), dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l)
        guesses = 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, X, taus, sigma=yerr, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5 = C
        fit_sp = fitting_function(X, C0, C1, C2, C3, C4, C5)
        fit = fitting_function(X_, C0, C1, C2, C3, C4, C5)

    C = (C[0], C[1], C[2])
    # print(C)

    resid = fit_sp - taus
    chisq = sum((resid / yerr)**2)
    red_chi = chisq / (len(dels) - npars)
    aic = AIC(npars, chisq, n=1)
    bic = BIC(npars, len(taus), chisq)

    return a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, C_




# bsx, bsy, nbins, npars = 0.25e-1, 0.25e-1, 40, 6
bsx, bsy, nbins, npars = 1e-3, 3e-3, 10, 3

a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, C_ = binning(8, path, Lambda, kind, bsx, bsy, nbins, npars)

print(C)
print(C_)

# fit = C[0] + C[1] * dc_l + C[2] * dv_l
# fit_ = C_[0] + C_[1] * dc_l + C_[2] * dv_l
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots()
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
# ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# ax.set_title(r'$a = {}$'.format(np.round(a, 3)), fontsize=18)
#
# plt.plot(x, tau_l, c='b', label=r'measured')
# plt.plot(x, fit, c='r', ls='dashed', label='fit')
# plt.plot(x, fit_, c='k', ls='dashed', label='using derivatives')
# plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
# plt.show()
# # plt.savefig('../plots/test/new_paper_plots/new_fits.png'.format(kind), bbox_inches='tight', dpi=150)
# # plt.close()
