#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functions import read_sim_data


path = 'cosmo_sim_1d/new_sim_k_1_11/run1/'

A = []
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
plots_folder = 'test/'
mode = 1
rho_0 = 27.755
H0 = 100
n_runs = 24
n_use = 23

j = 20
a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
taus = []
taus.append(tau_l_0)
rho_b = rho_0 / a**3

for run in range(1, n_runs+1):
    if run <= 10:
        path = path[:-2] + '{}/'.format(run)
    else:
        path = path[:-3] + '{}/'.format(run)

    taus.append(read_sim_data(path, Lambda, kind, j)[1])

Nt = len(taus)
tau_l = sum(np.array(taus)) / Nt

diff = np.array([(taus[i] - taus[0])**2 for i in range(1, Nt)])
yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))

n_ev = int(x.size / n_use)
dc_l_sp = dc_l[0::n_ev]
dv_l_sp = dv_l[0::n_ev]
tau_l_sp = tau_l[0::n_ev]
yerr_sp = yerr[0::n_ev]

def model(x1, x2, a0, a1, a2):
    return a0 + a1*x1 + a2*x2

def sq_residuals(d, X, p, err):
    """Returns the sum of square residuals from the data, input, and errors."""
    weights = 1/err**2
    x1, x2 = X
    a0, a1, a2 = p
    S = sum([weights[i] * (d[i] - model(x1[i], x2[j], a0, a1, a2))**2 for i in range(x1.size)])
    return S

def MCMC(d, X, p0, err, n=100, step=0.5):
    a0_, a1_, a2_ = p0
    res_ = sq_residuals(d, X, p0, err)
    chain = np.empty((n+1, 3))
    chain[0,:] = [a0_, a1_, a2_]
    acc_rate = 1

    for j in range(n):
        a0 = np.random.normal(a0_, step)
        a1 = np.random.normal(a1_, step)
        a2 = np.random.normal(a2_, step)
        p = (a0, a1, a2)
        res = sq_residuals(d, X, p, err)

        res_ratio = res / res_ #in log space, the ratio becomes the difference
        rand = np.random.uniform()

        if res_ratio >= rand:
            a0_, a1_, a2_ = a0, a1, a2
            res_ = res
            acc_rate +=1

        chain[j+1,:] = [a0_, a1_, a2_]

    print("Acceptance rate: " + str(100*acc_rate/n) + "%")

    return chain

guesses = 1, 1, 1
def fitting_function(X, a0, a1, a2):
    x1, x2 = X
    return a0 + a1*x1 + a2*x2
FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
C0, C1, C2 = FF[0]
cov = FF[1]
err0, err1, err2 = np.sqrt(np.diag(cov))
C = [C0, C1, C2]
# guesses = (C0, C1, C2)
guesses = (1, 1, 1)
chain = MCMC(tau_l, (dc_l, dv_l), guesses, yerr, n=1000, step=0.1)

plt.plot(chain[:,2])
plt.axhline(C2)
plt.show()

# #here we define the weighted matrices
# X = (np.ones(yerr_sp.size), dc_l_sp, dv_l_sp)
# (C0__, C1__, C2__), cov__, corr__ = weighted_ls_fit(tau_l_sp, X, yerr_sp, len(X))
# C__ = (C0__, C1__, C2__)
#
# err1__ = cov__[1,1]
# err2__ = cov__[2,2]
# terr__ = (err1__**2 + err2__**2 + corr__[1,2]*err1__*err2__ + corr__[2,1]*err2__*err1__)**(0.5)
# fit__ = fitting_function((dc_l, dv_l), C0__, C1__, C2__)
#
# plt.rcParams.update({"text.usetex": True})
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
# ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
#
# line0 = ax.plot(x, tau_l, c='b', lw=1.5)
# line1 = ax.plot(x, fit, c='r', ls='dashdot', lw=1.5)
# # ax.fill_between(x, fit-terr, fit+terr, color='darkslategray', alpha=0.35, rasterized=True)
# fill1 = ax.fill(np.NaN, np.NaN, color='darkslategray', alpha=0.35, rasterized=True)
#
# line2 = ax.plot(x, fit__, c='k', ls='dashed', lw=1.5)
# ax.fill_between(x, fit__-terr__, fit__+terr__, color='midnightblue', alpha=0.35, rasterized=True)
# fill2 = ax.fill(np.NaN, np.NaN, color='midnightblue', alpha=0.35, rasterized=True)
#
# labels = ['measured', r'fit using \texttt{curve\_fit}', r'fit using WLS']
# handles = [(line0[0],), (fill1[0], line1[0],), (fill2[0], line2[0],),]
#
# ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.legend(handles, labels)
# ax.yaxis.set_ticks_position('both')
#
# text = r'$N_{{\mathrm{{points}}}} = {}$'.format(n_use)
# ax.text(0.45, 0.05, text, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax.transAxes)
#
# plt.show()
