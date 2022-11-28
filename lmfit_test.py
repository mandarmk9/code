#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from lmfit import Minimizer, Parameters, report_fit
from scipy.optimize import curve_fit
from functions import read_sim_data


path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
n_runs = 24

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# n_runs = 8

A = []
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
plots_folder = 'test/'
mode = 1
rho_0 = 27.755
H0 = 100
n_use = n_runs-1

j = 0
a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
taus = []
taus.append(tau_l_0)
rho_b = rho_0 / a**3

for run in range(2, n_runs+1):
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
x_sp = x[0::n_ev]

def fitting_function(X, a0, a1, a2):
    x1, x2 = X
    return a0 + a1*x1 + a2*x2

guesses = 1, 1, 1
FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, (1,1,1), sigma=yerr_sp, method='lm', absolute_sigma=True)
C0, C1, C2 = FF[0]
cov = FF[1]
err0, err1, err2 = np.sqrt(np.diag(cov))
fit = fitting_function((dc_l_sp, dv_l_sp), C0, C1, C2)
C = [C0, C1, C2]
tau_err = np.sqrt(err0**2 + err1**2 + err2**2)

resid = fit - tau_l_sp
chisq = sum((resid / yerr_sp)**2)
red_chi = chisq / (n_use - 3)

# create data to be fitted
data = tau_l_sp

def model(x0, x1, x2, params):
    a0 = params['C0']
    a1 = params['C1']
    a2 = params['C2']

    return a0*x0 + a1*x1 + a2*x2

def sq_residuals(params, d, X, err):
    """Returns the sum of square residuals from the data, input, and errors."""
    weights = 1/err**2
    x0, x1, x2 = X
    # S = sum([weights[i] * (d[i] - model(x0[i], x1[i], x2[i], params))**2 for i in range(x1.size)])
    resid = [weights[i] * (d[i] - model(x0[i], x1[i], x2[i], params)) for i in range(x1.size)]
    return resid

# create a set of Parameters
params = Parameters()
params.add('C0', value=1)
params.add('C1', value=1)
params.add('C2', value=1)

# do fit, here with the default leastsq algorithm
d = tau_l_sp
X = (np.ones(dc_l_sp.size), dc_l_sp, dv_l_sp)
guesses = (1, 1, 1)
minner = Minimizer(sq_residuals, params, fcn_args=(d, X, yerr_sp))#, scale_covar=False)
result = minner.minimize()

# # cov = result.covar
# # params = result.params
# # red_chi = result.redchi
# # print(result.params['C0'].value)
# # calculate final result
# final = data + result.residual
# print(result.residual)
# # write error report
# report_fit(result)
# print('\nfrom curve_fit:')
# print('params: ', C)
# print('errors: {}, {}, {}'.format(err0, err1, err2))
# print('chi-squared: ', chisq)
# print('red chi-squared: ', red_chi)
#
C0_ = result.params['C0'].value
C1_ = result.params['C1'].value
C2_ = result.params['C2'].value
cov_ = result.covar
fit_lm = fitting_function((dc_l_sp, dv_l_sp), C0_, C1_, C2_)
err0_ = cov[0,0]
err1_ = cov[1,1]
err2_ = cov[2,2]
tau_err_ = np.sqrt(err0_**2 + err1_**2 + err2_**2)

print('curve_fit: ', cov)
print('\nlmfit: ', cov_)

# plt.rcParams.update({"text.usetex": True})
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
# ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
#
# line0 = ax.plot(x, tau_l, c='b', lw=1.5)
# line0_ = ax.plot(x_sp, tau_l_sp, c='k', lw=1.5, ls='dashdot')
#
# line1 = ax.plot(x_sp, fit_lm, c='r', ls='dashed', lw=1.5, marker='+')
# ax.fill_between(x, tau_l-tau_err, tau_l+tau_err, color='darkslategray', alpha=0.35, rasterized=True)
# fill1 = ax.fill(np.NaN, np.NaN, color='darkslategray', alpha=0.35, rasterized=True)
#
# line2 = ax.plot(x_sp, fit, c='orange', ls='dotted', lw=1.5, marker='o')
# ax.fill_between(x, tau_l-tau_err_, tau_l+tau_err_, color='seagreen', alpha=0.5, rasterized=True)
# fill2 = ax.fill(np.NaN, np.NaN, color='seagreen', alpha=0.35, rasterized=True)
#
#
# labels = ['original', 'sampled', r'fit using \texttt{curve\_fit}', r'fit using \texttt{lmfit}']
# handles = [(line0[0],), (line0_[0],), (fill1[0], line1[0],), (fill2[0], line2[0],),]
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


# # try to plot results
# try:
#     plt.plot(x, tau_l, label='original data')
#     plt.plot(x_sp, tau_l_sp, ls='dashdot', label='sampled data')
#     plt.plot(x_sp, fit_lm, ls='dashed', marker='+', label='lmfit')
#     plt.plot(x_sp, fit, ls='dotted', marker='o', label='curve_fit')
#     plt.fill_between(x, tau_l-tau_err, tau_l+tau_err, color='darkslategray', alpha=0.35, rasterized=True)
#     plt.fill_between(x, tau_l-tau_err, tau_l+tau_err, color='darkslategray', alpha=0.35, rasterized=True)
#
#     plt.legend()
#     plt.show()
# except ImportError:
#     pass
# # <end of examples/doc_parameters_basic.py>
