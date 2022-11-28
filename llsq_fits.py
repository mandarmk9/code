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

def fitting_function(X, a0, a1, a2):
    x1, x2 = X
    return a0 + a1*x1 + a2*x2

guesses = 1, 1, 1
FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
C0, C1, C2 = FF[0]
cov = FF[1]
err0, err1, err2 = np.sqrt(np.diag(cov))

fit = fitting_function((dc_l, dv_l), C0, C1, C2)
fit_sp = fit[0::n_ev]
C = [C0, C1, C2]

resid = fit_sp - tau_l_sp
chisq = sum((resid / yerr_sp)**2)
red_chi = chisq / (n_use - 3)

cov = np.array(cov)
corr = np.zeros(cov.shape)

for i in range(3):
    for j in range(3):
        corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])

terr = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)

def weighted_ls_fit(d, x, err, Np):
    """Fits a multivariate linear model to a data vector.
    d: the data vector
    x: a tuple of size Np containing the fields used for the fit
    err: error on the fields in x (assumed to be the variance)
    Np: number of parameters to be used for the fit

    Returns the parameters of the fit, and the covariance and correlation matrices.
    """
    assert len(x) == Np, "x must be a tuple of size Np"

    weights = 1/err**2 #the weights are defined as the inverse of the variance
    W = np.diag([sum(weights) for j in range(Np)]) #the weight matrix
    y = np.array([sum(weights * d * field) for field in x])
    X = np.empty(shape=(Np, Np))
    for j in range(Np):
        X[j] = np.array([sum(weights*field*x[j]) for field in x])

    #calculate the covariance and correlation matrices
    cov = (np.linalg.inv((X.T).dot(W.dot(X))))

    corr = np.zeros(cov.shape)
    for i in range(Np):
        corr[i,:] = [cov[i,j] / np.sqrt(cov[i,i]*cov[j,j]) for j in range(Np)]

    params = (np.linalg.inv(X)).dot(y)

    return params, cov, corr


def fit_bootstrap(d, x, err):
    from scipy.optimize import leastsq
    X1, X2 = x

    # Fit first time
    pfit, perr = leastsq(fitting_function, p0, args=(d, X1, X2), full_output=0)

    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_err = np.std(residuals)

    # N_sets random data sets are generated and fitted
    ps = []
    for i in range(N_sets):
        randomDelta = np.random.normal(0., sigma_err, len(ydata))
        randomdataY = ydata + randomDelta

        randomfit, randomcov = leastsq(errfunc, p0, args=(xdata, randomdataY), full_output=0)

        ps.append(randomfit)

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0)

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit

    return pfit_bootstrap, perr_bootstrap


#here we define the weighted matrices
X = (np.ones(yerr_sp.size), dc_l_sp, dv_l_sp)
(C0__, C1__, C2__), cov__, corr__ = weighted_ls_fit(tau_l_sp, X, yerr_sp, len(X))
C__ = (C0__, C1__, C2__)

err1__ = cov__[1,1]
err2__ = cov__[2,2]
terr__ = (err1__**2 + err2__**2 + corr__[1,2]*err1__*err2__ + corr__[2,1]*err2__*err1__)**(0.5)
fit__ = fitting_function((dc_l, dv_l), C0__, C1__, C2__)

plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots()
ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)

line0 = ax.plot(x, tau_l, c='b', lw=1.5)
line1 = ax.plot(x, fit, c='r', ls='dashdot', lw=1.5)
# ax.fill_between(x, fit-terr, fit+terr, color='darkslategray', alpha=0.35, rasterized=True)
fill1 = ax.fill(np.NaN, np.NaN, color='darkslategray', alpha=0.35, rasterized=True)

line2 = ax.plot(x, fit__, c='k', ls='dashed', lw=1.5)
ax.fill_between(x, fit__-terr__, fit__+terr__, color='midnightblue', alpha=0.35, rasterized=True)
fill2 = ax.fill(np.NaN, np.NaN, color='midnightblue', alpha=0.35, rasterized=True)

labels = ['measured', r'fit using \texttt{curve\_fit}', r'fit using WLS']
handles = [(line0[0],), (fill1[0], line1[0],), (fill2[0], line2[0],),]

ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.legend(handles, labels)
ax.yaxis.set_ticks_position('both')

text = r'$N_{{\mathrm{{points}}}} = {}$'.format(n_use)
ax.text(0.45, 0.05, text, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax.transAxes)

plt.show()
