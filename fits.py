#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from functions import read_sim_data, AIC, sub_find, binning
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, npars=3, data_cov=False):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, '')

    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    taus = []
    taus.append(tau_l_0)
    for run in range(2, n_runs+1):
        if run > 10:
            ind = -3
        else:
            ind = -2
        path = path[:ind] + '{}/'.format(run)
        taus.append(read_sim_data(path, Lambda, kind, j, '')[1])

    taus = np.array(taus)
    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt

    diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))


    n_ev = x.size // n_use
    dc_l_sp = dc_l[0::n_ev]
    dv_l_sp = dv_l[0::n_ev]
    dc2_l_sp = (dc_l**2)[0::n_ev]
    dv2_l_sp = (dv_l**2)[0::n_ev]
    dc_dv_l_sp = (dc_l*dv_l)[0::n_ev]

    tau_l_sp = tau_l[0::n_ev]
    yerr_sp = yerr[0::n_ev]
    x_sp = x[0::n_ev]
    taus_sp = np.array([taus[k][0::n_ev] for k in range(n_runs)])

    if npars == 3:
        def fitting_function(X, a0, a1, a2):
            x1, x2 = X
            return a0 + a1*x1 + a2*x2
        guesses = 1, 1, 1
        C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
        C0, C1, C2 = C
        fit = fitting_function((dc_l, dv_l), C0, C1, C2)

    elif npars == 6:
        def fitting_function(X, a0, a1, a2, a3, a4, a5):
            x1, x2, x3, x4, x5 = X
            return a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
        guesses = 1, 1, 1, 1, 1, 1
        C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp, dc2_l_sp, dv2_l_sp, dc_dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
        C0, C1, C2, C3, C4, C5 = C
        fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), C0, C1, C2, C3, C4, C5)


    fit_sp = fit[0::n_ev]

    resid = fit_sp - tau_l_sp

    cs2 = np.real(C1 / rho_b)
    cv2 = -np.real(C2 * H0 / (rho_b * np.sqrt(a)))
    ctot2 = (cs2 + cv2)

    f1 = (1 / rho_b)
    f2 = (-H0 / (rho_b * np.sqrt(a)))

    cov[0,1] *= f1
    cov[1,0] *= f1
    cov[0,2] *= f2
    cov[2,0] *= f2
    cov[1,1] *= f1**2
    cov[2,2] *= f2**2
    cov[2,1] *= f2*f1
    cov[1,2] *= f1*f2

    corr = np.zeros(cov.shape)
    err0, err1, err2 = np.sqrt(np.diag(cov[:3,:3]))
    corr[1,2] = cov[1,2] / np.sqrt(cov[1,1]*cov[2,2])
    corr[2,1] = cov[2,1] / np.sqrt(cov[1,1]*cov[2,2])

    ctot2 = (cs2 + cv2)

    try:
        terr = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)
    except:
        terr = 0

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    tau_l_k = np.fft.fft(tau_l) / x.size
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0

    ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

    T = -dv_l / (H0 / (a**(1/2)))

    def Power_fou(f1, f2):
        f1_k = np.fft.fft(f1)
        f2_k = np.fft.fft(f2)
        corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
        return corr[1]

    ctot2_3 = np.real(Power_fou(tau_l/rho_b, dc_l) / Power_fou(dc_l, T))
    return a, x, ctot2, ctot2_2, ctot2_3, tau_l, fit, terr, taus

a_list, ctot2_list, ctot2_2_list, ctot2_3_list, err1_list, err2_list, chi_list, terr = [], [], [], [], [], [], [], []

A = [-0.05, 1, -0.5, 11]
Nfiles = 51
Lambda = 3 * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

path, n_runs = 'cosmo_sim_1d/sim_k_1_11/run1/', 8
n_use = n_runs-2
npars = 6
mode = 1
data_cov = False
flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
for j in range(Nfiles):
    sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, npars, data_cov)
    a_list.append(sol[0])
    ctot2_list.append(sol[2])
    ctot2_2_list.append(sol[3])
    ctot2_3_list.append(sol[4])

    terr.append(sol[-2])
    print('a = ', sol[0])
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)

chi_list = np.array(chi_list)
terr = np.array(terr)


ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.fill_between(a_list, ctot2_list-terr, ctot2_list+terr, color='darkslategray', alpha=0.55, zorder=2)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)


ctot2_line, = ax.plot(a_list, ctot2_list, c='k', lw=1.5, zorder=4, marker='o')
ctot2_2_line, = ax.plot(a_list, ctot2_2_list, c='brown', lw=1.5, marker='*', zorder=2)
ctot2_3_line, = ax.plot(a_list, ctot2_3_list, c='orange', lw=1.5, marker='v', zorder=3)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'], fontsize=14)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')

# plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# plt.savefig('../plots/test/new_paper_plots/fit_comps/curve_fit/{}_{}_ctot2_ev_{}.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# plt.savefig('../plots/test/new_paper_plots/fit_comps/binning/{}_ctot2_ev_{}.png'.format(nbins_x, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
#
# plt.close()
plt.show()
