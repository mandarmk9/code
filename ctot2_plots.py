#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from functions import read_sim_data, AIC, sub_find, binning, spectral_calc, deriv_param_calc
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method='curve_fit', nbins_x=10, nbins_y=10, npars=3, data_cov=False):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, '')

    if fitting_method == 'curve_fit':
        rho_0 = 27.755
        rho_b = rho_0 / a**3
        H0 = 100


        # def fitting_function(X, a0, a1, a2, a3, a4, a5):
        #     x1, x2 = X
        #     return a0 + a1*x1 + a2*x2 + a3*(x1**2) + a4*(x2**2) + a5*(x1*x2)

        n_ev = x.size // 20
        dc_l_sp = dc_l[0::n_ev]
        dv_l_sp = dv_l[0::n_ev]
        tau_l_sp = tau_l[0::n_ev]
        # yerr_sp = yerr[0::n_ev]
        x_sp = x[0::n_ev]

        if npars == 3:
            def fitting_function(X, a0, a1, a2):
                x1, x2 = X
                return a0 + a1*x1 + a2*x2

            guesses = 1, 1, 1 #, 1, 1, 1
            C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=None, method='lm', absolute_sigma=True)

            fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2])#, C[3], C[4], C[5])
            fit_sp = fit[0::n_ev]

        elif npars == 6:
            def fitting_function(X, a0, a1, a2, a3, a4, a5):
                x1, x2 = X
                return a0 + a1*x1 + a2*x2 + a3*x1**2 + a4*x2**2 + a5*x1*x2

            guesses = 1, 1, 1, 1, 1, 1
            C, cov = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=None, method='lm', absolute_sigma=True)
            fit = fitting_function((dc_l, dv_l), C[0], C[1], C[2], C[3], C[4], C[5])
            fit_sp = fit[0::n_ev]


        resid = fit_sp - tau_l_sp
        chisq = 1#sum((resid / yerr_sp)**2)
        red_chi = 1#chisq / (n_use - 3)

        cs2 = np.real(C[1] / rho_b)
        cv2 = -np.real(C[2] * H0 / (rho_b * np.sqrt(a)))
        ctot2 = (cs2 + cv2)

        err0, err1, err2 = 0, 0, 0#np.sqrt(np.diag(cov))
        yerr = 0
        taus = []
        ctot2 = (cs2 + cv2)
        terr = 0

        x_binned = None

    else:
        a, x, tau_l, dc_l, dv_l, taus, dels, thes, delsq, thesq, delthe, yerr, aic, bic, fit_sp, fit, cov, C, x_binned = binning(j, path, Lambda, kind, nbins_x, nbins_y, npars)

        resid = fit_sp - taus
        chisq = sum((resid / yerr)**2)
        red_chi = chisq / (len(dels) - npars)

        C0, C1, C2 = C[:3]
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
        terr = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    tau_l_k = np.fft.fft(tau_l) / x.size
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    num[Lambda_int+1:ntrunc] = 0
    denom[Lambda_int+1:ntrunc] = 0

    ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

    # Baumann estimator
    T = -dv_l / (H0 / (a**(1/2)))

    def Power_fou(f1, f2):
        f1_k = np.fft.fft(f1)
        f2_k = np.fft.fft(f2)
        corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
        return corr[1]

    ctot2_3 = np.real(Power_fou(tau_l/rho_b, dc_l) / Power_fou(dc_l, T))

    # def Power(f1, f2):
    #     f1_k = np.fft.fft(f1)
    #     f2_k = np.fft.fft(f2)
    #
    #     corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    #     return np.real(np.fft.ifft(corr))
    #
    # A = spectral_calc(tau_l, 1, o=2, d=0) / rho_b / (a**2)
    # T = -dv_l / (H0 / (a**(1/2)))
    # P_AT = Power(A, T)
    # P_dT = Power(dc_l, T)
    # P_Ad = Power(A, dc_l)
    # P_TT = Power(T, T)
    # P_dd = Power(dc_l, dc_l)
    #
    # num_cs2 = (P_AT * spectral_calc(P_dT, 1, o=2, d=0)) - (P_Ad * spectral_calc(P_TT, 1, o=2, d=0))
    # den_cs2 = ((spectral_calc(P_dT, 1, o=2, d=0))**2 / (a**2)) - (spectral_calc(P_dd, 1, o=2, d=0) * spectral_calc(P_TT, 1, o=2, d=0) / a**2)
    #
    # num_cv2 = (P_Ad * spectral_calc(P_dT, 1, o=2, d=0)) - (P_AT * spectral_calc(P_dd, 1, o=2, d=0))
    # cs2_3 = num_cs2 / den_cs2
    # cv2_3 = num_cv2 / den_cs2
    # ctot2_3 = np.median(np.real(cs2_3 + cv2_3))

    # def Power(f1_k, f2_k, Lambda_int):
    #   corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
    #   ntrunc = corr.size - Lambda_int
    #   corr[Lambda_int+1:ntrunc] = 0
    #   return corr
    #
    # A = np.fft.fft(tau_l) / rho_b / tau_l.size
    # T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
    #
    # Ad = Power(A, dc_l, Lambda_int)[mode]
    # AT = Power(A, T, Lambda_int)[mode]
    # P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
    # P_TT = Power(T, T, Lambda_int)[mode]
    # P_dT = Power(dc_l, T, Lambda_int)[mode]
    #
    # cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
    # cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
    #
    # ctot2_3 = np.real(cs2_3 + cv2_3)
    sol_deriv = deriv_param_calc(dc_l, dv_l, tau_l)
    ctot2_4 = sol_deriv[0][1] / rho_b
    err_4 = sol_deriv[1][1] / rho_b

    return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb, P_1l, d1k, taus, x_binned, chisq, ctot2_4, err_4


a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err1_list, err2_list, chi_list, t_err, a_list4, err4_list = [], [], [], [], [], [], [], [], [], [], []

A = [-0.05, 1, -0.5, 11]
Nfiles = 51
Lambda = 3 * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

path, n_runs = 'cosmo_sim_1d/sim_k_1_11/run1/', 8
# path, n_runs = 'cosmo_sim_1d/final_sim_k_1_11/run1/', 16
# path, n_runs = 'cosmo_sim_1d/new_sim_k_1_11/run1/', 24
n_use = n_runs-2
mode = 1

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

f1 = 'curve_fit'
# f1 = ''
nbins_x = 20
nbins_y = 20
npars = 6
data_cov = False #False

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(10, 6))
for j in range(Nfiles):
    sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=f1, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, data_cov=data_cov)
    a_list.append(sol[0])
    ctot2_list.append(sol[2])
    ctot2_2_list.append(sol[3])
    ctot2_3_list.append(sol[4])
    if np.abs(sol[-2]) < 1:
        ctot2_4_list.append(sol[-2])
        err4_list.append(sol[-1])
        a_list4.append(sol[0])
        print('ctot2 = {}, err = {}'.format(sol[-2], sol[-1]))
    else:
        print('boo')
    err1_list.append(sol[6])
    err2_list.append(sol[7])
    chi_list.append(sol[10])
    t_err.append(sol[-8])


    print('a = ', sol[0])
    if flags[j] == 1:
        sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
    else:
        sc_line = ax.axvline(0.5, c='teal', lw=1, zorder=1)
        pass

chi_list = np.array(chi_list)
t_err = np.array(t_err)
ctot2_4_list = np.array(ctot2_4_list)
err4_list = np.array(err4_list)

# ind = np.where(np.max(chi_list) == chi_list)[0][0]
# chi_list[ind] = chi_list[ind-1]


ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
ax.set_xlabel(r'$a$', fontsize=20)
ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)

# print(ctot2_4_list)

# ax.fill_between(a_list, ctot2_list-t_err, ctot2_list+t_err, color='darkslategray', alpha=0.55, zorder=2)
ctot2_line, = ax.plot(a_list, ctot2_list, c='k', lw=1.5, zorder=4, marker='o')
ctot2_2_line, = ax.plot(a_list, ctot2_2_list, c='brown', lw=1.5, marker='*', zorder=2)
ctot2_3_line, = ax.plot(a_list, ctot2_3_list, c='orange', lw=1.5, marker='v', zorder=3)
ctot2_4_line, = ax.plot(a_list4, ctot2_4_list, c='blue', lw=1.5, marker='+', zorder=1)
ctot2_4_err = ax.fill_between(a_list4, ctot2_4_list-err4_list, ctot2_4_list+err4_list, color='darkslategray', alpha=0.55, zorder=2)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, (ctot2_4_line, ctot2_4_err)], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=3, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{C^{+12}}$'], fontsize=14, loc=1, framealpha=1)

# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'], fontsize=14)

ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.yaxis.set_ticks_position('both')

plt.savefig('../plots/test/new_paper_plots/ctot2_ev_6par_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150)

# plt.savefig('../plots/test/new_paper_plots/fit_comps/curve_fit/{}_{}_ctot2_ev_{}.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_ctot2_ev_{}.png'.format(nbins_x, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_{}_ctot2_ev_{}_median_w_binned_fit.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
plt.close()
# plt.show()
