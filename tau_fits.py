#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from functions import read_sim_data, sub_find
from scipy.optimize import curve_fit

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
n_runs = 8
A = [-0.05, 1, -0.5, 11]

# path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
# n_runs = 24
# A = [-0.05, 1, -0.5, 11]

mode = 1
Lambda = (2*np.pi) * 3
n_use = n_runs-1
kinds = ['sharp', 'gaussian']
kinds_txt = ['sharp cutoff', 'Gaussian smoothing']

which = 1
kind = kinds[which]
kind_txt = kinds_txt[which]

def calc(j, Lambda, path, A, mode, kind, n_runs, n_use, n_fits):
    a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
    taus = []
    taus.append(tau_l_0)
    for run in range(2, n_runs+1):
        if run > 10:
            ind = -3
        else:
            ind = -2
        path = path[:ind] + '{}/'.format(run)
        taus.append(read_sim_data(path, Lambda, kind, j)[1])

    Nt = len(taus)

    tau_l = sum(np.array(taus)) / Nt

    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt-1))

    guesses = 1, 1, 1
    def fitting_function(X, a0, a1, a2):
        x1, x2 = X
        return a0 + a1*x1 + a2*x2

    FF = []
    for j in range(n_fits):
        sub = sub_find(n_use, 62500)
        dc_l_sp = np.array([dc_l[j] for j in sub])
        dv_l_sp = np.array([dv_l[j] for j in sub])
        tau_l_sp = np.array([tau_l[j] for j in sub])
        yerr_sp = np.array([yerr[j] for j in sub])
        x_sp = np.array([x[j] for j in sub])
        taus_sp = np.array([np.array([taus[k][j] for j in sub]) for k in range(n_runs)])

        cov_mat = np.empty(shape=(n_use, n_use))
        corr_mat = np.empty(shape=(n_use, n_use))

        for i in range(n_use):
            for j in range(n_use):
                tau_k_i = np.array([taus_sp[k][i] for k in range(n_runs)])
                tau_k_j = np.array([taus_sp[k][j] for k in range(n_runs)])
                cov = np.cov(tau_k_i, tau_k_j)
                cov_mat[i][j] = cov[0][1]

        # for i in range(n_use):
        #     for j in range(n_use):
        #         corr_mat[i,j] = cov_mat[i,j] / np.sqrt(cov_mat[i,i] * cov_mat[j,j])
        #
        # import seaborn as sns
        # plt.figure(figsize=(10,10))
        # hm = sns.heatmap(corr_mat,
        #              cbar=True,
        #              annot=True,
        #              square=True,
        #              fmt='.3f',
        #              annot_kws={'size': 10})
        # plt.title('Correlation matrix')
        # plt.xlabel(r'$\tau(x_{j})$', fontsize=16)
        # plt.ylabel(r'$\tau(x_{i})$', fontsize=16)
        #
        # plt.tight_layout()
        # plt.show()
        # # plt.savefig('../plots/test/new_paper_plots/data_corr_{}.png'.format(n_use), bbox_inches='tight', dpi=150)
        #


        FF.append(curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=cov_mat, method='lm', absolute_sigma=True))
        # FF.append(curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True))

    C0 = np.mean([FF[k][0][0] for k in range(n_fits)]) #/ n_fits
    C1 = np.mean([FF[k][0][1] for k in range(n_fits)]) #/ n_fits
    C2 = np.mean([FF[k][0][2] for k in range(n_fits)]) #/ n_fits
    C = [C0, C1, C2]

    # cov = np.sqrt(np.array([sum([(FF[k][1])**2 for k in range(n_fits)])]) / n_fits)
    cov = np.sqrt((sum([(FF[k][1])**2 for k in range(n_fits)])) / n_fits)

    fit = fitting_function((dc_l, dv_l), C0, C1, C2)
    fit_sp = np.array([fit[j] for j in sub])
    resid = fit_sp - tau_l_sp
    chisq = sum((resid / yerr_sp)**2)
    red_chi = chisq / (n_use - 3)

    cs2 = np.real(C1 / rho_b)
    cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
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

    for i in range(3):
        for j in range(3):
            corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])

    err0, err1, err2 = np.sqrt(np.diag(cov))

    ctot2 = (cs2 + cv2)
    terr = (err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)**(0.5)
    return a, x, tau_l, fit, dc_l, dv_l, dc_l_sp, dv_l_sp, tau_l_sp, fit_sp, x_sp, C

j = 12
n_fits = 100
a, x, tau_l, fit, dc_l, dv_l, dc_l_sp, dv_l_sp, tau_l_sp, fit_sp, x_sp, C = calc(j, Lambda, path, A, mode, kind, n_runs, n_use, n_fits)


# from matplotlib.pyplot import cm
# color = iter(cm.rainbow(np.linspace(0, 1, M)))
# ls = iter([':', '-.', '-', '--']*6)

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots()
fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)#, x=0.5, y=0.92)
ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)

ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
ax.set_title(r'$a = {}$'.format(np.round(a, 3)), x=0.15, y=0.9)

ax.plot(x, tau_l, c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
ax.plot(x, fit, c='k', lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
# # # ax.plot(x, mean_fit, c='k', lw=1.5, ls=next(ls), label=r'mean fit to $\left<[\tau]_{\Lambda}\right>$')
# # # # ax.plot(x, mean_fit, c='k', lw=1.5, ls=next(ls), label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
# # #
# # # # for n in range(M):
# # # #     ax.plot(x, fits[n], c=next(color), lw=1.5, ls=next(ls))
# # #
ax.plot(x_sp, tau_l_sp, c='r', lw=1.5, marker='+', ls='dashdot', label=r'$\left<[\tau]_{\Lambda}\right>$: sampled')
ax.plot(x_sp, fit_sp, c='orange', lw=2, marker='o', ls='dotted', label=r'fit to $\left<[\tau]_{\Lambda}\right>$: sampled')
# # # # # # ax.plot(x, fit, c='k', lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
# # # # # # # fac = -np.sqrt(a) / 100
# # # # # # # # ax.plot(x, dc_l, c='r', label=r'$\delta_{l}$')
# # # # # # # # ax.plot(x, dv_l*fac, c='b', ls='dashdot', label=r'$\theta_{l}$')
# # # # # # # # ax.plot(x, C[0] - C[1]*dc_l + C[2]*dv_l, c='r', ls='dotted', label=r'fit to $\tau_{l}$')
# # # # # # # fit_man = a0 + a2*dv_l # + a2*dv_l
# # # # # # # ax.plot(x, fit_man, c='r', ls='dotted', label=r'fit to $\tau_{l}$')

# # C = [8.050853484043458, 5104.251420590308, 35.625904458402644]
# m = 1.400
# C[1] /= m
# # C[2] /= m
# tauu = C[1]*dc_l + C[2]*dv_l
# ax.plot(x, tauu, lw=1.5, c='b')
# # ax.plot(x, dc_l, lw=1.5, c='b')
# # ax.plot(x, -dv_l / (100 / np.sqrt(a)), lw=1.5, c='r', ls='dashed')

# ax.plot(x_sp, dc_l_sp, lw=1.5, c='b')
# ax.plot(x_sp, -dv_l_sp / (100 / np.sqrt(a)), lw=1.5, c='r')


ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)

plt.legend(fontsize=12)
# chi_str = r'$\chi^{{2}}/{{\mathrm{{d.o.f.}}}} = {}$'.format(np.round(chi_list, 3))
# ax.text(0.35, 0.05, chi_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax.transAxes)
# ax.yaxis.set_ticks_position('both')
plots_folder = 'sim_k_1_11/sm/gauss_tau/'

fig.align_labels()
plt.subplots_adjust(hspace=0, wspace=0)
# plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
# plt.close()
plt.show()
