#!/usr/bin/env python3
import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_sm, SPT_tr, write_sim_data, read_sim_data, read_hier
from zel import initial_density


def EFT_solve(j, Lambda, path, A, kind):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x_cell = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C1_nbody = moments_file[:,5]
    M0_hier = M0_nbody

    # a, dx, M0_nbody, M1_nbody, M2_nbody, C0_nbody, C1_nbody, C2_nbody = read_hier(path, j)
    # x = np.arange(0, 1.0, dx)

    x = x_cell
    L = 1.0 #x[-1]
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    m_nb = rho_0 / x_nbody.size
    H0 = 100

    def dc_in_finder(path, j, x):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(0)
        moments_file = np.genfromtxt(path + moments_filename)
        a0 = moments_file[:,-1][0]

        initial_file = np.genfromtxt(path + 'output_initial.txt')
        q = initial_file[:,0]
        Psi = initial_file[:,1]

        nbody_file = np.genfromtxt(path + 'output_{0:04d}.txt'.format(j))
        x_in = nbody_file[:,-1]

        Nx = x_in.size
        L = np.max(x_in)
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
        dc_in = -spectral_calc(Psi, L, o=1, d=0) / a0

        f = interp1d(q, dc_in, kind='cubic', fill_value='extrapolate')
        dc_in = f(x)

        return dc_in

    dc_in = dc_in_finder(path, j, x)
    d1k, P_1l_a_sm, P_2l_a_sm = SPT_sm(dc_in, k, L, Lambda, a)

    d1k, d2k, P_1l_a_tr, P_2l_a_tr = SPT_tr(dc_in, k, L, Lambda, kind, a)

    P_lin_a = np.real(d1k * np.conj(d1k)) * (a**2)

    dk_par, a, dx = read_density(path, j)
    x_grid = np.arange(0, 1.0, dx)
    if str(path[-2]) == '1':
        k_par = np.fft.ifftshift(2.0 * np.pi * np.arange(-x_grid.size/2, x_grid.size/2))
        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par = (M0_par / np.mean(M0_par)) - 1
        M0_par = smoothing(M0_par, k_par, Lambda, kind)
        M0_k = np.fft.fft(M0_par) / M0_par.size
        P_nb_a = np.real(M0_k * np.conj(M0_k))
    else:
        # print(path)
        # print('This is not the main run, saving zeros for spectrum', path[-2])
        P_nb_a = np.zeros(d1k.size)


    M0 = M0_nbody * rho_b #this makes M0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)
    M1 = M1_nbody * rho_b / a #this makes M1 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)
    M2 = M2_nbody * rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    #now all long-wavelength moments
    rho = M0
    rho_l = smoothing(M0, k, Lambda, kind) #this is ρ_{l}
    rho_s = rho - rho_l
    pi_l = smoothing(M1, k, Lambda, kind) #this is π_{l}
    sigma_l = smoothing(M2, k, Lambda, kind) #this is σ_{l}
    dc = M0_nbody - 1 #this is the overdensity δ from the hierarchy
    dc_l = (rho_l / rho_b) - 1
    dc_s = dc - dc_l

    #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
    #in 1D, κ_{l} = σ_{l} - ([π_{l}]^{2} / ρ_{l})
    kappa_l = (sigma_l - (pi_l**2 / rho_l))

    v_l = pi_l / rho_l
    dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{l}

    #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
    rhs = (3 * H0**2 / (2 * a)) * dc #using the hierarchy δ here
    phi = spectral_calc(rhs, L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    rhs_l = (3 * H0**2 / (2 * a)) * dc_l
    phi_l = spectral_calc(rhs_l, L, o=2, d=1)
    grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

    grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
    grad_phi2_l = smoothing(grad_phi**2, k, Lambda, kind) #this is [(∇ϕ)^2]_l

    phi_s = phi - phi_l

    dx_phi = spectral_calc(phi, L, o=1, d=0)
    dx_phi_l = spectral_calc(phi_l, L, o=1, d=0)
    dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)
    # Phi_l = -(rho_0 / (3 * (H0**2) * (a**2))) * smoothing(dx_phi_s**2, k, Lambda)

    Phi_l_true = smoothing(dx_phi * rho, k, Lambda, kind)
    Phi_l_cgpt = (dx_phi_l * rho_l) + smoothing(dx_phi_s * rho_s, k, Lambda, kind)
    Phi_l_bau = (dx_phi_l * rho_l) + spectral_calc(smoothing(dx_phi_s**2, k, Lambda, kind), L, o=1, d=0) / (3 * H0**2 * a**2 / rho_0)

    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_d2 = (rho_l * (dv_l**2) / Lambda**2) - (dx_phi_l**2 / (3 * H0**2 * a**2 / rho_0))
    tau_l = (kappa_l + Phi_l)


    return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, d2k, kappa_l, Phi_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, M0_nbody, M0_hier, v_l, M2_nbody, C1_nbody
    # return a, x, d1k, dc_l, dv_l, tau_l, P_nb_a, P_1l_a_tr

def param_calc(j, Lambda, path, A, mode, kind):
    a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l_0, dc_l, dv_l, d1k, d2k, kappa_l, Phi_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, M0_nbody, M0_hier, v_l, M2, C1_nbody = EFT_solve(j, Lambda, path, A, kind)
    file_num = j
    taus = []
    taus.append(tau_l_0)
    #we already save the \tau from run1 in the last line. the next loop should run from 2 to 5 (or 9)
    for run in range(2, 9):
        path = path[:-2] + '{}/'.format(run)
        sol_new = EFT_solve(j, Lambda, path, A, kind)
        taus.append(sol_new[9])

    Nt = len(taus)
    # print(Nt)
    tau_l = sum(np.array(taus)) / len(taus)
    rho_0 = 27.755
    rho_b = rho_0 / a**3
    H0 = 100

    diff = np.array([(taus[i] - taus[0])**2 for i in range(1, Nt)])
    yerr = np.sqrt(sum(diff) / (Nt*(Nt-1)))

    n_use = 10
    n_ev = int(250000 / n_use)
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

    cs2 = np.real(C1 / rho_b)
    cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
    ctot2 = (cs2 + cv2)

    resid = fit_sp - tau_l_sp
    chisq = sum((resid / yerr_sp)**2)
    red_chi = chisq / (n_use - 3)

    cov = np.array(cov)
    corr = np.zeros(cov.shape)

    for i in range(3):
        for j in range(3):
            corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])


    #to propagate the errors from C_i to c^{2}, we must divide by rho_b (the minus sign doesn't matter as we square the sigmas)
    err1 /= rho_b
    err2 /= rho_b

    total_err = np.sqrt(err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)
    # print(ctot2, total_err)

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))[:Lambda_int]
    denom = ((d1k * np.conj(d1k)) * (a**2))[:Lambda_int]
    ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

    # Baumann estimator
    def Power(f1_k, f2_k):
      corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
      return corr

    A = np.fft.fft(tau_l) / rho_b / tau_l.size
    T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
    d = np.fft.fft(dc_l) / dc_l.size
    Ad = Power(A, dc_l)[mode]
    AT = Power(A, T)[mode]
    P_dd = Power(dc_l, dc_l)[mode]
    P_TT = Power(T, T)[mode]
    P_dT = Power(dc_l, T)[mode]

    cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
    cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)

    ctot2_3 = np.real(cs2_3 + cv2_3)

    errors = [(tau - tau_l) * 100 / tau_l for tau in taus]
    linestyles = ['solid', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)), 'dashdot', 'dashed', 'dotted']
    colors = ['brown', 'darkcyan', 'dimgray', 'violet', 'orange', 'cyan', 'b', 'r', 'k']
    labels = [r'$\left<[\tau]_{\Lambda}\right>$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
    # print(len(taus), len(linestyles), len(colors), len(labels), len(errors))

    plots_folder = 'sim_k_1_11/tau_all/'
    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
    ax[0].set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=14)

    ax[0].plot(x, tau_l, c=colors[0], lw=2.5, ls=linestyles[0], label=labels[0])

    for i in range(8):
        ax[0].plot(x, taus[i], c=colors[i+1], lw=2.5, ls=linestyles[i+1], label=labels[i+1])
        ax[1].plot(x, errors[i], c=colors[i+1], lw=2.5, ls=linestyles[i+1])

    ax[0].set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
    ax[1].set_ylabel('% err', fontsize=14)
    ax[1].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
    ax[0].legend(fontsize=12, bbox_to_anchor=(1,1))

    ax[1].axhline(0, c='brown', lw=2.5)
    for i in range(2):
        ax[i].minorticks_on()
        ax[i].tick_params(axis='both', which='both', direction='in', labelsize=12)
        ax[i].ticklabel_format(scilimits=(-2, 3))
        ax[i].yaxis.set_ticks_position('both')
    plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, file_num), bbox_inches='tight', dpi=150)
    plt.close()
    # plt.show()

    # return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody, d1k, d2k, total_err
    # return a, x, ctot2, ctot2_2, ctot2_3, total_err#0, err1, err2, total_err #cs2, cv2, chi_squared
    return a, x, d1k, dc_l, dv_l, tau_l, P_nb_a, P_1l_a_tr
    # return a, x, tau_l, fit, yerr, red_chi

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
mode = 1
Lambda = (2*np.pi) * 3

A = [-0.05, 1, -0.5, 11]
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
Nfiles = 51
for j in range(51):
    # j = 0
    sol = param_calc(j, Lambda, path, A, mode, kind)
    print('a = {}\n'.format(sol[0]))

# path = 'cosmo_sim_1d/nbody_phase_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# Lambda = 3 * (2 * np.pi)
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'

# a_list, cs2_list, cv2_list, err1_list, err2_list, chi_list = [], [], [], [], [], []
#
# for j in range(2):
#     sol = param_calc(j, Lambda, path, A, mode, kind)
#     cs2_list.append(sol[8])
#     cv2_list.append(sol[9])
#     err1_list.append(sol[6])
#     err2_list.append(sol[7])
#     chi_list.append(sol[10])
#     a_list.append(sol[0])
#     print('a = ', sol[0])
#
# data = zip(a_list, cs2_list, cv2_list, err1_list, err2_list, chi_list)
# header = ['a', 'cs2', 'cv2', 'err_cs2', 'err_cv2', 'chi_sq']
# import csv
# with open('../data/phase_runs_params_{}.csv'.format(kind), 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     for row in data:
#         writer.writerow(row)


# results = pd.read_csv('../phase_runs_params_{}.csv'.format(kind))
# a = results['a']
# cs2 = results['cs2']
# cv2 = results['cv2']
# err_cs2 = results['err_cs2']
# err_cv2 = results['err_cv2']
#
# ctot2 = cs2 + cv2
# err_ctot2 = np.sqrt(err_cs2**2 + err_cv2**2)
#
# colour1 = 'b'
# s1 = 5
# fig, ax = plt.subplots()
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.set_xlabel(r'$a$', fontsize=16)
# ax.set_ylabel('$c^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=16)
# ax.errorbar(a, ctot2, yerr=err_ctot2, c='k', lw=2, marker='o', markeredgecolor=colour1, markerfacecolor=colour1, markersize=s1, ecolor=colour1, label=r'$c^{2}_{\mathrm{tot}}$')
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11)
# # plt.show()
# plt.savefig('../plots/test/ctot2/err/ctot2_w_err.png', bbox_inches='tight', dpi=150)
# plt.close()
#
#
# fig, ax = plt.subplots()
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.set_xlabel(r'$a$', fontsize=16)
# ax.set_ylabel('$c^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=16)
# ax.errorbar(a, cs2, yerr=err_cs2, c='k', lw=2, marker='o', markeredgecolor=colour1, markerfacecolor=colour1, markersize=s1, ecolor=colour1, label=r'$c^{2}_{\mathrm{s}}$')
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11)
# # plt.show()
# plt.savefig('../plots/test/ctot2/err/cs2_w_err.png', bbox_inches='tight', dpi=150)
# plt.close()
#
# fig, ax = plt.subplots()
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.set_xlabel(r'$a$', fontsize=16)
# ax.set_ylabel('$c^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=16)
# ax.errorbar(a, cv2, yerr=err_cv2, c='k', lw=2, marker='o', markeredgecolor=colour1, markerfacecolor=colour1, markersize=s1, ecolor=colour1, label=r'$c^{2}_{\mathrm{v}}$')
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11)
# # plt.show()
# plt.savefig('../plots/test/ctot2/err/cv2_w_err.png', bbox_inches='tight', dpi=150)
# plt.close()
#
# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'
#
# Lambda_list = np.arange(2, 7)
# nums = [[0, 15], [35, 45]]
# ctot2_0_list, ctot2_1_list, ctot2_2_list, error_list = [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]]
# a_list = [[[], []], [[], []]]
#
# for i1 in range(2):
#     print(i1)
#     for i2 in range(2):
#         print(i2)
#         file_num = nums[i1][i2]
#         c0, c1, c2, err = [], [], [], []
#         for i3 in range(Lambda_list.size):
#             print(Lambda_list[i3])
#             Lambda = Lambda_list[i3] * (2*np.pi)
#             sol = param_calc(file_num, Lambda, path, A, mode, kind)
#             df = pd.DataFrame(sol)
#             pickle.dump(df, open("lam_dep/lam_dep_sol_{}_{}_{}_{}.p".format(kind, i1, i2, i3), "wb"))
#             # df = pd.DataFrame(pickle.load(open("lam_dep/lam_dep_sol_{}_{}_{}_{}.p".format(kind, i1, i2, i3), "rb" )))
#             # # sol_0 = df.iloc[0]
#             # sol = [df.iloc[j] for j in range(len(df.index))]
#             # a_list[i1][i2] = float(sol[0])
#             # c0.append(float(sol[2]))
#             # c1.append(float(sol[3]))
#             # c2.append(float(sol[4]))
#             # err.append(float(sol[5]))
#             # print(float(sol[2]), float(sol[5]))
# #         ctot2_0_list[i1][i2] = c0
# #         ctot2_1_list[i1][i2] = c1
# #         ctot2_2_list[i1][i2] = c2
# #         error_list[i1][i2] = err
# #
# # # print(ctot2_0_list, errors)
# #
# # error_list = np.array(error_list)
# # ctot2_0_list = np.array(ctot2_0_list)
# #
# # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
# # for i in range(2):
# #     print(i)
# #     # ax[0,0].set_ylabel(r'$c^{2}_{v}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
# #     # ax[0,0].set_ylabel(r'$c^{2}_{s}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
# #     ax[0,0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
# #     ax[1,1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), x=0, fontsize=16)
# #     ax[i,1].tick_params(labelleft=False, labelright=True)
# #     for j in range(2):
# #         print(ctot2_0_list[i][j])
# #         ax[i,j].set_title(r'$a = {}$'.format(a_list[i][j]), x=0.25, y=0.9)
# #         # ax[i,j].fill_between(Lambda_list, ctot2_0_list[i][j]-error_list[i][j], ctot2_0_list[i][j]+error_list[i][j], color='darkslategray', alpha=0.35, rasterized=True)
# #
# #
# #         ax[i,j].plot(Lambda_list, ctot2_1_list[i][j], c='cyan', lw=1.5, marker='v', label=r'M&W')
# #         ax[i,j].plot(Lambda_list, ctot2_2_list[i][j], c='orange', lw=1.5, marker='*', label=r'$B^{+12}$')
# #         ax[i,j].errorbar(Lambda_list, ctot2_0_list[i][j], yerr=error_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
# #         # ax[i,j].plot(Lambda_list, ctot2_0_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
# #         ax[i,j].minorticks_on()
# #         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
# #         ax[i,j].yaxis.set_ticks_position('both')
# #
# #     # if kind == 'sharp':
# #     #     ax[0,0].set_ylim(0.206, 0.242)
# #     #     ax[0,1].set_ylim(-6, 9.4)
# #     #     ax[1,0].set_ylim(-1.1, 3.2)
# #     #     ax[1,1].set_ylim(0, 1.5)
# #     #
# #     # elif kind == 'gaussian':
# #     #     ax[1,0].set_ylim(0, 1.7)
# #
# #     else:
# #         pass
# #
# #
# # plt.legend(fontsize=12, bbox_to_anchor=(1,1))
# # fig.align_labels()
# # plt.subplots_adjust(hspace=0, wspace=0)
# # plt.savefig('../plots/test/ctot2/Lam_dep/ctot2_lambda_dep_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# # plt.savefig('../plots/test/ctot2/Lam_dep/ctot2_lambda_dep_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# # # plt.show()
# # # plt.close()

# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# Lambda = (2*np.pi) * 3
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'

# kinds = ['sharp', 'gaussian']
# kinds_txt = ['sharp cutoff', 'Gaussian smoothing']

# for kind in kinds:
#     sol_0 = param_calc(0, Lambda, path, A, mode, kind)
#     sol_1 = param_calc(13, Lambda, path, A, mode, kind)
#     sol_2 = param_calc(30, Lambda, path, A, mode, kind)
#     sol_3 = param_calc(46, Lambda, path, A, mode, kind)
#     df = pandas.DataFrame(data=[sol_0, sol_1, sol_2, sol_3])
#     pickle.dump(df, open("tau_fits_plots_{}.p".format(kind), "wb"))

# for j in range(2):
#     kind = kinds[j]
#     kind_txt = kinds_txt[j]
#
#     df = pandas.DataFrame(pickle.load(open("tau_fits_plots_{}.p".format(kind), "rb" )))
#
#     sol_0 = df.iloc[0]
#     sol_1 = df.iloc[1]
#     sol_2 = df.iloc[2]
#     sol_3 = df.iloc[3]
#
#     # print([sol_0, sol_1, sol_2, sol_3])
#
#     x = sol_0[1]
#     a_list = [[sol_0[0], sol_1[0]], [sol_2[0], sol_3[0]]]
#     tau_list = [[sol_0[2], sol_1[2]], [sol_2[2], sol_3[2]]]
#     fit_list = [[sol_0[3], sol_1[3]], [sol_2[3], sol_3[3]]]
#     yerr_list = [[sol_0[4], sol_1[4]], [sol_2[4], sol_3[4]]]
#     chi_list = [[sol_0[5], sol_1[5]], [sol_2[5], sol_3[5]]]
#     # C_list = [[sol_0[4], sol_1[4]], [sol_2[4], sol_3[4]]]
#     print(chi_list)
#     # C_list = [[sol_0[4], [[0,0,0], [0,0,0]]], [[0,0,0], [0,0,0]]]
#
#     fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
#     fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16, x=0.5, y=0.92)
#     for i in range(2):
#         ax[i,0].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#         ax[i,1].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#         ax[i,1].yaxis.set_label_position('right')
#
#         ax[1,i].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
#         ax[i,1].tick_params(labelleft=False, labelright=True)
#         ax[1,i].set_title(r'$a = {}$'.format(a_list[1][i]), x=0.15, y=0.9)
#         ax[0,i].set_title(r'$a = {}$'.format(a_list[0][i]), x=0.15, y=0.9)
#         for j in range(2):
#             # ax[i,j].errorbar(x, tau_list[i][j], yerr=yerr_list[i][j], ecolor='r', errorevery=10000, c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].plot(x, tau_list[i][j], c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].fill_between(x, tau_list[i][j]-yerr_list[i][j], tau_list[i][j]+yerr_list[i][j], color='darkslategray', alpha=0.35, rasterized=True)
#             ax[i,j].plot(x, fit_list[i][j], c='k', lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].minorticks_on()
#             ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#             # err_str = r'$C_{{0}} = {}$'.format(np.round(C_list[i][j][0], 3)) + '\n' + r'$C_{{1}} = {}$'.format(np.round(C_list[i][j][1], 3)) + '\n' + r'$C_{{2}} = {}$'.format(np.round(C_list[i][j][2], 3))
#             # ax[i,j].text(0.35, 0.05, err_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i,j].transAxes)
#             chi_str = r'$\chi^{{2}}/{{\mathrm{{d.o.f.}}}} = {}$'.format(chi_list[i][j])
#             # ax[i,j].text(0.35, 0.05, chi_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i,j].transAxes)
#
#             ax[i,j].yaxis.set_ticks_position('both')
#
#     # for tick in ax[1,1].yaxis.get_majorticklabels():
#     #     tick.set_horizontalalignment("right")
#
#     plt.legend(fontsize=12, bbox_to_anchor=(0.975, 2.2))
#     fig.align_labels()
#     plt.subplots_adjust(hspace=0, wspace=0)
#     plt.savefig('../plots/test/paper_plots/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
#     plt.close()
#     # plt.savefig('../plots/test/early_tau_av_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
#     # plt.show()

# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11] #regular
# Lambda = (2*np.pi) * 3
# kind = 'sharp'
# C1_list, C2_list = [], []
# err1_list, err2_list, a_list = [], [], []
#
# for j in range(10, 51):
#     sol = param_calc(j, Lambda, path, A, mode, kind)
#     C1_list.append(sol[1][0])
#     err1_list.append(sol[2][0])
#     a_list.append(sol[0])
#     print('a = ', sol[0])
#
# fig, ax = plt.subplots()
# ax.errorbar(a_list, C1_list, yerr=err1_list, c='r', lw=2, marker='o')
# ax.set_ylabel(r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\,s^{-2}]$', fontsize=14)
# ax.set_xlabel(r'$a$', fontsize=14)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.yaxis.set_ticks_position('both')
# plt.savefig('../plots/test/av4_ctot2_err.png', bbox_inches='tight', dpi=150)
# plt.close()


# #solving the weighted linear least-squares of the form Y = X\beta + resid, with weight matrix W
# # X = np.array([n_ev, sum(dc_l_sp), sum(dv_l_sp), sum(dc_l_sp), sum(dc_l_sp**2), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp**2)])
# # Y = np.array([sum(tau_l_sp), sum(tau_l_sp*dc_l_sp), sum(tau_l_sp*dv_l_sp)])
# X = (np.array([np.ones(len(dc_l_sp)), dc_l_sp, dv_l_sp])).reshape((len(dc_l_sp), 3))
# Y = np.array(tau_l_sp)
# W = np.diag(np.array(yerr_sp))
# A = np.linalg.inv(X.T.dot(W.dot(X)))
# B = X.T.dot(W.dot(Y))
# C0_, C1_, C2_ = A.dot(B)
# COV = np.linalg.inv(X.T.dot(W.dot(X)))
#
# print(C0_, C1_, C2_)
