#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_sm, SPT_tr, dc_in_finder
from zel import initial_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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

    x = x_cell
    L = x[-1]
    Nx = x.size
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    rho_0 = 27.755 #this is the comoving background density
    rho_b = rho_0 / (a**3) #this is the physical background density
    m_nb = rho_0 / x_nbody.size
    H0 = 100

    dc_in, k_in = dc_in_finder(path, x, interp=True)
    d1k, P_1l_a_sm, P_2l_a_sm = SPT_sm(dc_in, k_in, L, Lambda, a)

    # dc_in = smoothing(dc_in, k, Lambda, kind)
    d1k, d2k, P_1l_a_tr, P_2l_a_tr = SPT_tr(dc_in, k_in, L, Lambda, kind, a)

    P_lin_a = np.real(d1k * np.conj(d1k)) * (a**2)

    dk_par, a, dx = read_density(path, j)
    x_grid = np.arange(0, 1.0, dx)
    k_par = np.fft.ifftshift(2.0 * np.pi * np.arange(-x_grid.size/2, x_grid.size/2))
    M0_par = np.real(np.fft.ifft(dk_par))
    M0_par = (M0_par / np.mean(M0_par)) - 1
    M0_par = smoothing(M0_par, k_par, Lambda, kind)
    M0_k = np.fft.fft(M0_par) / M0_par.size
    P_nb_a = np.real(M0_k * np.conj(M0_k))

    # M0_nbody = M0_par
    # print(np.fft.fft(M0_nbody)[0] / M0_nbody.size)
    M0_nbody /= np.mean(M0_nbody)

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
    # print('dv_l = ', np.real(np.fft.fft(dv_l)[:12]))
    # print('\ndc_l = ', np.real(np.fft.fft(dc_l)[:12]))

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

    M2_l = smoothing(M2, k, Lambda, kind)
    V_l = smoothing(M0 * grad_phi, k, Lambda, kind) * x
    vir = M2_l - V_l

    # plt.plot(x, tau_l, c='b', lw=2)
    # print(np.sum(tau_l))
    # plt.show()

    # print(d1k.size, tau_l.size)
    return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, d2k, kappa_l, Phi_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, vir, M0_nbody, M0_hier, v_l, M2_nbody, C1_nbody

def param_calc(j, Lambda, path, A, mode, kind):
   a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, d2k, kappa_l, Phi_l, Phi_l_true, Phi_l_cgpt, Phi_l_bau, vir, M0_nbody, M0_hier, v_l, M2, C1_nbody = EFT_solve(j, Lambda, path, A, kind)
   rho_0 = 27.755
   rho_b = rho_0 / a**3
   H0 = 100
   # err_nb = np.abs(P_nb_a[1] - P_lin_a[1]) * 100 / P_lin_a[1]
   # err_1l = np.abs(P_1l_a_tr[1] - P_lin_a[1]) * 100 / P_lin_a[1]
   #
   # print("N-body abs err: {}%".format(np.round(err_nb, 4)))
   # print("1-loop SPT abs err: {}% \n".format(np.round(err_1l, 4)))

   # for 3 parameters a0, a1, a2 such that τ_l = a0 + a1 × (δ_l) + a2 × dv_l
   def fitting_function(X, a0, a1, a2):
      x1, x2 = X
      return a0 + a1*x1 + a2*x2

   guesses = 1, 1, 1
   FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=1e-15*np.ones(x.size), method='lm')
   C0, C1, C2 = FF[0]
   cov = FF[1]
   err0, err1, err2 = np.sqrt(np.diag(cov))
   fit = fitting_function((dc_l, dv_l), C0, C1, C2)
   # C2 = C2 * (np.sqrt(a) / H0)
   C = [C0, C1, C2]

   cs2 = np.real(C1 / rho_b)
   cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
   ctot2 = (cs2 + cv2)
   # print('\nFit:')
   # print('cs2 = ', cs2)
   # print('cv2 = ', cv2)
   # print('cs2 + cv2 = ', ctot2)

   #to propagate the errors from C_i to c^{2}, we must divide by rho_b (the minus sign doesn't matter as we square the sigmas)
   err1 /= rho_b
   err2 /= rho_b
   total_err = np.sqrt(err1**2 + err2**2)

   # M&W Estimator
   Lambda_int = int(Lambda / (2*np.pi))
   num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))[:Lambda_int]
   denom = P_lin_a[:Lambda_int]
   ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b

   # print('num = ', sum(num))
   # print('den = ', sum(denom))

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
   #
   # num_cv2 = (P_Ad * spectral_calc(P_dT, 1, o=2, d=0)) - (P_AT * spectral_calc(P_dd, 1, o=2, d=0))
   # print(num_cs2, den_cs2)
   # cs2_3 = num_cs2 / den_cs2
   # print(cs2_3)
   # cv2_3 = num_cv2 / den_cs2

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

   num_cs = (P_TT * Ad) - (P_dT * AT)
   num_cv = (P_dT * Ad) - (P_dd * AT)
   den = (P_dd * P_TT - (P_dT)**2)
   cs2_3 = num_cs / den
   cv2_3 = num_cv / den

   ctot2_3 = np.real(cs2_3 + cv2_3)

   # print(P_TT, Ad, P_dT, AT, num_cs)
   # print(P_TT*Ad, P_dT*AT, num_cs)

   # print(num_cs, num_cv, den)
   print(cs2_3, cv2_3, ctot2_3)
   # print(ctot2, ctot2_2, ctot2_3)

   # print('fit: ', ctot2)
   # print('MW: ', ctot2_2)
   # print('percentage difference: ', (ctot2_2 - ctot2) / ctot2 * 100)

   # print(ctot2, ctot2_2, ctot2_3)
   # num = (P_TT * Ad) - (P_dT * AT)
   # den = (P_dd * P_TT - (P_dT)**2)
   # print('\nBaumann')
   # print('P_dd = ', P_dd)
   # print('P_dT = ', P_dT)
   # print('P_TT = ', P_TT)
   # # print('P_Ad = ', Ad)
   # # print('P_AT = ', AT)
   # # print(P_TT * Ad)
   # # print(P_dT * AT)

   # # print(d)
   # # print(np.fft.fft(M0_nbody)/M0_nbody.size)
   # # print(T)
   # print('cs2:')
   # print('num1 = ', (P_TT * Ad))
   # print('num2 = ', (P_dT * AT))
   # print('num =', num)
   #
   # print('den1 = ', (P_dd * P_TT))
   # print('den2 = ', (P_dT)**2)
   # print('den = ', den)
   # #
   # # print('ratio = ', cs2_3)
   # #
   # # print('cv2:')
   # # print('num1 = ', (P_dT * Ad))
   # # print('num2 = ', (P_dd * AT))
   # # print('num = ', (P_dT * Ad)-(P_dd * AT))
   # # print('ratio = ', cv2_3)
   # # print('sum = ', ctot2_3)
   #
   # print('cs2 + cv2 = ', ctot2_3)
   # # print('Fit: ', ctot2)
   # # print('M&W: ', ctot2_2)
   # # print('Baumann: ', ctot2_3, '\n')
   # A *= tau_l.size
   # stoch = np.real(np.fft.ifft(Power(A, A)))
   # #
   # plots_folder = '/sim_k_1_11/' #/paper_plots'
   # if True:
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, kappa_l, c='b', lw=2, label=r'$\kappa_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\kappa_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     # plt.show()
   #     plt.savefig('../plots/{}/kappa/kappa_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #     # fig, ax = plt.subplots()
   #     # ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     # ax.plot(x, Phi_l_true, c='k', lw=2, label='true')
   #     # ax.plot(x, Phi_l_cgpt, c='b', lw=2, label='CGPT', ls='dashdot')
   #     # ax.plot(x, Phi_l_bau, c='r', lw=2, label='Baumann', ls='dashed')
   #     # ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     # ax.set_ylabel(r'$\left[\rho\partial_{x}\phi\right]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #     # ax.minorticks_on()
   #     # ax.tick_params(axis='both', which='both', direction='in')
   #     # ax.ticklabel_format(scilimits=(-2, 3))
   #     # ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     # ax.yaxis.set_ticks_position('both')
   #     # plt.savefig('../plots/{}/Phi_cgpt/Phi_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     # plt.close()
   #
   #     # C1_bau = rho_b*np.real(cs2_3)
   #     # C2_bau = -rho_b*np.real(cv2_3)*np.sqrt(a)/H0
   #     # print(dc_l, dv_l)
   #     # fit_bau = C0 + C1_bau*dc_l + C2_bau*dv_l
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=12)
   #     ax.plot(x, tau_l, c='b', lw=2, label=r'$[\tau]_{\Lambda}$')
   #     ax.plot(x, fit, c='k', ls='dashed', lw=2, label=r'fit to $[\tau]_{\Lambda}$') #label=r'$\left<[\tau]_{\Lambda}\right>$ (fit)')
   #     # # ax.plot(x, fit_bau, c='y', ls='dotted', lw=2, label=r'fit from $B^{+12}$')
   #
   #     # stoch = (tau_l - fit)
   #     # ax.plot(x, stoch, c='r', lw=2, label=r'$[\tau]_{\Lambda} - \left<[\tau]_{\Lambda}\right>$')
   #
   #     # ax.plot(x, tau_d2, c='k', ls='dashed', lw=2, label=r'$[\tau]^{\partial^{2}}_{\Lambda}$')
   #
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
   #     ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     # ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.legend(fontsize=12, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     # plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
   #
   #     # plt.savefig('../plots/{}/tau_{}.pdf'.format(plots_folder, j), bbox_inches='tight', dpi=300)
   #     # plt.savefig('../plots/{}/tau/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
   #     plt.show()
   #     # plt.close()
   # # # #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, Phi_l, c='k', lw=2, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\Phi_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/Phi/Phi_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #
   #     theta_l = -dv_l*np.sqrt(a)/H0
   #     fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
   #     ax[0].set_title(r'a = {}, $\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind), fontsize=14)
   #     ax[1].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax[0].set_ylabel(r'$f_{l}$', fontsize=14)
   #     ax[0].plot(x, 1+dc_l, c='b', lw=2, label='$1+\delta_{l}$')
   #     ax[0].plot(x, 1+theta_l, ls='dashed', c='k', lw=2, label=r'$1+\theta_{l}$')
   #     ax[1].plot(x, (theta_l - dc_l) * 100 / (1+dc_l), c='k', ls='dashed', lw=2.5)
   #     ax[1].axhline(0, c='b')
   #     for i in range(2):
   #         ax[i].minorticks_on()
   #         ax[i].tick_params(axis='both', which='both', direction='in')
   #         ax[i].yaxis.set_ticks_position('both')
   #
   #         ax[0].legend(fontsize=11)
   #         ax[1].set_ylabel('% diff', fontsize=16)
   #     plt.subplots_adjust(hspace=0)
   #     plt.savefig('../plots/{}/dc_dv/dc_dv_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
   #     plt.close()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}$'.format(a))
   #     ax.plot(x, M0_nbody-1, c='k', lw=2)
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\delta(x)$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M0/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #     # plt.show()

   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}$'.format(a))
   #     ax.plot(x, M0_hier-1, c='k', lw=2)
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\delta(x)$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M0/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #     # plt.show()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}$'.format(a))
   #     ax.plot(x, C1_nbody, c='k', lw=2)
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\bar{v}\;[km\;s^{-1}]$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/C1/C1_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}$'.format(a))
   #     ax.plot(x, M2, c='k', lw=2)
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\mathrm{M}_{2}$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M2/M2_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   # #
       # fig, ax = plt.subplots(2, 2, figsize=(12,10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
       # fig.suptitle(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (sharp)'.format(a, int(Lambda/(2*np.pi))), fontsize=14)
       # ax[0,0].plot(x, tau_l, c='b', lw=2, label=r'$[\tau]_{\Lambda}$')
       # ax[0,0].plot(x, fit, c='k', ls='dashed', lw=2, label=r'fit to $[\tau]_{\Lambda}$')
       # ax[0,0].set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
       # ax[0,0].legend()
       # ax[1,0].set_ylabel(r'$[\tau]_{\Lambda} - \mathrm{fit}$', fontsize=14)
       # ax[1,0].plot(x, tau_l - fit, c='r', lw=2, label=r'residual')
       #
       # ax[0,1].plot(x, C1*dc_l, c='b', lw=2, label=r'$C_{1}\delta_{l}$')
       # ax[0,1].set_ylabel(r'$C_{1}\delta_{l}$', fontsize=14)
       #
       #
       # ax[1,1].plot(x, C2*dv_l, c='b', lw=2, label=r'$C_{2}\partial_{x} v_{l}$')
       # ax[1,1].set_ylabel(r'$C_{2}\partial_{x} v_{l}$', fontsize=14)
       #
       # err_str = 'C0 = {} \nC1 = {} \nC2 = {}'.format(np.round(C0, 5), np.round(C1, 5), np.round(C2, 5))
       # ax[0,1].text(0.775, 15, err_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12)
       #
       # for i in range(2):
       #     for j in range(2):
       #         ax[i,j].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
       #         ax[i,j].minorticks_on()
       #         ax[i,j].tick_params(axis='both', which='both', direction='in')
       #         ax[i,j].yaxis.set_ticks_position('both')
       # # ax.ticklabel_format(scilimits=(-2, 3))
       # # # ax.grid(lw=0.2, ls='dashed', color='grey')
       # # plt.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
       #
       # plt.tight_layout()
       # plt.savefig('../plots/test/tau.png'.format(plots_folder, j), bbox_inches='tight', dpi=300)
       # plt.close()
   # return a, x, ctot2, ctot2_2, ctot2_3
   # return a, x, tau_l, fit, total_err
   # return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, total_err
   # print('a = ', a)
   return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody, d1k, d2k, total_err #, dc_l, dv_l

# path = 'cosmo_sim_1d/phase_full_run1/'
# path = 'cosmo_sim_1d/test_run2/'

mode = 1
Lambda = (2*np.pi) * 3

A = [-0.05, 1, -0.5, 11]
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
# for j in [0, 13, 30, 46]:
#     for Lambda in range(7, 8, 2):
#         Lambda *= (2*np.pi)
        # param_calc(j, Lambda, path, A, mode, kind)

# Nfiles = 41
# for j in range(1):

for j in range(1):
    # path = 'cosmo_sim_1d/sim_k_1_11/run1/'
    path = 'cosmo_sim_1d/sim_k_1/run1/'

    # path = 'cosmo_sim_1d/final_phase_run{}/'.format(l+1)
    j = 0
    sol = param_calc(j, Lambda, path, A, mode, kind)
    # vir = sol[-6]
    # print('a = {}\n'.format(sol[0]))
    # print(vir)

# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'
#
# Lambda_list = np.arange(2, 8)
# nums = [[0, 13], [30, 46]]
# ctot2_0_list, ctot2_1_list, ctot2_2_list = [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]]
# a_list = [[[], []], [[], []]]
#
# for i1 in range(2):
#     print(i1)
#     for i2 in range(2):
#         print(i2)
#         file_num = nums[i1][i2]
#         c0, c1, c2 = [], [], []
#         for i3 in range(Lambda_list.size):
#             Lambda = Lambda_list[i3] * (2*np.pi)
#             print(Lambda)
#             sol = param_calc(file_num, Lambda, path, A, mode, kind)
#             a_list[i1][i2] = sol[0]
#             c0.append(sol[2])
#             c1.append(sol[3])
#             c2.append(sol[4])
#
#         ctot2_0_list[i1][i2] = c0
#         ctot2_1_list[i1][i2] = c1
#         ctot2_2_list[i1][i2] = c2
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
# for i in range(2):
#     print(i)
#     ax[0,0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     ax[1,1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), x=0, fontsize=16)
#     ax[i,1].tick_params(labelleft=False, labelright=True)
#     for j in range(2):
#         print(ctot2_0_list[i][j])
#         ax[i,j].set_title(r'$a = {}$'.format(a_list[i][j]), x=0.25, y=0.9)
#         ax[i,j].plot(Lambda_list, ctot2_2_list[i][j], c='orange', lw=1.5, marker='*', label=r'$B^{+12}$')
#         ax[i,j].plot(Lambda_list, ctot2_1_list[i][j], c='cyan', lw=1.5, marker='v', label=r'M&W')
#         ax[i,j].plot(Lambda_list, ctot2_0_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         ax[i,j].yaxis.set_ticks_position('both')
#     ax[0,0].set_ylim(0.211, 0.242)
#     ax[0,1].set_ylim(0.9, 1.75)
#     # ax[0,0].set_ylim(-0.42, 0.8)
#
#
# plt.legend(fontsize=12, bbox_to_anchor=(1,1))
# fig.align_labels()
# plt.subplots_adjust(hspace=0, wspace=0)
# # plt.show()
# plt.savefig('../plots/test/ctot2/Lam_dep/phase_run_1_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.savefig('../plots/test/ctot2/Lam_dep/phase_run1_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.close()
#

# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# # A = [0.05, 1, -0.5, 11] #amp_inv
# A = [-0.05, 1, -0.5, 11] #regular
#
# Lambda = (2*np.pi) * 6
# kind = 'sharp'
# # kind = 'gaussian'
# # j = 0
# for j in range(15, 21):#0, 51):
#     sol = param_calc(j, Lambda, path, A, mode, kind)
#     a = sol[0]
#     print('a = ', a)


# # mode = 1
# # for j in range(33):
# #     sol = param_calc(j, Lambda, path, A, mode, kind)
# # a = sol[0]
# # x = sol[1]
# # delta_l = sol[2]
# # theta_l = -sol[3]
# #
# # plt.title('a = {}'.format(a))
# # plt.plot(x, delta_l, c='r', lw=2, label=r'$\delta_{l}$')
# # plt.plot(x, theta_l, c='b', lw=2, ls='dashed', label=r'$\theta_{l}$')
# # plt.xlabel('x')
# # plt.legend()
# # plt.savefig('../plots/test/dc_dv.png')
#
# # den_list, num_list, a_list = [], [], []
# # for j in range(33):
# #     sol = param_calc(j, Lambda, path, A, mode, kind)
# #     num_list.append(sol[-2])
# #     den_list.append(sol[-1])
# #     a_list.append(sol[0])
# #
# # plt.xlabel('a')
# # plt.title('M&W, components of ctot2')
# # plt.plot(a_list, num_list, lw=2, c='b', label='num')
# # plt.plot(a_list, den_list, lw=2, c='r', label='den')
# # plt.legend()
# # plt.savefig('../plots/num_ev.png')
#
# # x = sol[1]
# # cs2 = sol[2]
# # print(cs2)
# # plt.plot(x, cs2)
# # print(np.median(cs2))
# # plt.savefig('../plots/test/cs2_test.png')
#
# path = 'cosmo_sim_1d/nbody_new_run6/'
# A = [-0.05, 1, -0.5, 11]#, -0.01, 2, -0.01, 3, -0.01, 4]
# # Lambda = (2*np.pi) * 3
# # kind = 'sharp'
# # kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
# mode = 1
#
# # cs2, cv2, ctot2, mode_list = [], [], [], []
# # file_num = 21
# # for mode in range(1, 6):
# #     sol = param_calc(file_num, Lambda, path, A, mode, kind)
# #     a = sol[0]
# #     cs2.append(sol[2])
# #     cv2.append(sol[3])
# #     ctot2.append(sol[4])
# #     mode_list.append(mode)
# #
# # fig, ax = plt.subplots()
# # ax.plot(mode_list, ctot2, c='k', lw=2, label=r'$', marker='o')
# # ax.set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km\,s}^{-1}]$', fontsize=12)
# # ax.set_xlabel(r'$k\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=12)
# # ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt))
# # ax.minorticks_on()
# # ax.tick_params(axis='both', which='both', direction='in')
# # ax.grid(lw=0.2, ls='dashed', color='grey')
# # ax.yaxis.set_ticks_position('both')
# # plt.savefig('../plots/test/params/baumann_k_dep.png', bbox_inches='tight', dpi=150)
# # plt.close()
#
#
# path = 'cosmo_sim_1d/phase_full_run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# Lambda = (2*np.pi) * 3
# # kind = 'sharp'
# # kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
# sol_0 = param_calc(0, Lambda, path, A, mode, kind)
# sol_1 = param_calc(13, Lambda, path, A, mode, kind)
# sol_2 = param_calc(30, Lambda, path, A, mode, kind)
# sol_3 = param_calc(46, Lambda, path, A, mode, kind)
#
#
# x = sol_0[1]
# a_list = [[sol_0[0], sol_1[0]], [sol_2[0], sol_3[0]]]
# tau_list = [[sol_0[2], sol_1[2]], [sol_2[2], sol_3[2]]]
# fit_list = [[sol_0[3], sol_1[3]], [sol_2[3], sol_3[3]]]
# # C_list = [[sol_0[4], sol_1[4]], [sol_2[4], sol_3[4]]]
# fit_err = [[sol_0[4], sol_1[4]], [sol_2[4], sol_3[4]]]
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
# fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16, x=0.5, y=0.92)
# for i in range(2):
#     ax[i,0].set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#     ax[i,1].set_ylabel(r'$[\tau]_{\Lambda}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#     ax[i,1].yaxis.set_label_position('right')
#     ax[1,i].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
#     ax[i,1].tick_params(labelleft=False, labelright=True)
#     ax[1,i].set_title(r'$a = {}$'.format(a_list[1][i]), x=0.15, y=0.9)
#     ax[0,i].set_title(r'$a = {}$'.format(a_list[0][i]), x=0.15, y=0.9)
#     for j in range(2):
#         ax[i,j].plot(x, tau_list[i][j], c='b', lw=1.5, label=r'$[\tau]_{\Lambda}$')
#         ax[i,j].errorbar(x, fit_list[i][j], yerr=fit_err[i][j], errorevery=10000, c='k', lw=1.5, ls='dashed', label=r'fit to $[\tau]_{\Lambda}$')
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         # err_str = r'$C_{{0}} = {}$'.format(np.round(C_list[i][j][0], 3)) + '\n' + r'$C_{{1}} = {}$'.format(np.round(C_list[i][j][1], 3)) + '\n' + r'$C_{{2}} = {}$'.format(np.round(C_list[i][j][2], 3))
#         # ax[i,j].text(0.35, 0.05, err_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i,j].transAxes)
#         ax[i,j].yaxis.set_ticks_position('both')
#
# plt.legend(fontsize=12, bbox_to_anchor=(0.975,2.2))
# fig.align_labels()
# plt.subplots_adjust(hspace=0, wspace=0)
# # plt.savefig('../plots/test/paper_plots/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.show()
# # plt.close()

# param_calc(0, Lambda, path, A, 1)

# path = 'cosmo_sim_1d/nbody_new_run6/'
# A = [-0.05, 1, -0.5, 11]#, -0.01, 2, -0.01, 3, -0.01, 4]
# # Lambda = (2*np.pi) * 3
# # kind = 'sharp'
# # kind_txt = 'Sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
# mode = 1
# #
# # j = 21
#
# j = 0
# a_list = [0, 0, 0]
# Lambda_list = np.arange(2, 8)
# C0 = np.zeros(shape=(len(a_list), Lambda_list.size))
# C1 = np.zeros(shape=(len(a_list), Lambda_list.size))
# C2 = np.zeros(shape=(len(a_list), Lambda_list.size))
#
# file_nums = [7, 21, 32]
# for j in range(Lambda_list.size):
#     Lambda = Lambda_list[j]
#     print('\nLambda = ', Lambda)
#     Lambda *= (2 * np.pi)
#     for l in range(3):
#         print('file ', l)
#         sol = param_calc(file_nums[l], Lambda, path, A, mode, kind)
#         C = sol[-1]
#         C0[l][j] = C[0]
#         C1[l][j] = C[1]
#         C2[l][j] = C[2]
#         if Lambda == (4*np.pi):
#             a_list[l] = sol[0]
#
# units = [r'$[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', r'$[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', r'$[\mathrm{M}_{10}h\frac{\mathrm{km}}{\mathrm{Mpc}^{2}s}]$']
# fig, ax = plt.subplots(3, 3, figsize=(12, 8), sharex=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1, 1]})
# plt.subplots_adjust(wspace=0.325)
# # fig.suptitle(kind_txt, fontsize=16)
# for j in range(3):
#     ax[0,j].set_title(r'$a = {}$'.format(a_list[j]), fontsize=14, y=1.05)
#     ax[j,0].set_ylabel(r'$C_{}\;$'.format(j) + units[j], fontsize=14)
#     ax[2,0].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), fontsize=14, x=1.88)
#     ax[0,j].plot(Lambda_list, C0[j], marker='o', c='k', lw=1.5)
#     ax[1,j].plot(Lambda_list, C1[j], marker='o', c='k', lw=1.5)
#     ax[2,j].plot(Lambda_list, C2[j], marker='o', c='k', lw=1.5)
#     # ax[j,2].tick_params(labelleft=False, labelright=True)
#
#     for l in range(3):
#         ax[j, l].minorticks_on()
#         ax[j, l].tick_params(axis='both', which='both', direction='in', labelsize=12)
#         ax[j, l].yaxis.set_ticks_position('both')
#
# fig.align_labels()
# # plt.tight_layout()
# plt.savefig('../plots/test/paper_plots/params_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close('all')


# fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# fig.suptitle(r'$a = {}$ ({})'.format(sol[0], kind_txt), fontsize=16, x=0.5, y=0.92)
# for i in range(3):
#     ax[i].set_ylabel(r'$C_{}\;$'.format(i) + units[i], fontsize=16)
#     ax[i].plot(Lambda_list, C_lists[i], marker='o', c='b', lw=1.5)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[2].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=18)
# fig.align_labels()
# plt.savefig('../plots/test/paper_plots/params_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/cutoff_dep_gauss/params_{0:03d}.png'.format(j), bbox_inches='tight', dpi=150)
# plt.close('all')

# # # # d2I_list, a_list = [], []
# for j in range(43, 81, 1):
#     print(j)
#     param_calc(j, 5*(2*np.pi), 'cosmo_sim_1d/nbody_new_run6/')
# #    # d2I_list.append(d2I)
# #    # a_list.append(a)
#
# a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody = param_calc(0, Lambda, path)
#
#
#
# fig, ax = plt.subplots()
# ax.scatter(k, P_nb_a, c='k', lw=2, label=r'$N-$body')
# ax.scatter(k, P_1l_a_sm, c='k', lw=2, label=r'$N-$body')
# ax.scatter(k, P_1l_a_tr, c='k', lw=2, label=r'$N-$body')
# ax.scatter(k, P_1l_a_sm, c='k', lw=2, label=r'$N-$body')
#
# # ax.set_xlabel(r'$a$', fontsize=14)
# # ax.set_ylabel(r'$\mathrm{d}^{2}I(x_{\mathrm{vir}})$', fontsize=14)
# ax.minorticks_on()
# # ax.set_ylim(-100000, 100)
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
# ax.yaxis.set_ticks_position('both')
# plt.savefig('../plots/EFT_nbody/results/virial_edge.png', bbox_inches='tight', dpi=120)
# plt.close()
