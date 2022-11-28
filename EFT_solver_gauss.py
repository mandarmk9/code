#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from functions import spectral_calc, smoothing, EFT_sm_kern, read_density, SPT_sm, SPT_tr
from zel import initial_density
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def EFT_solve(j, Lambda, path, kind):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x = moments_file[:,0]
    M0_nbody = moments_file[:,2]
    M1_nbody = moments_file[:,4]
    M2_nbody = moments_file[:,6]
    C1_nbody = moments_file[:,5]
    M0_hier = M0_nbody

    sm_Lam = 50 * (2*np.pi)
    def den_par(path, j, x):
        dk_par, a, dx = read_density(path, j)
        x0 = 0.0
        xn = 1.0
        x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
        M0_par = np.real(np.fft.ifft(dk_par))
        f_M0 = interp1d(x_grid, M0_par, kind='cubic', fill_value='extrapolate')
        M0_par = f_M0(x)
        M0_par /= np.mean(M0_par)
        return M0_par

    L = x[-1]
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

    d1k, P_1l_a_tr, P_2l_a_tr = SPT_tr(dc_in, k, L, Lambda, kind, a)

    P_lin_a = np.real(d1k * np.conj(d1k)) * (a**2)

    M0_par = den_par(path, j, x)
    M0_nbody = M0_par

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

    dc_par = M0_par - 1 #this is the overdensity δ calculated from particle FT

    rho_par = (1 + dc_par) * rho_b
    rho_par_l = smoothing(rho_par, k, Lambda, kind)
    dc_par_l = smoothing(dc_par, k, Lambda, kind) #(rho_par_l / rho_b) - 1
    v_l = pi_l / rho_l
    dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{l}

    #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
    rhs = (3 * H0**2 / (2 * a)) * dc #using the hierarchy δ here
    phi = spectral_calc(smoothing(rhs, k, sm_Lam, 'gaussian'), L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    rhs_l = (3 * H0**2 / (2 * a)) * dc_l
    phi_l = spectral_calc(rhs_l, L, o=2, d=1)
    grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

    phi_s = phi - phi_l

    dx_phi = spectral_calc(phi, L, o=1, d=0)
    dx_phi_l = spectral_calc(phi_l, L, o=1, d=0)
    dx_phi_s = spectral_calc(phi_s, L, o=1, d=0)

    grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
    grad_phi2_l = smoothing(grad_phi**2, k, Lambda, kind) #this is [(∇ϕ)^2]_l

    #finally, the gravitational part of the smoothed stress tensor
    Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)


    # #here is the full stress tensor; this is the object to be fitted for the EFT paramters
    tau_d2 = (rho_l * (dv_l**2) / Lambda**2) - (dx_phi_l**2 / (3 * H0**2 * a**2 / rho_0))
    tau_l = (kappa_l + Phi_l)

    #PS from the particle FT δ
    dc_l_k = np.fft.fft(dc_par_l) / Nx
    P_nb_a = np.real(dc_l_k * np.conj(dc_l_k))

    return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, kappa_l, Phi_l, M0_nbody, M0_hier, v_l, M2_nbody, C1_nbody

def param_calc(j, Lambda, path, mode, kind):
   a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, kappa_l, Phi_l, M0_nbody, M0_hier, v_l, M2, C1_nbody = EFT_solve(j, Lambda, path, kind)
   rho_0 = 27.755
   rho_b = rho_0 / a**3
   H0 = 100

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
   C = [C0, C1, C2]

   cs2 = np.real(C1 / rho_b)
   cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
   ctot2 = (cs2 + cv2)

   # M&W Estimator
   Lambda_int = int(Lambda / (2*np.pi))
   num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))[:Lambda_int]
   denom = P_lin_a[:Lambda_int]
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

   plots_folder = 'nbody_gauss_run4/' #/paper_plots'
   if True:
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
   #     plt.savefig('../plots/{}/kappa/kappa_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()

       fig, ax = plt.subplots()
       ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=12)
       ax.plot(x, tau_l, c='b', lw=2, label=r'$[\tau]_{\Lambda}$')
       ax.plot(x, fit, c='k', ls='dashed', lw=2, label=r'$\left<[\tau]_{\Lambda}\right>$ (fit)')
       ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
       ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
       ax.minorticks_on()
       ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
       ax.ticklabel_format(scilimits=(-2, 3))
       # ax.grid(lw=0.2, ls='dashed', color='grey')
       ax.legend(fontsize=12, bbox_to_anchor=(1,1))
       ax.yaxis.set_ticks_position('both')
       # plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)

       plt.savefig('../plots/{}/tau_sharp_L3/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
       # plt.savefig('../plots/{}/tau/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
       # plt.savefig('../plots/{}/tau_{}.pdf'.format(plots_folder, j), bbox_inches='tight', dpi=300)
       plt.close()
       # plt.show()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, Phi_l, c='k', lw=2, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\Phi_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/Phi/Phi_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, dc_l, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\delta_{l}$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M0_l/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, M0_nbody-1, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\delta$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M0/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #     # plt.show()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, M0_hier-1, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\delta$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.set_xlim(0, 0.01)
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/hier_M0/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #     # plt.show()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, C1_nbody, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\bar{v}\;[km\;s^{-1}]$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/C1/C1_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()
   #
   #     fig, ax = plt.subplots()
   #     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))))
   #     ax.plot(x, M2, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #     ax.set_ylabel(r'$\mathrm{M}_{2}$', fontsize=14)
   #     ax.minorticks_on()
   #     ax.tick_params(axis='both', which='both', direction='in')
   #     ax.ticklabel_format(scilimits=(-2, 3))
   #     ax.grid(lw=0.2, ls='dashed', color='grey')
   #     # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #     ax.yaxis.set_ticks_position('both')
   #     plt.savefig('../plots/{}/M2/M2_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #     plt.close()

   # return a, x, ctot2, ctot2_2, ctot2_3
   return a, x, k, P_nb_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2, M0_nbody

path = 'cosmo_sim_1d/nbody_gauss_run4/'
Lambda = 3 * (2 * np.pi)
mode = 1
# kind = 'gaussian'
kind = 'sharp'
Nfiles = 51
# a_list = np.zeros(Nfiles)
# P_nb = np.zeros(Nfiles)
# P_1l = np.zeros(Nfiles)

for j in range(Nfiles):
    sol = param_calc(j, Lambda, path, mode, kind)
    a = sol[0]
    print(a)
    # a_list[j] = a
    # P_nb[j] = sol[3][mode]
    # P_1l[j] = sol[7][mode]

# plt.plot(a_list, P_nb, lw=2, c='b')
# plt.plot(a_list, P_1l, lw=2, c='k', ls='dashed')
# plt.savefig('../plots/nbody_gauss_run2/test_sepc.png', bbox_inches='tight', dpi=120)

# path = 'cosmo_sim_1d/nbody_gauss_run4/'
# mode = 1
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'
#
# Lambda_list = np.arange(2, 8, 1)
# nums = [[0, 7], [21, 32]]
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
#             sol = param_calc(file_num, Lambda, path, mode, kind)
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
#     # ax[0,0].set_ylabel(r'$c^{2}_{v}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     # ax[0,0].set_ylabel(r'$c^{2}_{s}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     ax[0,0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     ax[1,1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), x=0, fontsize=16)
#     ax[i,1].tick_params(labelleft=False, labelright=True)
#     for j in range(2):
#         print(ctot2_0_list[i][j])
#         ax[i,j].set_title(r'$a = {}$'.format(a_list[i][j]), x=0.25, y=0.9)
#         ax[i,j].plot(Lambda_list, ctot2_0_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
#         ax[i,j].plot(Lambda_list, ctot2_1_list[i][j], c='cyan', lw=1.5, marker='+', label=r'M&W')
#         # ax[i,j].plot(Lambda_list, ctot2_2_list[i][j], c='orange', lw=1.5, marker='*', label=r'$B^{+12}$')
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         ax[i,j].yaxis.set_ticks_position('both')
#     # ax[0,0].set_ylim(0.211, 0.24)
#     # ax[0,1].set_ylim(0.75, 0.86)
#
# plt.legend(fontsize=12, bbox_to_anchor=(1,1))
# fig.align_labels()
# plt.subplots_adjust(hspace=0, wspace=0)
# plt.savefig('../plots/nbody_gauss_run4/ctot2_param_dep_{}.png'.format(kind), bbox_inches='tight', dpi=120)
