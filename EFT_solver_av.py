#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from functions import spectral_calc, EFT_sm_kern, smoothing, SPT_sm, SPT_tr, plotter

def EFT_solve(j, Lambda, path):
   with h5py.File(path + '/psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))
      # print(A, psi.size, h)

   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
   rho_0 = 27.755 #this is the comoving background density
   rho_b = rho_0 / (a**3) #this is the physical background density

   sigma_x = 0.1 * np.sqrt(h / 2) #10 * dx
   sigma_p = h / (2 * sigma_x)
   sm = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sm))

   dc_in = (A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))

   W_EFT = EFT_sm_kern(k, Lambda)

   d1k, P_1l_a_sm, P_2l_a_sm = SPT_sm(dc_in, k, L, W_EFT, a)
   d1k, P_1l_a_tr, P_2l_a_tr = SPT_tr(dc_in, k, L, W_EFT, a)

   P_lin_a = np.real(d1k * np.conj(d1k)) * (a**2)

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, L, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), L, o=1, d=0)
   lap_psi = spectral_calc(psi, L, o=2, d=0)
   lap_psi_star = spectral_calc(np.conj(psi), L, o=2, d=0)

   #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
   MW_0 = np.abs(psi ** 2)
   MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
   MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))

   MH_0_k = np.fft.fft(MW_0) * W_k_an
   MH_0 = np.real(np.fft.ifft(MH_0_k))
   MH_0 *= rho_b #this makes MH_0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)

   MH_1_k = np.fft.fft(MW_1) * W_k_an
   MH_1 = np.real(np.fft.ifft(MH_1_k))
   MH_1 *= rho_b / a #this makes MH_0 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)

   MH_2_k = np.fft.fft(MW_2) * W_k_an
   MH_2 = np.real(np.fft.ifft(np.fft.fft(MW_2) * W_k_an)) + ((sigma_p**2) * MH_0)
   MH_2 *= rho_b / a**2 #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

   #now all long-wavelength moments
   M0_l = np.real(np.fft.ifft(np.fft.fft(MH_0) * W_EFT)) #this is ρ_{l}
   M1_l = np.real(np.fft.ifft(np.fft.fft(MH_1) * W_EFT)) #this is π_{l}
   M2_l = np.real(np.fft.ifft(np.fft.fft(MH_2) * W_EFT)) #this is σ_{l}

   #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
   #in 1D, κ_{l} = σ_{l} - ([π_{l}]^{2} / ρ_{l})
   kappa_l = (M2_l - (M1_l**2 / M0_l))

   dc = np.real(np.fft.ifft(np.fft.fft(MW_0 - 1) * W_k_an)) #this is the overdensity δ
   rho = (1 + dc) * rho_0 / (a **3)
   rho_l = np.real(np.fft.ifft(np.fft.fft(rho) * W_EFT)) #this is ρ_l
   dc_l = (rho_l / (rho_0 / (a **3))) - 1
   v_l = M1_l / M0_l
   dv_l = spectral_calc(v_l, L, o=1, d=0) #the derivative of v_{l}

   dc_l_k = np.fft.fft(dc_l) / Nx
   P_sch_a = np.real(dc_l_k * np.conj(dc_l_k))

   #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
   rhs = (3 * H0**2 / (2 * a)) * dc
   phi = spectral_calc(rhs, L, o=2, d=1)
   grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

   rhs_l = (3 * H0**2 / (2 * a)) * dc_l
   phi_l = spectral_calc(rhs_l, L, o=2, d=1)
   grad_phi_l = spectral_calc(phi_l, L, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

   grad_phi2_l = np.real(np.fft.ifft(np.fft.fft(grad_phi**2) * W_EFT)) #this is [(∇ϕ)^2]_l
   grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
   #finally, the gravitational part of the smoothed stress tensor
   Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi_l2 - grad_phi2_l)

   #here is the full stress tensor; this is the object to be fitted for the EFT paramters
   tau_l = (kappa_l + Phi_l)
   CH_1 =  MH_1 / MH_0

   # ind1, ind2 = 1200000, 1300000
   # T = (MH_2 - (sigma_p**2 * MH_0)) / (a**2)
   # nd = MW_0-1
   # V = x * smoothing(nd * spectral_calc(nd, L, o=1, d=1), W_k_an)
   #
   # T = sum(T[ind1:ind2])
   # V = sum(V[ind1:ind2])
   # vir = T / V

   return a, x, k, P_sch_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, kappa_l, Phi_l, dc, CH_1, MH_2#, vir

def param_calc(j, Lambda, path):
   a, x, k, P_sch_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, dc_l, dv_l, d1k, kappa_l, Phi_l, dc, CH_1, MH_2 = EFT_solve(j, Lambda, path)
   rho_0 = 27.755
   rho_b = rho_0 / a**3
   H0 = 100
   err_nb = np.abs(P_sch_a[1] - P_lin_a[1]) * 100 / P_lin_a[1]
   err_1l = np.abs(P_1l_a_tr[1] - P_lin_a[1]) * 100 / P_lin_a[1]

   # print("N-body abs err: {}%".format(np.round(err_nb, 4)))
   # print("1-loop SPT abs err: {}% \n".format(np.round(err_1l, 4)))

   # for 3 parameters a0, a1, a2 such that τ_l = a0 + a1 × (δ_l) + a2 × dv_l
   def fitting_function(X, a0, a1, a2):
      x1, x2 = X
      return a0 + a1*x1 + a2*x2

   guesses = 1, 1, 1
   FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=1e-15*np.ones(x.size), method='lm')
   C0, C1, C2 = FF[0]
   # print(C0, C1, C2)
   cov = FF[1]
   err0, err1, err2 = np.sqrt(np.diag(cov))
   fit = fitting_function((dc_l, dv_l), C0, C1, C2)

   # #Hertzberg's approach
   cs2 = np.real(C1 / rho_b)
   cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
   ctot2 = (cs2 + cv2)

   ctot2_2 = np.real(sum((np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))) / sum(P_lin_a)) / rho_b

   # def Power(f1_k, f2_k):
   #    # f1_k = np.fft.fft(f1) / f1.size
   #    # f2_k = np.fft.fft(f2) / f2.size
   #    corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
   #    return corr
   #
   # A = np.fft.fft(tau_l) / rho_b / tau_l.size
   # T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
   # d = np.fft.fft(dc_l) / dc_l.size
   # Ad = Power(A, dc_l)[1]
   # AT = Power(A, T)[1]
   # P_dd = Power(dc_l, dc_l)[1]
   # P_TT = Power(T, T)[1]
   # P_dT = Power(dc_l, T)[1]

   # # print(P_dd, P_TT, P_dT)
   # # print(P_dd * P_TT - (P_dT)**2)
   # cs2 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
   # cv2 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)

   ctot2_3 = 0#np.real(cs2 + cv2)

   # # print('Fit: ', ctot2)
   # # print('M&W: ', ctot2_2)
   # # print('Baumann: ', ctot2_3, '\n')
   # A *= tau_l.size
   # stoch = np.real(np.fft.ifft(Power(A, A)))


   # plots_folder = 'sch_new_run5_L3'
   #
   # if j%5 == 0:
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}, \Lambda = {} \;[h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda)))
   #  ax.plot(x, kappa_l, c='b', lw=2, label=r'$\kappa_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\kappa_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/kappa/kappa_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  # stoch = -spectral_calc(tau_l, 1.0, o=2, d=0) / 5000#-(stoch - np.mean(stoch)) / 20000
   #  # correction = tau_l - fit
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}, \Lambda = {}\;[h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, Lambda))
   #  ax.plot(x, tau_l, c='k', lw=2, label=r'$\tau_{l}$')
   #  ax.plot(x, fit, c='b', ls='dashdot', lw=2, label=r'fit to $\tau_{l}$')
   #  # ax.plot(x, stoch, c='r', ls='dashed', lw=2, label=r'stochastic part')
   #  # ax.plot(x, correction, c='b', ls='dotted', lw=2, label=r'correction')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\tau_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/tau/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}, \Lambda = {}\;[h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, Lambda))
   #  ax.plot(x, Phi_l, c='k', lw=2, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\Phi_{l}\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/Phi/Phi_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}, \Lambda = {}\;[h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, Lambda))
   #  ax.plot(x, dc_l, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\delta_{l}$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/M0_l/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}$'.format(a))
   #  ax.plot(x, dc, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\delta(x)$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/M0/M0_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}$'.format(a))
   #  ax.plot(x, CH_1, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\bar{v}[km\;s^{-1}]$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/C1/C1_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()
   #
   #  fig, ax = plt.subplots()
   #  ax.set_title(r'$a = {}$'.format(a))
   #  ax.plot(x, MH_2, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
   #  ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
   #  ax.set_ylabel(r'$\mathrm{M}_{2}$', fontsize=14)
   #  ax.minorticks_on()
   #  ax.tick_params(axis='both', which='both', direction='in')
   #  ax.ticklabel_format(scilimits=(-2, 3))
   #  ax.grid(lw=0.2, ls='dashed', color='grey')
   #  # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #  ax.yaxis.set_ticks_position('both')
   #  plt.savefig('../plots/{}/M2/M2_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=120)
   #  plt.close()

   return a, x, k, P_sch_a, P_lin_a, P_1l_a_sm, P_2l_a_sm, P_1l_a_tr, P_2l_a_tr, tau_l, fit, ctot2, ctot2_2, ctot2_3, cs2, cv2#, vir

# vir_list, a_list = [], []
# for j in range(1, 400, 10):
#     sol = param_calc(j, 5, '../data/new_run2/')
#     a = sol[0]
#     vir = np.mean(sol[-1])
#     print(a, vir)
#     vir_list.append(np.real(vir))
#     a_list.append(a)
#
# fig, ax = plt.subplots()
# ax.plot(a_list, vir_list, c='k', lw=2)#, label=r'$\Phi^{\mathrm{EFT}}_{l}$')
# ax.set_xlabel(r'$a$', fontsize=14)
# ax.set_ylabel(r'$\alpha$', fontsize=14)
# ax.minorticks_on()
# # ax.set_ylim(-100000, 100)
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
# ax.yaxis.set_ticks_position('both')
# plt.savefig('../plots/sch_new_run2_L5/virial_central.png', bbox_inches='tight', dpi=120)
# plt.show()
# # plt.close()

# for j in range(0, 5, 4):
#    param_calc(j, 5, '../data/new_run1')
