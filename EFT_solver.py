#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, dn, poisson_solver, Psi_q_finder
from SPT import SPT_final

def EFT_sm_kern(k, Lambda):
   kernel = np.exp(- (k ** 2) / (2 * Lambda**2))
   return kernel / sum(kernel)

def smoothing(field, kernel):
   return np.real(np.fft.ifft(np.fft.fft(field) * kernel))

def EFT_solve(j, Lambda, loc, run, EFT=0, zeld=0, num='_run1/'):
   num = ''
   with h5py.File(loc + 'data' + run + num + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))

   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
   rho_0 = 27.755 #this is the comoving background density
   rho_b = rho_0 / (a**3) #this is the physical background density

   sigma_x = np.sqrt(h / 2) #10 * dx
   sigma_p = h / (2 * sigma_x)
   sm = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sm))

   dc_in = (A[0] * np.cos(2 * np.pi * x * A[1] / L)) + (A[2] * np.cos(2 * np.pi * x * A[3] / L))

   W_EFT = EFT_sm_kern(k, Lambda)
   dc_in_bar = dc_in #smoothing(dc_in, W_k_an)
   # dc_in_bar = smoothing(dc_in, W_EFT)

   Psi_q = -Psi_q_finder(x, A, L)
   v_zel = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity

   n = 3 #overdensity order of the SPT
   F = dn(n, k, dc_in_bar)

   d1k = (np.fft.fft(F[0]) / Nx) * W_EFT
   d2k = (np.fft.fft(F[1]) / Nx) * W_EFT
   d3k = (np.fft.fft(F[2]) / Nx) * W_EFT

   order_2 = (d1k * np.conj(d1k)) * (a**2)
   order_3 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
   order_13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
   order_22 = (d2k * np.conj(d2k)) * (a**4)
   order_4 = order_22 + order_13
   order_5 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5) #+ (d1k * np.conj(d4k)) + ((d4k * np.conj(d1k))))
   order_6 = (d3k * np.conj(d3k)) * (a**6)

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, k, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
   lap_psi = spectral_calc(psi, k, o=2, d=0)
   lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

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

   # if j%25 == 0:
   #    plt.title('a = {}'.format(a))
   #    plt.plot(x, v_zel, lw=2, ls='dashed')
   #    plt.plot(x, MH_1 / MH_0, lw=2)
   #    plt.savefig('/vol/aibn31/data1/mandar/plots/sch_hfix_mix/with_rho_0/M1/M1_{}.png'.format(j))
   #    plt.close()

   #now all long-wavelength moments
   M0_l = np.real(np.fft.ifft(np.fft.fft(MH_0) * W_EFT)) #this is ρ_{l}
   M1_l = np.real(np.fft.ifft(np.fft.fft(MH_1) * W_EFT)) #this is π_{l}
   M2_l = np.real(np.fft.ifft(np.fft.fft(MH_2) * W_EFT)) #this is σ_{l}

   #now we calculate the kinetic part of the (smoothed) stress tensor in EFT (they call it κ_ij)
   #in 1D, κ_{l} = σ_{l} - ([π_{l}]^{2} / ρ_{l})
   kappa_l = (M2_l - (M1_l**2 / M0_l)) #/ M0_l

   dc = np.real(np.fft.ifft(np.fft.fft(MW_0 - 1) * W_k_an)) #this is the overdensity δ

   rho = (1 + dc) * rho_0 / (a **3)
   rho_l = np.real(np.fft.ifft(np.fft.fft(rho) * W_EFT)) #this is ρ_l
   dc_l = (rho_l / (rho_0 / (a **3))) - 1
   v_l = M1_l / M0_l
   dv_l = spectral_calc(v_l, k, o=1, d=0) #the derivative of v_{l}

   dc_l_k = np.fft.fft(dc_l) / Nx
   dk2_sch = dc_l_k * np.conj(dc_l_k)

   #next, we build the gravitational part of the smoothed stress tensor (this is a consequence of the smoothing)
   rhs = (3 * H0**2 / (2 * a)) * dc
   phi = poisson_solver(rhs, k)
   grad_phi = spectral_calc(phi, k, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

   rhs_l = (3 * H0**2 / (2 * a)) * dc_l
   phi_l = poisson_solver(rhs_l, k)
   grad_phi_l = spectral_calc(phi_l, k, o=1, d=0) #this is the gradient of the smoothed potential ∇(ϕ_l)

   grad_phi2_l = np.real(np.fft.ifft(np.fft.fft(grad_phi**2) * W_EFT)) #this is [(∇ϕ)^2]_l
   grad_phi_l2 = grad_phi_l**2 #this is [(∇(ϕ_{l})]**2
   #finally, the gravitational part of the smoothed stress tensor
   Phi_l = (rho_0 / (3 * (H0**2) * (a**2))) * (grad_phi2_l - grad_phi_l2) #is this rho_0 or rho_b (depends on the definition of critical density)

   #here is the full stress tensor; this is the object to be fitted for the EFT paramters
   tau_l = (kappa_l - Phi_l)
   d3_spt = (a * d1k) + ((a**2) * d2k) + ((a**3) * d3k)
   dk2_spt = d3_spt * np.conj(d3_spt)
   return a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, tau_l, dk2_sch, dk2_spt, dc_l, dv_l, d1k

def param_calc(j, Lambda, loc, run):
   a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, tau_l, dk2_sch, dk2_spt, dc_l, dv_l, d1k = EFT_solve(j, Lambda, loc, run)

   #for 3 parameters a0, a1, a2 such that τ_l = a0 + a1 × (δ_l) + a2 × dv_l
   from scipy.optimize import curve_fit
   def fitting_function(X, a0, a1, a2):
      x1, x2 = X
      return a0 + a1*x1 + a2*x2

   # if a < 4:
   #    guesses = 1e4, 1e4, 1
   # else:
   guesses = 1, 1, 1e-2

   FF = curve_fit(fitting_function, (dc_l, dv_l), tau_l, guesses, sigma=1e-20*np.ones(x.size), method='lm')
   C0, C1, C2 = FF[0]
   print(C0, C1, C2)
   cov = FF[1]
   err0, err1, err2 = np.sqrt(np.diag(cov))
   fit = fitting_function((dc_l, dv_l), C0, C1, C2)

   rho_0 = 27.755
   H0 = 100
   ##Hertzberg's approach
   cs2 = C1 * (a**3) / rho_0
   cv2 = -C2 * H0 * (a**(5/2)) / rho_0
   ctot2 = (cs2 + cv2)
   ctot2_2 = np.real(sum((a * np.conj(d1k) * ((np.fft.fft(tau_l)) / x.size)) / x.size) / sum(order_2 / x.size)) * (a**3) / rho_0

   # if j%25 == 0:
   #    fig, ax = plt.subplots()
   #    ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
   #    # ax.plot(x, tau_l1, c='r', lw=2, label=r'$\phi = 0$')
   #    # ax.plot(x, tau_l2, c='y', lw=2, label=r'$\phi = \pi/2$')
   #    # ax.plot(x, tau_l3, c='b', lw=2, label=r'$\phi = \pi$')
   #    # ax.plot(x, tau_l4, c='brown', lw=2, label=r'$\phi = 3\pi/2$')
   #    ax.plot(x, fit, c='b', lw=2, label='fit')
   #    ax.plot(x, tau_l, c='k', lw=2, ls='dashed', label='mean')
   #    ax.set_xlabel('x', fontsize=14)
   #    ax.set_ylabel(r'$\tau_{l}$', fontsize=14)
   #    ax.legend()
   #    plt.savefig('../plots/sch_hfix_mix/with_rho_0/tau_fit/M2_l{}.png'.format(j), bbox_inches='tight', dpi=120)
   #    plt.close()
   #
   #    fig, ax = plt.subplots()
   #    ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
   #    ax.plot(x, dc_l, c='b', lw=2)
   #    ax.set_xlabel('x', fontsize=14)
   #    ax.set_ylabel(r'$\delta_{l}$', fontsize=14)
   #    plt.savefig('../plots/sch_hfix_mix/with_rho_0/dl/d_l_{}.png'.format(j), bbox_inches='tight', dpi=120)
   #    plt.close()
   #
   #    fig, ax = plt.subplots()
   #    ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
   #    ax.plot(x, dv_l, c='b', lw=2)
   #    ax.set_xlabel('x', fontsize=14)
   #    ax.set_ylabel(r'$dv_{l}$', fontsize=14)
   #    plt.savefig('../plots/sch_hfix_mix/with_rho_0/dvl/dv_l{}.png'.format(j), bbox_inches='tight', dpi=120)
   #    plt.close()

   return a, x, k, dk2_sch, order_2, order_3, order_4, order_5, order_6, tau_l, fit, dk2_sch, dk2_spt, ctot2, ctot2_2, cs2, cv2, dc_l, dv_l
