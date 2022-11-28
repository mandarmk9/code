#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, poisson_solver
from EFT_solver import EFT_solve

def EFT_sm_kern(k, Lambda):
   kernel = np.exp(- (k ** 2) / (2 * Lambda**2))
   return kernel #/ sum(kernel)

def smoothing(field, kernel):
   return np.real(np.fft.ifft(np.fft.fft(field) * kernel))

loc = '../'
run = '/sch_hfix_run19/'

def calc_euler(i, loc, run):
    with h5py.File(loc + 'data' + run + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
        ls = list(hdf.keys())
        A = np.array(hdf.get(str(ls[0])))
        a0 = np.array(hdf.get(str(ls[1])))
        L, h, m, H0 = np.array(hdf.get(str(ls[2])))
        psi = np.array(hdf.get(str(ls[3])))

    Nx = psi.size
    dx = L / Nx
    x = np.arange(0, L, dx)
    k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
    rho_0 = 27.755 #this is the comoving background density
    rho_b = m / (a0**3) #this is the physical background density
    G = (3 * H0**2) / (8 * np.pi * rho_0)

    sigma_x = np.sqrt(h/2) * 10 #25 * dx
    sigma_p = h / (2 * sigma_x)
    # print(sigma_x, sigma_p, h)
    sm = 1 / (4 * (sigma_x**2))
    W_k_an = np.exp(- (k ** 2) / (4 * sm))

    dc_in = (A[0] * np.cos(A[1]*x)) + (A[2] * np.cos(A[3]*x))

    psi_star = np.conj(psi)
    grad_psi = spectral_calc(psi, k, o=1, d=0)
    grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
    lap_psi = spectral_calc(psi, k, o=2, d=0)
    lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

    #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
    MW_0 = np.abs(psi ** 2)
    MW_1 = ((1j * h / 1) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
    MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))
    MH_0_k = np.fft.fft(MW_0) * W_k_an
    MH_0 = np.real(np.fft.ifft(MH_0_k))
    MH_0 *= rho_b #m / (a0**3) #this makes MH_0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)

    MH_1_k = np.fft.fft(MW_1) * W_k_an
    MH_1 = np.real(np.fft.ifft(MH_1_k))
    MH_1 *= rho_b / (m * a0) #1 / (a0**4) #this makes MH_0 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)

    MH_2_k = np.fft.fft(MW_2) * W_k_an
    MH_2 = np.real(np.fft.ifft(np.fft.fft(MW_2) * W_k_an)) + ((sigma_p**2) * MH_0)
    MH_2 *= rho_b / (m * a0)**2 #1 / (m * a0**5) #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    # euler : dM1_dt = - (dx_M2 / (m * a**2)) - (m * dx_phi * M0)
    # poisson : d2x_phi = (3 * H0**2 / (2 * a)) * (M0 - 1)
    # thus, dx_phi = integrate((3 * H0**2 / (2 * a)) * (M0 - 1))

    # poisson_rhs = 4 * np.pi * G * a0**2 * (MH_0 - (rho_0 / a0**3))
    poisson_rhs = (3 * H0**2 / (2 * a0)) * ((MH_0 / rho_b) - 1)
    dx_phi = spectral_calc(poisson_rhs, k, o=1, d=1)
    C0, C1, C2 = EFT_solve(i, 5, loc, run, EFT=1)[-3:]
    return MH_0, MH_1, MH_2, dx_phi, a0, H0, m, x, k, sigma_p, C0, C1, C2

error, a_list, lhs_list, rhs_list = [], [], [], []

i = 300
l = 5
iml = calc_euler(i-l, loc, run)
MH0_im1, MH1_im1, a0_im1 = iml[0], iml[1], iml[4]
MH_0, MH_1, MH_2, dx_phi, a0, H0, m, x, k, sigma_p, C0, C1, C2 = calc_euler(i, loc, run)
ipl = calc_euler(i+l, loc, run)
MH0_ip1, MH1_ip1, a0_ip1 = ipl[0], ipl[1], ipl[4]

# print(a0_im1, a0, a0_ip1)

rho_0 = 27.755
rho_b = rho_0 / (a0**3)
Lambda = 5 #must be less than k_NL; we think k_NL < 11
W_EFT = EFT_sm_kern(k, Lambda)

a_dot = H0 * (a0**(-1/2))
H = a_dot / a0
G = (3 * H0**2) / (8 * np.pi * rho_0)

# dM1_da = (MH1_ip1 - MH1_im1) / (a0_ip1 - a0_im1)
# dM1_dt = dM1_da * (a_dot)
#
# dx_M2 = spectral_calc(MH_2, k, o=1, d=0)
# LHS = dM1_dt #+ (dx_M2 / (m * a0**2)) + (m * dx_phi * MH_0)
# RHS = - (dx_M2 / (m * a0**2)) - (m * dx_phi * MH_0)
# zero = LHS - RHS

rho = MH_0
rho_l = smoothing(MH_0, W_EFT)
dc = MH_0 / (m / (a0**3))
dc_l = smoothing(dc, W_EFT)
# poisson_rhs_l = 4 * np.pi * G * a0**2 * (rho_l - (rho_0 / a0**3))
poisson_rhs_l = (3 * H0**2 / (2 * a0)) * (dc_l)

pi_l = smoothing(MH_1, W_EFT)
v_l = pi_l / rho_l
dx_v_l = spectral_calc(v_l, k, o=1, d=0)
sigma_l = smoothing(MH_2, W_EFT)
kappa_l = sigma_l - (pi_l**2 / rho_l)
dx_kappa_l = spectral_calc(kappa_l, k, o=1, d=0)

MH0_ipl1 = smoothing(MH0_ip1, W_EFT)
MH0_iml1 = smoothing(MH0_im1, W_EFT)
MH1_ipl1 = smoothing(MH1_ip1, W_EFT)
MH1_iml1 = smoothing(MH1_im1, W_EFT)

v_ipl1 = MH1_ipl1 / MH0_ipl1
v_iml1 = MH1_iml1 / MH0_iml1

dvl_da = (v_ipl1 - v_iml1) / (a0_ip1 - a0_im1)

dx_phi_l = spectral_calc(poisson_rhs_l, k, o=1, d=1)

dx_phi_l2 = dx_phi_l**2 #this is [(∇(ϕ_{l})]**2
dx_phi2_l = smoothing(dx_phi**2, W_EFT) #this is [(∇ϕ)^2]_l

#finally, the gravitational part of the smoothed stress tensor
# Phi_l = (rho_0 / (3 * (H0**2) * (a0**2))) * (dx_phi2_l - dx_phi_l2) #is this rho_0 or rho_b (depends on the definition of critical density)
Phi_l = (rho_0 / (3 * (H0**2) * (a0**2))) * (dx_phi2_l - dx_phi_l2) #is this rho_0 or rho_b (depends on the definition of critical density)

tau_l = kappa_l + Phi_l
# tau_l = C0 + (C1 * dc_l) + (C2 * dx_v_l)
dx_tau_l = spectral_calc(tau_l, k, o=1, d=0)

LHS = (a_dot * dvl_da) + (H * v_l)
# RHS =  - (v_l * dx_v_l / a0) - (dx_phi_l / a0) - (dx_tau_l / (a0 * rho_l))
# RHS = - (v_l * dx_v_l / a0) - (dx_phi_l / a0) - (dx_tau_l / (a0 * rho_l))
RHS = - (v_l * dx_v_l / a0) - (dx_phi_l / a0) - (dx_tau_l / (a0 * rho_l))

err = (RHS - LHS)
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
plt.grid(linewidth=0.2, color='gray', linestyle='dashed')
ax[0].set_title(r'Both sides of the Euler equation: a = {}, $\Lambda = {}$'.format(np.round(a0, 3), Lambda), fontsize=14)
ax[1].set_xlabel(r'$\mathrm{x}\;[h^{-1}\;\mathrm{Mpc}]$')
ax[1].set_ylabel(r'difference')
ax[0].plot(x, tau_l, c='b', label=r'RHS')
# ax[0].plot(x, zero, c='r', label=r'ZERO')
# ax[0].plot(x, LHS, c='k', ls='dashed', label='LHS')
ax[0].legend(fontsize=12, loc='upper right')

ax[1].axhline(0, c='brown')
ax[1].plot(x, err, c='green')

print('Saving Nt = {}'.format(i))
plt.savefig('/vol/aibn31/data1/mandar/plots/' + run + '/euler_test_{}.png'.format(i), dpi=120, bbox_inches='tight')
# plt.close()
