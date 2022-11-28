#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, poisson_solver

def EFT_sm_kern(k, Lambda):
   kernel = np.exp(- (k ** 2) / (2 * Lambda**2))
   return kernel #/ sum(kernel)

def smoothing(field, kernel):
   return np.real(np.fft.ifft(np.fft.fft(field) * kernel))

loc = '../'
run = '/sch_hfix_run29/'

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
    rho_b = rho_0 / (a0**3) #this is the physical background density

    sigma_x = np.sqrt(h/2) * 10 #25 * dx
    sigma_p = h / (2 * sigma_x)
    print(sigma_x, sigma_p, h)
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
    poisson_rhs = (3 * H0**2 / (2 * a0)) * (MH_0 - 1)
    MH_0 *= rho_b # m / (a0**3) #this makes MH_0 a physical density ρ, which is the same as defined in Eq. (8) of Hertzberg (2014)

    MH_1_k = np.fft.fft(MW_1) * W_k_an
    MH_1 = np.real(np.fft.ifft(MH_1_k))
    MH_1 *= rho_b / (m*a0) #1 / (a0**4) #this makes MH_0 a velocity density ρv, which the same as π defined in Eq. (9) of Hertzberg (2014)

    MH_2_k = np.fft.fft(MW_2) * W_k_an
    MH_2 = np.real(np.fft.ifft(np.fft.fft(MW_2) * W_k_an)) + ((sigma_p**2) * MH_0)
    MH_2 *= rho_b / (m * a0)**2 #1 / (m * a0**5) #this makes MH_2 into the form ρv^2 + κ, which this the same as σ as defiend in Eq. (10) of Hertzberg (2014)

    # euler : dM1_dt = - (dx_M2 / (m * a**2)) - (m * dx_phi * M0)
    # poisson : d2x_phi = (3 * H0**2 / (2 * a)) * (M0 - 1)
    # thus, dx_phi = integrate((3 * H0**2 / (2 * a)) * (M0 - 1))

    dx_phi = spectral_calc(poisson_rhs, k, o=1, d=1)

    return MH_0, MH_1, MH_2, dx_phi, a0, H0, m, x, k, sigma_p

error, a_list, lhs_list, rhs_list = [], [], [], []

i = 250
l = 5
iml = calc_euler(i-l, loc, run)
MH0_im1, MH1_im1, a0_im1 = iml[0], iml[1], iml[4]
MH_0, MH_1, MH_2, dx_phi, a0, H0, m, x, k, sigma_p = calc_euler(i, loc, run)
ipl = calc_euler(i+l, loc, run)
MH0_ip1, MH1_ip1, a0_ip1 = ipl[0], ipl[1], ipl[4]

print(a0_im1, a0, a0_ip1)

rho_0 = 27.755
rho_b = rho_0 / (a0**3)
Lambda = 6 #must be less than k_NL; we think k_NL < 11
W_EFT = EFT_sm_kern(k, Lambda)

a_dot = H0 * (a0**(-1/2))
H = a_dot / a0

# dM1_da = (MH1_ip1 - MH1_im1) / (a0_ip1 - a0_im1)
# dM1_dt = dM1_da * (a_dot)
#
# dx_M2 = spectral_calc(MH_2, k, o=1, d=0)
# LHS = dM1_dt #+ (dx_M2 / (m * a0**2)) + (m * dx_phi * MH_0)
# RHS = - (dx_M2 / (m * a0**2)) - (m * dx_phi * MH_0)
# zero = LHS - RHS

rho = MH_0
pi = MH_1
sigma = MH_2
v = pi / rho
dx_v = spectral_calc(v, k, o=1, d=0)
kappa = sigma - (pi**2 / rho)

v_ip1 = MH1_ip1 / MH0_ip1
v_im1 = MH1_im1 / MH0_im1

dv_da = (v_ip1 - v_im1) / (a0_ip1 - a0_im1)

dx_kappa = spectral_calc(kappa, k, o=1, d=0)

LHS = (a_dot * dv_da) + (H * v) # #np.zeros(v.size)
RHS =  - (v * dx_v / a0) - (dx_phi / a0) - (dx_kappa / (a0 * rho))

err = (RHS - LHS)
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
plt.grid(linewidth=0.2, color='gray', linestyle='dashed')
ax[0].set_title(r'Both sides of the Euler equation: a = {}'.format(np.round(a0, 3)), fontsize=14)
ax[1].set_xlabel(r'$\mathrm{x}\;[h^{-1}\;\mathrm{Mpc}]$')
ax[1].set_ylabel(r'difference')
ax[0].plot(x, RHS, c='b', label=r'RHS')
# ax[0].plot(x, zero, c='r', label=r'ZERO')
ax[0].plot(x, LHS, c='k', ls='dashed', label='LHS')
ax[0].legend(fontsize=12, loc='upper right')

ax[1].axhline(0, c='brown')
ax[1].plot(x, err, c='green')

print('Saving Nt = {}'.format(i))
plt.savefig('/vol/aibn31/data1/mandar/plots/schr_tests/euler_test_{}.png'.format(i), dpi=120, bbox_inches='tight')
# plt.close()
