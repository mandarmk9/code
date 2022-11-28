#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import *
# plt.style.use('simple_plt')

run = '/mz_run14/'
loc2 = '/vol/aibn31/data1/mandar/'

# M = 2000
for Nt in range(39):
    dd_dt = []
    a_val = []
    for i in range(Nt, Nt+2):
        with h5py.File(loc2 + 'data' + run + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
            ls = list(hdf.keys())
            A = np.array(hdf.get(str(ls[0])))
            a0 = np.array(hdf.get(str(ls[1])))
            L, h, m, H0 = np.array(hdf.get(str(ls[2])))
            psi = np.array(hdf.get(str(ls[3])))

        Nx = psi.size
        dx = L / Nx

        x = np.arange(0, L, dx)
        k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
        sigma_x = 25 * dx
        sigma_k = 1 / (4 * (sigma_x**2))
        W_k_an = np.exp(- (k ** 2) / (4 * sigma_k))

        psi_star = np.conj(psi)
        grad_psi = spectral_calc(psi, k, o=1, d=0)
        grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)

        lap_psi = spectral_calc(psi, k, o=2, d=0)
        lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

        MW_0 = np.abs(psi ** 2)
        MH_0_k = np.fft.fft(MW_0) * W_k_an
        MH_0 = np.real(np.fft.ifft(MH_0_k))

        MW_00 = (MW_0 - np.mean(MW_0)) / np.mean(MW_0)
        MW_1 = (1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)) #this is a momentum density, divide by MW_0, a0, m to get peculiar velocity

        MH_1_k = np.fft.fft(MW_1) * W_k_an
        MH_1 = np.real(np.fft.ifft(MH_1_k))

        v_pec = MH_1 / MH_0

        Lambda = 6 #must be less than k_NL; we think k_NL < 11
        EFT_sm_kern = np.exp(- (k ** 2) / (2 * Lambda**2))
        MH_0_l = MH_0#np.real(np.fft.ifft(np.fft.fft(MH_0) * EFT_sm_kern))
        MH_1_l = MH_1#np.real(np.fft.ifft(np.fft.fft(MH_1) * EFT_sm_kern))

        # dl_vl = (1 + d_l) * v_l
        dv_l = -spectral_calc(MH_1_l, k, d=0, o=1) / (m * (a0**(3/2)) * H0) #this is the rhs of the continuity eq

        dd_dt.append(MH_0_l)
        a_val.append(a0)

    sch_vel = -1j * h * np.log(psi / (np.sqrt(np.abs(psi**2))))
    zel_vel = np.fft.ifft(np.fft.fft(((A[0] / (A[1]**2)) * np.cos(A[1] * x) + (A[2] / (A[3]**2)) * np.cos(A[3] * x))) * W_k_an)

    dd_dt = np.array(dd_dt)
    dd_dt_num = (dd_dt[1] - dd_dt[0]) / (a_val[1] - a_val[0])
    dv_dx = dv_l
    # dv_dx = a0**2
    fig1, ax1 = plt.subplots()
    plt.grid(linewidth=0.2, color='gray', linestyle='dashed')
    ax1.set_title(r'Both sides of the continuity equation: a = {}, $\Lambda = 6$'.format(np.round(a_val[0], 3)), fontsize=14)
    ax1.set_xlabel(r'$\mathrm{x}\;[h^{-1}\;\mathrm{Mpc}]$')
    # ax1.set_ylabel(r'$\mathrm{v}\;[\mathrm{km/s}]$')
    # ax1.plot(x, dv_dx, c='b', label=r'RHS')
    # ax1.plot(x, dd_dt_num, c='k', ls='dashed', label='LHS')
    ax1.plot(x, MH_0, c='b', label='Sch')
    # ax1.plot(x, zel_vel, c='k', label='Zel', ls='dashed')

    # print(r'$\hbar$ = {}, $\sigma_x$ = {}, $\sigma_p$ = {}'.format(h, sigma_x, h / (2 * sigma_x)))
    # print(np.max(dd_dt_num))
    # print(np.max(dv_dx / a0**2) / np.max(dd_dt_num))
    # plt.plot(x_zel, Z1, c='r', ls='dotted', lw=2)

    plt.legend(fontsize=12, loc='upper right')
    plt.savefig(loc2 + 'plots/mz_runs/dc_run14/den_{}.png'.format(i), bbox_inches='tight', dpi=120)
    # plt.show()
    plt.close()
    print('Saving Nt = {}'.format(Nt))
    # break
