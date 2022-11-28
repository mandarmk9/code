#!/usr/bin/env python3
import sys
sys.path.append('/vol/aibn31/data1/mandar/code/')
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings
from functions import *
from SPT import *
from zel import eulerian_sampling as es

loc2 = '/vol/aibn31/data1/mandar/data/sch_hfix_run11/'
filename_SPT = loc2 + 'spt_kern.hdf5'

Nfiles = 594

a_list = np.empty(Nfiles)
for i in range(Nfiles):
   with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
       ls = list(hdf.keys())
       a = np.array(hdf.get(str(ls[1])))
       a_list[i] = a

filename = 'sch_a_list_run10.hdf5'
with h5py.File(filename, mode='w') as hdf:
    hdf.create_dataset('a_list', data=a_list)

print("Collecting Schroedinger scalefactors...")
with h5py.File(filename, mode='r') as hdf:
    ls = list(hdf.keys())
    a_list = np.array(hdf.get(str(ls[0])))

print("Matching to Gadget scalefactors...")
Nfiles_gad = 60

for j in range(17, Nfiles_gad-7):
    gadget_files = '/vol/aibn31/data1/mandar/code/N420/'
    file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
    pos = np.array(file['/Positions'])
    vel = np.array(file['/Velocities'])
    header = file['/Header']
    a_gadget = header.attrs.get('a')
    N = int(pos.size)
    file.close()

    a_diff = np.abs(a_list - a_gadget)
    a_ind = np.where(a_diff == np.min(a_diff))[0][0]

    print(a_list[a_ind], a_gadget)

    with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(a_ind), 'r') as hdf:
        ls = list(hdf.keys())
        A = np.array(hdf.get(str(ls[0])))
        a = np.array(hdf.get(str(ls[1])))
        L, h, m, H0 = np.array(hdf.get(str(ls[2])))
        psi = np.array(hdf.get(str(ls[3])))
        print('\na =', a)

    Nx = psi.size
    dx = L / Nx

    x = np.arange(0, L, dx)
    k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
    p = np.sort(k * h)

    q = np.arange(0, L, L/N)
    k_pos = np.fft.fftfreq(q.size, q[1]-q[0]) * 2 * np.pi

    sigma_x = 5 * dx
    sigma_p = h / (2 * sigma_x)
    # print(r'h = {}, $\sigma_x$ = {}, $\sigma_p$ = {}, p = {}'.format(h, sigma_x, sigma_p, m*(np.max(v))/a0))
    sm = 1 / (4 * (sigma_x**2))

    dist = x - 0
    dist[dist < 0] += L
    dist[dist > L/2] = - L + dist[dist > L/2]

    W_x = np.exp(-sm * ((dist)**2))
    W_k_an = np.exp(- (k ** 2) / (4 * sm))
    W_k_num = np.fft.fft(W_x) * dx

    Lambda = 6 #must be less than k_NL; we think k_NL < 11
    EFT_sm_kern = np.exp(- (k ** 2) / (2 * Lambda**2))
    EFT_sm_kern_nbody = np.exp(- (k_pos ** 2) / (2 * Lambda**2))

    # n = 4
    # F = SPT_agg(n, x, s=1)
    # spt_write_to_hdf5(filename_SPT, F)

    n = 3
    F = spt_read_from_hdf5(filename_SPT)
    den_spt = SPT_final(F, a)[n-1]
    dk_spt = np.fft.fft(den_spt) * dx
    dk_spt *= EFT_sm_kern
    dx_spt = np.real(np.fft.ifft(dk_spt)) / dx

    theta = - den_spt * H0 / np.sqrt(a)
    v_spt = np.real(np.fft.ifft(np.fft.fft(spectral_calc(theta, k, o=1, d=1)) * EFT_sm_kern))

    #Husimi Moments
    psi_star = np.conj(psi)
    grad_psi = spectral_calc(psi, k, o=1, d=0)
    grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
    lap_psi = spectral_calc(psi, k, o=2, d=0)
    lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

    MW_0 = np.abs(psi ** 2)
    MH_0_k = np.fft.fft(MW_0) * W_k_an
    MH_0 = np.real(np.fft.ifft(MH_0_k))

    MW_1 = (1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi))
    MH_1_k = np.fft.fft(MW_1) * W_k_an
    MH_1 = np.real(np.fft.ifft(MH_1_k))

    MW_2 = - ((h**2) / 4) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))
    MH_2_k = np.fft.fft(MW_2) * W_k_an
    MH_2 = np.real(np.fft.ifft(MH_2_k))

    CH_1 = MH_1 / MH_0
    v_pec = CH_1 / (m * a)
    v_sch = np.real(np.fft.ifft(np.fft.fft(v_pec) * EFT_sm_kern))

    v_nbody = np.real(np.fft.ifft(np.fft.fft(vel) * EFT_sm_kern_nbody))
    #
    # d_l = np.real(np.fft.ifft(np.fft.fft(MH_0) * EFT_sm_kern))
    # v_l = np.real(np.fft.ifft(np.fft.fft(v_pec) * EFT_sm_kern))
    # tau_l = np.real(np.fft.ifft(np.fft.fft(MH_2) * EFT_sm_kern))
    #
    # dv_l = spectral_calc(v_l, k, o=1, d=0) * np.sqrt(a) / H0
    #
    # from scipy.optimize import curve_fit
    # def fitting_function(X, a0, a1, a2):
    #     x1, x2 = X
    #     return a0 + a1*x1 + a2*x2
    #
    # guesses = 1e-5, 1e-5, 1e-5
    # FF = curve_fit(fitting_function, (d_l, dv_l), tau_l, guesses, method='lm')
    # a0, a1, a2 = FF[0]
    # cov = FF[1]
    # err0, err1, err2 = np.sqrt(np.diag(cov))
    #
    # fit = fitting_function((d_l, dv_l), a0, a1, a2)
    #
    # print(r'$C_0 = {} \pm {}$'.format(np.round(a0, 7), np.round(err0, 7)))
    # print(r'$C_1 = {} \pm {}$'.format(np.round(a1, 7), np.round(err1, 7)))
    # print(r'$C_2 = {} \pm {}$'.format(np.round(a2, 7), np.round(err2, 7)))
    #
    #
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.set_title(r'a = {}'.format(str(np.round(a, 3))), fontsize=14)
    # ax.grid(linewidth=0.2, color='gray', linestyle='dashed')
    # ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=14)
    # # ax.set_ylabel(r'$\delta$(x)', fontsize=14)
    # # ax.set_ylabel(r'$\partial_{x}v_{l}\;\;[h\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$', fontsize=14)
    # ax.set_ylabel(r'$\tau_{l} \; [\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\left(\frac{\mathrm{km}}{\mathrm{s}}\right)^{2}] $', fontsize=14)
    #
    # ax.plot(x, tau_l, c='b', lw=2)#, label=r'$\tau_{l}$ from $\psi$')
    # ax.plot(x, fit, ls='dashed', lw=2, c='k', label=r'EFT fit to $\tau_{l}$')
    # # ax.plot(x, d_l / 100, c='r', lw=2, label=r'$\delta_l$')
    # # ax.plot(x, dv_l / 100, c='k', lw=2, label=r'$\partial_{x} v_{l}$')
    # # ax.plot(x, tau_l, c='b', label=r'$\tau_{l}$')
    # # plt.legend(fontsize=12, loc='upper right')
    #
    # print('saving...')
    # plt.savefig('/vol/aibn31/data1/mandar/plots/EFT_sch_nbody/fit.pdf', bbox_inches='tight', dpi=120)
    # plt.close()

    # den_sch = MH_0 #schrodinger overdensity from Husimi
    # dk_sch = np.fft.fft(den_sch) * dx
    # dk_sch *= EFT_sm_kern #the EFT smoothing
    # dx_sch = np.real(np.fft.ifft(dk_sch)) / dx
    #
    # den_nbody = kde_gaussian(q, pos, sm, L)
    #
    # den_zel = np.zeros(x.size)
    #
    # zel_flag = 1
    # warnings.filterwarnings('error')
    # try:
    #    # den_zel[:] = es(x, a, A)[1][:]
    #    # dk_zel = np.fft.fft(den_zel) * dx
    #    # dk_zel *= W_k_an
    #    # dk2_zel = np.abs(dk_zel) ** 2 #/ (a**2)
    #    # dx_zel = np.real(np.fft.ifft(dk_zel)) / dx
    # except Warning:
    #    zel_flag = 0
    #    # dk2_zel = np.empty(Nx) #np.zeros(Nx)

    # dk_nbody = np.fft.fft(den_nbody) * EFT_sm_kern_nbody * ((L / N)) #factor of L/N for fourier space comparisons; use den_nbody when comparing in real space
    #
    # dk2_nbody = np.abs(dk_nbody) ** 2 #/ (a**2)
    # dk2_spt = np.abs(dk_spt) ** 2 #/ (a**2)
    # dk2_sch = np.abs(dk_sch) ** 2 #/ (a_sch**2)


    CH_1 = MH_1 / MH_0
    CH_1_l = np.real(np.fft.ifft(np.fft.fft(CH_1) * EFT_sm_kern))
    v_sch = CH_1 / (m * a)
    #
    # Psi_q = -Psi_q_finder(x, A)
    # Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
    # x_eul = x + Psi_t #eulerian position
    # v_zel_us = H0 * np.sqrt(a) * Psi_q #peculiar velocity
    #
    # v_zel = np.real(np.fft.ifft(np.fft.fft(v_zel_us) * W_k_an))

    # theta = - den_spt * H0 / np.sqrt(a)
    # v_spt_us = spectral_calc(theta, k, o=1, d=1)
    # v_spt = np.real(np.fft.ifft(np.fft.fft(v_spt_us) * W_k_an))

    fig, ax = plt.subplots()
    ax.set_title(r'a = {}'.format(str(np.round(a, 3))))
    ax.grid(linewidth=0.15, color='gray', linestyle='dashed')

    ax.plot(x, v_sch, c='b', lw=2, label='Sch')
    # ax.plot(x, v_spt, c='r', ls='dashed', lw=2, label='SPT')
    # ax.plot(q, vel, c='r', lw=2, label='N-body')

    # ax.plot(x, dx_spt, c='k', lw=2, ls='dashed', label='{}SPT'.format(n))
    # dv_l = -spectral_calc(v_sch * MH_0, k, o=1, d=0) / (m * (a**2))

    # ax.plot(x, dv_l, c='b', lw=2, label='Sch')
    # ax.plot(x_eul, v_zel, c='k', lw=2, ls='dashed', label='Zel')

    ax.plot(pos, vel, c='r', lw=2, ls='dashed', label='N-body')

    # if zel_flag == 1:
    #     # ax.scatter(k, dk2_zel, color='yellow', s=50, label='Zel')
    #     # ax.plot(x, dx_zel, c='yellow', lw=2, ls='dotted', label='Zel')
    #     ax.plot(x, v_zel, c='y', lw=2, ls='dotted', label='Zel')
    # else:
    #     pass


    ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
    ax.set_xlabel(r'x$\,$[$h^{-1} \; \mathrm{Mpc}$]', fontsize=12)
    # ax.set_ylabel(r'$\delta(x)$', fontsize=12)

    # ax.scatter(k, dk2_sch, color='b', s=30, label='Sch')
    # ax.scatter(k, dk2_spt, color='k', s=15, label='{}SPT'.format(n))
    # ax.scatter(k_pos, dk2_nbody, color='r', s=25, label='Nbody')
    # ax.set_xlabel(r'$k\;[h\mathrm{Mpc}^{-1}]$', fontsize=12)
    # ax.set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=12)
    # # ax.yaxis.set_label_position("right")
    # ax.ticklabel_format(style='scientific', scilimits=(-3, 3))
    # # ax.set_ylim(-1e2, 1.2e3) #for long λ modes
    # ax.set_xlim(0, 13.5)
    # ax.set_ylim(-0.01, 0.16) #for all modes

    # ax.set_ylim(-0.41, 0.45) #for long λ mode in real space with Λ = 1
    # ax.set_ylim(-0.3, 0.4) #for long λ mode in real space with Λ = 2
    plt.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))

    # print('saving...')
    plt.show()
    # plt.savefig('/vol/aibn31/data1/mandar/plots/test/vel_{0:03d}.png'.format(a_ind), bbox_inches='tight', dpi=120)
    # plt.close()
    #
    break
    #
    # den_arrays = [dk2_spt, dk2_sch]
    # # a2 = [a, a_gadget]
    # a2 = a
    # filename = '/vol/aibn31/data1/mandar/data/modes/all_modes_sch_spt/dk2_{0:03d}.hdf5'.format(a_ind)
    # write_to_hdf5(filename, den_arrays, a2, k, n, Lambda)


tn = time.time()
print("Done! Finished in {}s".format(np.round(tn-t0, 3)))


# FMXE-LOAD-G5U21
