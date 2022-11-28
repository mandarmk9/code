#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import *

run = '/sch_hfix_run18/'
loc2 = '/vol/aibn31/data1/mandar/'
# j = 75659
mean_MH2 = []
a_list = []
for j in range(4400, 4955):
   with h5py.File(loc2 + 'data' + run + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      print(ls)
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, m, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))
      print('a = ', a)
   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
   p = k * h #np.sort(k * h)
   dp = np.abs(p[1] - p[0])

   sigma_x = 2.5 * dx
   sigma_p = h / (2 * sigma_x)
   sigma_k = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sigma_k))

   # f_H = husimi(psi, x, p, sigma_x, h, L)
   # MH_0_f = moment(f_H, p, dp, 0)
   # MH_0_fn = (MH_0_f - np.mean(MH_0_f)) / np.mean(MH_0_f)
   #
   # MH_1_f = moment(f_H, p, dp, 1)

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, k, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
   lap_psi = spectral_calc(psi, k, o=2, d=0)
   lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

   MW_0 = np.abs(psi ** 2)
   MW_1 = (1j * h / 2) * ((psi * grad_psi_star) - (psi_star * grad_psi))
   MW_2 = - ((h**2 / 4)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))

   MH_0_k = np.fft.fft(MW_0) * W_k_an
   MH_0 = np.real(np.fft.ifft(MH_0_k))

   MH_1_k = np.fft.fft(MW_1) * W_k_an
   MH_1 = np.real(np.fft.ifft(MH_1_k))

   MH_2_k = np.fft.fft(MW_2) * W_k_an
   MH_2 = np.real(np.fft.ifft(np.fft.fft(MW_2) * W_k_an)) + ((sigma_p**2) * MH_0)
   # break

# v_l = MH_1 / MH_0

   fig, ax = plt.subplots(figsize=(8, 5))
   ax.set_title(r'a = {}'.format(str(np.round(a, 6))))
   ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
   ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
   # ax.set_xlabel('a', fontsize=12)
   # ax.set_ylabel(r'$[\tau]_{\Lambda} \; [\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\left(\frac{\mathrm{km}}{\mathrm{s}}\right)^{2}] $', fontsize=12)
   # ax.set_ylabel(r'$M^{(0)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}]$', fontsize=12)
   # ax.set_ylabel(r'$M^{(1)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=12)
   # ax.set_ylabel(r'$M^{(2)}_{H}\,[\mathrm{M}_{10}\;\mathrm{Mpc}^{-1}\;\left(\mathrm{km}\;\mathrm{s}^{-1}\right)^{2}]$', fontsize=12)
   # ax.set_ylabel(r'$\frac{M^{(1)}_{H}}{M^{(0)}_{H}}\,[\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=12)

   # ax.plot(x, MH_1_f, c='k', lw=2, label=r'from $f_{H}$')
   # ax.plot(x, MH_1, c='b', ls='dashed', lw=2, label=r'from $\psi$')
   ax.plot(x, MH_0, c='b')
   plt.legend()

   plt.show()
   # print('saving...')
   # plt.savefig(loc2 + 'plots/mz_runs/CH1_run16/ch1_{}.png'.format(j), bbox_inches='tight', dpi=120)
   # plt.close()
   break

tn = time.time()
print('This took {}s'.format(np.round(tn-t0, 3)))
