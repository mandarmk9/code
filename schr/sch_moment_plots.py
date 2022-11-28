#!/usr/bin/env python3
import sys
sys.path.append('/vol/aibn31/data1/mandar/code/')
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import spectral_calc, dn, poisson_solver, Psi_q_finder, EFT_sm_kern
from SPT import SPT_final
from zel import eulerian_sampling

def smoothing(field, kernel):
   return np.real(np.fft.ifft(np.fft.fft(field) * kernel))

l = 250
for j in range(l, l+1):
   with h5py.File('/vol/aibn31/data1/mandar/data/sch_multi_k/psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))

   print('a = ', a)
   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))

   sigma_x = 0.1 * np.sqrt(h / 2)
   sigma_p = h / (2 * sigma_x)
   sm = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sm))

   Psi_q = -Psi_q_finder(x, A, L)
   x_eul = x + a*Psi_q
   v_zel = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity
   # nd = eulerian_sampling(x, a, A, L)[1] + 1

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, L, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), L, o=1, d=0)
   lap_psi = spectral_calc(psi, L, o=2, d=0)
   lap_psi_star = spectral_calc(np.conj(psi), L, o=2, d=0)

   #we will scale the Sch moments to make them compatible with the definition in Hertzberg (2014), for instance
   MW_0 = np.abs(psi ** 2)
   MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))
   MW_2 = (- ((h**2 / 2)) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star)))

   MH_0 = MW_0#smoothing(MW_0, W_k_an)
   MH_1 = MW_1#smoothing(MW_1, W_k_an)
   MH_2 = MW_2#smoothing(MW_2, W_k_an) + ((sigma_p**2) * MH_0)

   # Lambda = 5 * (2 *np.pi / L)
   # W_EFT = EFT_sm_kern(k, Lambda)
   # MH_0 = smoothing(MH_0, W_EFT)
   # MH_1 = smoothing(MH_1, W_EFT)
   # MH_2 = smoothing(MH_2, W_EFT)
   plots_folder = 'test'
   ind1, ind2 = 0, -1
   # print(x[ind1], x[ind2])
   fig, ax = plt.subplots()
   ax.set_title('a = {}'.format(a))
   ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
   ax.set_ylabel(r'$\mathrm{M}^{(0)}$')
   ax.plot(x[ind1:ind2], MH_0[ind1:ind2]-1, c='k', lw=2, label=r'Husimi from $\Psi$')
   # ax.plot(x, nd, c='cyan', ls='dashdot', lw=2, label=r'Zel')
   # ax.plot(x, MW_0 -1, c='r', ls='dotted', lw=2, label=r'Wigner; from $\psi$')

   ax.tick_params(axis='both', which='both', direction='in')
   ax.ticklabel_format(scilimits=(-2, 3))
   ax.grid(lw=0.2, ls='dashed', color='grey')
   ax.yaxis.set_ticks_position('both')
   ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   plt.show()
   # plt.savefig('/vol/aibn31/data1/mandar/plots/{}/M0_l/M0_{}.png'.format(plots_folder, j))
   # plt.close()
   #
   # fig, ax = plt.subplots()
   # ax.set_title('a = {}'.format(a))
   # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
   # ax.set_ylabel(r'$\mathrm{M}^{(1)}$')
   #
   # ax.plot(x, MH_1, c='k', lw=2, label=r'from $\Psi$')
   # # ax.plot(x_eul, v_zel * a**2, c='cyan', ls='dashed', lw=2, label=r'Zel')
   #
   # ax.tick_params(axis='both', which='both', direction='in')
   # ax.ticklabel_format(scilimits=(-2, 3))
   # ax.grid(lw=0.2, ls='dashed', color='grey')
   # ax.yaxis.set_ticks_position('both')
   # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #
   # plt.savefig('/vol/aibn31/data1/mandar/plots/{}/M1_l/M1_{}.png'.format(plots_folder, j))
   # plt.close()
   #
   # fig, ax = plt.subplots()
   # ax.set_title('a = {}'.format(a))
   # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$')
   # ax.set_ylabel(r'$\mathrm{M}^{(2)}$')
   #
   # ax.plot(x, MH_2, c='k', lw=2, label=r'from $\Psi$')
   # # ax.plot(x, MW_2, c='r', ls='dotted', lw=2, label=r'Wigner; from $\psi$')
   #
   # ax.tick_params(axis='both', which='both', direction='in')
   # ax.ticklabel_format(scilimits=(-2, 3))
   # ax.grid(lw=0.2, ls='dashed', color='grey')
   # ax.yaxis.set_ticks_position('both')
   # ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   #
   # plt.savefig('/vol/aibn31/data1/mandar/plots/{}/M2_l/M2_{}.png'.format(plots_folder, j))
   # plt.close()
