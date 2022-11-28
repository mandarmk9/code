#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import h5py
import matplotlib.pyplot as plt

from functions import *
from zel import eulerian_sampling

loc2 = '/vol/aibn31/data1/mandar/'
run = '/sch_hfix_run10/'

filename = 'sch_a_list_run7.hdf5'
# Nfiles = 50075
# a_list = np.empty(Nfiles)
# for i in range(Nfiles):
#    with h5py.File(loc2 + 'data' + run + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
#        ls = list(hdf.keys())
#        a = np.array(hdf.get(str(ls[1])))
#        a_list[i] = a
#
# with h5py.File(filename, mode='w') as hdf:
#     hdf.create_dataset('a_list', data=a_list)

# print("Collecting Schroedinger scalefactors...")
# with h5py.File(filename, mode='r') as hdf:
#     ls = list(hdf.keys())
#     a_list = np.array(hdf.get(str(ls[0])))

# print("Matching to Gadget scalefactors...")

# for j in range(0, 35, 3):
for a_ind in range(118, 250, 4):
   # gadget_files = '/vol/aibn31/data1/mandar/code/N420/'
   # file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
   # pos = np.array(file['/Positions'])
   # header = file['/Header']
   # a_gadget = header.attrs.get('a')
   # vel = np.array(file['/Velocities']) / np.sqrt(a_gadget)
   # N = int(pos.size)
   # file.close()

   # a_diff = np.abs(a_list - a_gadget)
   # a_ind = np.where(a_diff == np.min(a_diff))[0][0]
   #
   # print(a_list[a_ind], a_gadget)
   j = a_ind
   with h5py.File(loc2 + 'data' + run + 'psi_{0:05d}.hdf5'.format(a_ind), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, m, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))

   print(a)

   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

   # q = np.arange(0, L, L/N)
   # k_pos = np.fft.fftfreq(q.size, q[1]-q[0]) * 2 * np.pi
   rho_0 = 27.755

   # Lambda = 6
   # sm = (Lambda**2) / 2
   # norm = (np.sqrt(np.pi / sm))
   sigma_x = 500 * dx #np.sqrt(h / 2)
   sigma_p = h / (2 * sigma_x)
   print(r'h = {}, $\sigma_x$ = {}, $\sigma_p$ = {}'.format(h, sigma_x, sigma_p))
   sm = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sm))
   # W_k_nbody = np.exp(- (k_pos ** 2) / (4 * sm))


   # den_nbody = kde_gaussian(q, pos, sm, L)

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, k, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
   lap_psi = spectral_calc(psi, k, o=2, d=0)
   lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)
   MW_0 = np.abs(psi ** 2)
   MW_00 = np.abs(psi ** 2) - 1
   MW_1 = (1j * h / m) * ((psi * grad_psi_star) - (psi_star * grad_psi))
   MW_2 = - ((h**2) / 4) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))

   MH_0_k = np.fft.fft(MW_0) * W_k_an
   MH_0 = np.real(np.fft.ifft(MH_0_k))

   MH_00_k = np.fft.fft(MW_00) * W_k_an
   MH_00 = np.real(np.fft.ifft(MH_00_k))

   MH_1_k = np.fft.fft(MW_1) * W_k_an
   MH_1 = np.real(np.fft.ifft(MH_1_k))

   MH_2_k = np.fft.fft(MW_2) * W_k_an
   MH_2 = np.real(np.fft.ifft(MH_2_k))

   CH_1 = MH_1 / MH_0
   v_pec = CH_1 / a

   # p = np.sort(k * h)
   # dv = p[1] - p[0]
   # f_H = husimi(psi, x, p, sm, h, L)
   #
   # MW_0f = np.trapz(f_H, x=p, dx=dv, axis=0)
   # MH_0f = np.real(np.fft.ifft(np.fft.fft(MW_0f) * W_k_an))
   #
   # for l in range(p.size):
   #    f_H[l, :] *= p[l]
   #
   # MW_1f = np.trapz(f_H, x=p, dx=dv, axis=0)
   # MH_1f = np.real(np.fft.ifft(np.fft.fft(MW_1f) * W_k_an))
   # CH_1f = MH_1f / MH_0f / a / dx / rho_0

   # vel_sm = np.real(np.fft.ifft(np.fft.fft(vel) * W_k_nbody))

   # d_l = MH_00
   # v_l = ((MH_1 / (MH_0)) / a ) * (2 * norm)
   # dv_l = spectral_calc(v_l, k, o=1, d=0) * np.sqrt(a) / H0

   Psi_q = -Psi_q_finder(x, A)
   Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
   x_zel = x + Psi_t #eulerian position
   v = H0 * np.sqrt(a) * Psi_q #peculiar velocity

   v_k = np.fft.fft(v)
   v_k *= (W_k_an)
   v_zel = np.real(np.fft.ifft(v_k))

   # den_zel = eulerian_sampling(x, a, A)[1]
   # dc_zel = np.real(np.fft.ifft(np.fft.fft(den_zel) * W_k_an))

   fig, ax = plt.subplots(figsize=(8, 5))
   ax.set_title(r'a = {}'.format(str(np.round(a, 4))))
   ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
   ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
   ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)

   # ax.plot(x, dc_zel, c='r', lw=2, label='N-body')
   # ax.plot(x, MH_00, c='b', ls='dashed', lw=2, label='Sch')

   # ax.plot(q, vel_sm, c='r', lw=2, label='N-body')
   ax.plot(x, v_zel, c='k', lw=2, label='Zel')
   ax.plot(x, v_pec, c='b', ls='dashed', lw=2, label='Sch')

   plt.legend()

   print('saving...')
   plt.savefig(loc2 + 'plots/sch_vel/den_{0:03d}.png'.format(j), bbox_inches='tight', dpi=120)
   plt.close()
   # break


tn = time.time()
print('This took {}s'.format(np.round(tn-t0, 3)))
