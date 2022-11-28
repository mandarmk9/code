#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py

from functions import spectral_calc, dn, Psi_q_finder
from SPT import SPT_final
from zel import eulerian_sampling

def EFT_sm_kern(k, Lambda):
   kernel = np.exp(- (k ** 2) / (2 * Lambda**2))
   return kernel / sum(kernel)

def smoothing(field, kernel):
   return np.real(np.fft.ifft(np.fft.fft(field) * kernel))

loc = '../'
run = '/sch_early_start_run1/'

for j in range(0, 24, 23):
   with h5py.File(loc + 'data' + run + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
      ls = list(hdf.keys())
      A = np.array(hdf.get(str(ls[0])))
      a = np.array(hdf.get(str(ls[1])))
      L, h, H0 = np.array(hdf.get(str(ls[2])))
      psi = np.array(hdf.get(str(ls[3])))

   print('a = ', a)

   Nx = psi.size
   dx = L / Nx
   x = np.arange(0, L, dx)
   k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
   Lambda = 5
   a_dot = H0 * (a**(-1/2))

   sigma_x = np.sqrt(h / 2) #/ 25
   sigma_p = h / (2 * sigma_x)
   sm = 1 / (4 * (sigma_x**2))
   W_k_an = np.exp(- (k ** 2) / (4 * sm))
   # Lambda_husimi = 1 / (np.sqrt(2) * sigma_x)
   # W_k_an = np.exp(- (k ** 2) / (2 * Lambda_husimi**2))

   nd = eulerian_sampling(x, a, A)[1]

   W_EFT = EFT_sm_kern(k, Lambda)
   dc_zel = smoothing(nd, W_EFT)

   psi_star = np.conj(psi)
   grad_psi = spectral_calc(psi, k, o=1, d=0)
   grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)

   MW_0 = np.abs(psi ** 2)
   MW_1 = ((1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi)))

   MH_1_k = np.fft.fft(MW_1) * W_k_an * W_EFT
   MH_1 = np.real(np.fft.ifft(MH_1_k))

   MH_0_k = np.fft.fft(MW_0 - 1) * W_k_an * W_EFT
   MH_0 = np.real(np.fft.ifft(np.fft.fft(MW_0) * W_k_an * W_EFT))

   dc_sch = np.real(np.fft.ifft(MH_0_k))
   v_sch = MW_1 / MW_0 / a
   v_sch = smoothing(v_sch, W_EFT)

   Psi_q = -Psi_q_finder(x, A)
   x_eul = x + a*Psi_q

   n = 3 #overdensity order of the SPT
   dc_in = (A[0] * np.cos(A[1]*x)) + (A[2] * np.cos(A[3]*x))

   def spt_tr(x, k, a_dot, dc_in, W_EFT):
      """SPT with a truncation of the 'initial' conditions. No smoothing."""
      dc_in_bar = smoothing(dc_in, W_EFT)
      F = dn(n, k, dc_in_bar)
      den = SPT_final(F, a)[2]
      theta = - den * H0 / np.sqrt(a)
      vel = spectral_calc(theta, k, o=1, d=1)
      return den, vel

   def spt_sm(x, k, a_dot, dc_in, W_EFT):
      """vanilla SPT with a smoothing of the evolved fields."""
      F = dn(n, k, dc_in)
      den = SPT_final(F, a)[2]
      theta = - den * H0 / np.sqrt(a)
      v_spt = spectral_calc(theta, k, o=1, d=1)
      vel = smoothing(v_spt, W_EFT)
      den = smoothing(den, W_EFT)
      return den, vel

   def spt_tr_sm(x, k, a_dot, dc_in, W_EFT):
      """SPT with a truncation of the 'initial' conditions
         and smoothing of the evolved field."""
      dc_in_bar = smoothing(dc_in, W_k_an)
      F = dn(n, k, dc_in_bar)
      den = SPT_final(F, a)[2]
      theta = - den * H0 / np.sqrt(a)
      v_spt = spectral_calc(theta, k, o=1, d=1)
      vel = smoothing(v_spt, W_EFT)
      den = smoothing(den, W_EFT)
      return den, vel

   dc_spt_tr = spt_tr(x, k, a_dot, dc_in, W_EFT)[0]
   dc_spt_sm = spt_sm(x, k, a_dot, dc_in, W_EFT)[0]
   dc_spt_tr_sm = spt_tr_sm(x, k, a_dot, dc_in, W_EFT)[0]

   v_spt_tr = spt_tr(x, k, a_dot, dc_in, W_EFT)[1]
   v_spt_sm = spt_sm(x, k, a_dot, dc_in, W_EFT)[1]
   v_spt_tr_sm = spt_tr_sm(x, k, a_dot, dc_in, W_EFT)[1]

   Psi_q = -Psi_q_finder(x, A)
   v_zel = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity
   v_zel = smoothing(v_zel, W_EFT)

   fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1,4]})
   ax[0].set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
   ax[0].set_ylabel(r'$\delta(x)$', fontsize=14)
   ax[1].set_xlabel(r'$x$', fontsize=14)

   # ax[0].plot(x, dc_zel, c='k', lw=2, label='Zel')
   ax[0].plot(x, dc_sch, c='b', lw=2, label='Sch: husimi') #, ls='dashed',
   ax[0].plot(x, dc_spt_tr, c='r', ls='dashdot', lw=2, label='SPT: truncated')
   ax[0].plot(x, dc_spt_sm, c='brown', ls='dashed', lw=2, label='SPT: smoothed')
   ax[0].plot(x, dc_spt_tr_sm, c='cyan', ls='dotted', lw=2, label='SPT: tr and sm')

   #bottom panel; errors
   # err_sch = (dc_sch - dc_zel) #* 100 / dc_zel
   err_spt_tr = (dc_spt_tr - dc_sch) #* 100 / dc_zel
   err_spt_sm = (dc_spt_sm - dc_sch) #* 100 / dc_zel
   err_spt_tr_sm = (dc_spt_tr_sm - dc_sch) #* 100 / dc_zel

   ax[1].axhline(0, color='b')
   # ax[1].plot(x, err_sch, ls='dashed', lw=2.5, c='b')
   ax[1].plot(x, err_spt_tr, ls='dashdot', lw=2.5, c='r')
   ax[1].plot(x, err_spt_sm, ls='dashed', lw=2.5, c='brown')
   ax[1].plot(x, err_spt_tr_sm, ls='dotted', lw=2.5, c='cyan')

   # ax[1].set_ylabel('% err', fontsize=14)
   ax[1].set_ylabel('difference', fontsize=14)

   ax[1].minorticks_on()

   for i in range(2):
       ax[i].tick_params(axis='both', which='both', direction='in')
       ax[i].ticklabel_format(scilimits=(-2, 3))
       ax[i].grid(lw=0.2, ls='dashed', color='grey')
       ax[i].yaxis.set_ticks_position('both')

   ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   plt.savefig('../plots/sch_early_start_run1/sch_spt_tests/eft_den_{}.png'.format(j), bbox_inches='tight', dpi=120)
   plt.close()

   fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1,4]})
   ax[0].set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
   ax[0].set_ylabel(r'$v(x)$', fontsize=14)
   ax[1].set_xlabel(r'$x$', fontsize=14)

   # ax[0].plot(x, v_zel, c='k', lw=2, label='Zel')
   ax[0].plot(x, v_sch, c='b', lw=2, label='Sch: husimi') # ls='dashed',
   ax[0].plot(x, v_spt_tr, c='r', ls='dashdot', lw=2, label='SPT: truncated')
   ax[0].plot(x, v_spt_sm, c='brown', ls='dashed', lw=2, label='SPT: smoothed')
   ax[0].plot(x, v_spt_tr_sm, c='cyan', ls='dotted', lw=2, label='SPT: tr and sm')

   #bottom panel; errors
   # err_sch = (v_sch - v_zel) #* 100 / dc_zel
   err_spt_tr = (v_spt_tr - v_sch) #* 100 / dc_zel
   err_spt_sm = (v_spt_sm - v_sch) #* 100 / dc_zel
   err_spt_tr_sm = (v_spt_tr_sm - v_sch) #* 100 / dc_zel

   ax[1].axhline(0, color='b')
   # ax[1].plot(x, err_sch, ls='dashed', lw=2.5, c='b')
   ax[1].plot(x, err_spt_tr, ls='dashdot', lw=2.5, c='r')
   ax[1].plot(x, err_spt_sm, ls='dashed', lw=2.5, c='brown')
   ax[1].plot(x, err_spt_tr_sm, ls='dotted', lw=2.5, c='cyan')

   # ax[1].set_ylabel('% err', fontsize=14)
   ax[1].set_ylabel('difference', fontsize=14)

   ax[1].minorticks_on()

   for i in range(2):
       ax[i].tick_params(axis='both', which='both', direction='in')
       ax[i].ticklabel_format(scilimits=(-2, 3))
       ax[i].grid(lw=0.2, ls='dashed', color='grey')
       ax[i].yaxis.set_ticks_position('both')

   ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
   plt.savefig('../plots/sch_early_start_run1/sch_spt_tests/eft_vel_{}.png'.format(j), bbox_inches='tight', dpi=120)
   plt.close()

   # break
