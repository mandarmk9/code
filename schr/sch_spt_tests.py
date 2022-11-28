#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
from functions import spectral_calc, dn
from EFT_solver import EFT_sm_kern, smoothing
from SPT import SPT_final
# plt.style.use('clean_1d')

loc = '../'
run = '/sch_no_m_run3/'
Lambda = 5
j = 0

with h5py.File(loc + 'data' + run + 'psi_{0:05d}.hdf5'.format(j), 'r') as hdf:
    ls = list(hdf.keys())
    A = np.array(hdf.get(str(ls[0])))
    a = np.array(hdf.get(str(ls[1])))
    L, h, H0 = np.array(hdf.get(str(ls[2])))
    psi = np.array(hdf.get(str(ls[3])))

Nx = psi.size
dx = L / Nx
x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
rho_0 = 27.755
sigma_x = np.sqrt(h / 2)
sm = 1 / (4 * (sigma_x**2))
W_k_an = np.exp(- (k ** 2) / (4 * sm))
W_EFT = EFT_sm_kern(k, Lambda)

dc_in = (A[0] * np.cos(A[1]*x)) + (A[2] * np.cos(A[3]*x))
# dc_in_bar = smoothing(dc_in, W_EFT)

n = 3 #overdensity order of the SPT
F = dn(n, k, dc_in)

d1k = (np.fft.fft(F[0]) / Nx) #* W_EFT
d2k = (np.fft.fft(F[1]) / Nx) #* W_EFT
d3k = (np.fft.fft(F[2]) / Nx) #* W_EFT

order_2 = (d1k * np.conj(d1k)) * (a**2)
order_3 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
order_13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
order_22 = (d2k * np.conj(d2k)) * (a**4)
order_4 = order_22 + order_13
order_5 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5) #+ (d1k * np.conj(d4k)) + ((d4k * np.conj(d1k))))
order_6 = (d3k * np.conj(d3k)) * (a**6)

dk_spt = order_2 + order_3 + order_4 + order_5 + order_6

MW_0 = np.abs(psi ** 2)
dk_sch = np.fft.fft(MW_0 - 1) / Nx #* W_k_an * W_EFT
# sch = np.abs(dk_sch)**2

dc_sch = np.real(np.fft.ifft(dk_sch * Nx))
dc_spt = np.real(np.fft.ifft(dk_spt * Nx))

dc_spt_eft = np.real(np.fft.ifft(dk_spt * W_EFT * Nx))

fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
# ax[0].set_title('k = {} [$h \; \mathrm{{Mpc}}^{{-1}}$]'.format(k[mode]))
ax[0].set_ylabel(r'$\delta(x)$', fontsize=14) # / a^{2}$')
ax[1].set_xlabel(r'$x$', fontsize=14)
ax[0].plot(x, dc_sch, c='b', label='sch', lw=2)
ax[0].plot(x, dc_spt, c='k', label='spt', lw=2)
ax[0].plot(x, dc_spt_eft, c='r', label='spt, two smoothings', lw=2)

ax[1].axhline(0, color='b')
# ax[1].plot(a_list, err_spt, color='k')
# ax[1].plot(a_list_gad, err_nbody, color='r')
# ax[1].plot(a_list, err_sch, color='b')/

# ax[1].set_xlim(0, 30)
# ax[1].set_ylim(-1, 5)
# ax[1].minorticks_on(direction='in')
ax[1].set_ylabel('% err', fontsize=14)

for i in range(2):
    ax[i].tick_params(axis='both', which='both', direction='in')
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].grid(lw=0.2, ls='dashed', color='grey')
    ax[i].yaxis.set_ticks_position('both')
    # ax[i].axvline(2, color='g', ls='dashed', label=r'a$_{\mathrm{sc}}$')

ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))

plt.savefig('/vol/aibn31/data1/mandar/plots/sch_no_m_run3/sch_vs_spt/dc_{}.png'.format(j), bbox_inches='tight', dpi=120)
