#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from zel import eulerian_sampling

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
path_k1 = 'cosmo_sim_1d/sim_k_1/run1/'
A = [-0.05, 1, -0.0, 11]
Lambda_int = 3
Lambda = Lambda_int * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

# a0 = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(0))
# q = np.genfromtxt(path + 'output_{0:04d}.txt'.format(0))[:,0]
# q_zel = q[::50]
# P_zel, a_zel = [], []
# k_zel = np.fft.ifftshift(2.0 * np.pi * np.arange(-q_zel.size/2, q_zel.size/2))
# dc_zel = eulerian_sampling(q_zel, a0, A, 1.0)[1]+1 #smoothing(eulerian_sampling(q_zel, a, A, 1.0)[1], k_zel, Lambda, kind)
# dc_zel = smoothing(dc_zel, k_zel, 2*np.pi, kind)

# dk_zel = np.fft.fft(dc_zel) / dc_zel.size
# P_zel_a = np.real(dk_zel * np.conj(dk_zel))
# P_zel.append(P_zel_a[mode])
# a_zel.append(a)

def extract_sm_fields(path, file_num, Lambda, kind, sm=True):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
    moments_file = np.genfromtxt(path + moments_filename)

    x = moments_file[:,0]
    a = moments_file[:,-1][0]
    dc = moments_file[:,2]
    v = moments_file[:,5]
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
    if sm == True:
        dc = smoothing(dc, k, Lambda, kind)
        v = smoothing(v, k, Lambda, kind)
    # dc_k = np.fft.fft(dc)
    # dc_k[2:-1] = 0
    # dc = np.real(np.fft.ifft(dc_k))
    return a, x, dc, v

# j = 10
a_0, x, dc_0, v_0 = extract_sm_fields(path, 0, Lambda, kind)
a_1, x, dc_1, v_1 = extract_sm_fields(path, 12, Lambda, kind)
a_2, x, dc_2, v_2 = extract_sm_fields(path, 23, Lambda, kind)
a_3, x, dc_3, v_3 = extract_sm_fields(path, 35, Lambda, kind)

a_0, x_k1, dc_k1_0, v_k1_0 = extract_sm_fields(path_k1, 0, Lambda, kind)#, sm=False)
a_1, x_k1, dc_k1_1, v_k1_1 = extract_sm_fields(path_k1, 12, Lambda, kind)#, sm=False)
a_2, x_k1, dc_k1_2, v_k1_2 = extract_sm_fields(path_k1, 23, Lambda, kind)#, sm=False)
a_3, x_k1, dc_k1_3, v_k1_3 = extract_sm_fields(path_k1, 35, Lambda, kind)#, sm=False)


# a_1 = 1.93
a_list = [a_0, a_1, a_2 , a_3]
den_list = [dc_0, dc_1, dc_2, dc_3]
vel_list = [v_0, v_1, v_2, v_3]
den_k1_list = [dc_k1_0, dc_k1_1, dc_k1_2, dc_k1_3]
vel_k1_list = [v_k1_0, v_k1_1, v_k1_2, v_k1_3]

# print(a_list)


#density plot
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})

fig.suptitle(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)
# fig.suptitle(r'$k = 1 \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=16, y=0.91)

for i in range(4):
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.88, y=0.85, fontsize=18)
    ax[i].plot(x, den_list[i], c='b', lw=1.5, label=r'\texttt{sim\_k\_1\_11}')
    # if i == 0:
    #     ax[i].plot(q_zel, dc_zel, c='r', ls='dotted', lw=1.5, label='Zel')

    ax[i].plot(x_k1, den_k1_list[i], c='k', lw=1.5, ls='dashed', label=r'\texttt{sim\_k\_1}')
    ax[i].set_ylabel(r'$1+\delta_{l}$', fontsize=20)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
plt.rcParams.update({"text.usetex": True})
# ax[0].legend(fontsize=12.5, loc=1)
plt.rcParams.update({"text.usetex": False})
ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
plt.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/den_ev_sm.pdf', bbox_inches='tight', dpi=300)#, pad_inches=0.3)
# # plt.savefig('../plots/test/den_ev.png', bbox_inches='tight', dpi=300)
plt.close()


#velocity plot
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(Lambda_int, kind_txt), fontsize=18, y=0.91)
# fig.suptitle(r'$k = 1 \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$', fontsize=16, y=0.91)

for i in range(4):
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.88, y=0.85, fontsize=18)
    ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'\texttt{sim\_k\_1\_11}')
    ax[i].plot(x_k1, vel_k1_list[i], c='k', lw=1.5, ls='dashed', label=r'\texttt{sim\_k\_1}')

    ax[i].set_ylabel(r'$\bar{v}_{l}\;[\mathrm{km\,s}^{-1}]$', fontsize=20)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
plt.rcParams.update({"text.usetex": True})
ax[0].legend(fontsize=13, loc=3)
plt.rcParams.update({"text.usetex": False})
ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
plt.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/vel_ev_sm.pdf', bbox_inches='tight', dpi=300)#, pad_inches=0.3)
# plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
plt.close()
