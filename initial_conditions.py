#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

from functions import *

path = 'cosmo_sim_1d/sim_k_1_11/run1/'

def extract_fields(path, file_num):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(file_num)
    moments_file = np.genfromtxt(path + moments_filename)
    # dk_par, a, dx = read_density(path, file_num)
    # x0 = 0.0
    # xn = 1.0 #+ dx
    # x_grid = np.arange(x0, xn, (xn-x0)/dk_par.size)
    # M0_par = np.real(np.fft.ifft(dk_par))
    # M0_par /= np.mean(M0_par)
    # f_M0 = interp1d(x_grid, M0_par, fill_value='extrapolate')

    x = moments_file[:,0]
    a = moments_file[:,-1][0]
    # dc = f_M0(x)
    dc = moments_file[:,2]
    v = moments_file[:,5]
    return a, x, dc, v

# j = 10
a_0, x, dc_0, v_0 = extract_fields(path, 0)
a_1, x, dc_1, v_1 = extract_fields(path, 12)
a_2, x, dc_2, v_2 = extract_fields(path, 23)
a_3, x, dc_3, v_3 = extract_fields(path, 35)

# for j in range(0, 34):
#     fig, ax = plt.subplots()
#     ax.set_title(r'$a = {}$'.format(a_0), fontsize=14)
#     ax.set_ylabel(r'$1+\delta$', fontsize=14)
#     ax.plot(x, dc_0, lw=2, c='b')
#     ax.set_xlim(0.4, 0.45)
#     ax.set_ylim(-5, 100)
#
#     ax.minorticks_on()
#     ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
#     ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
#     ax.ticklabel_format(scilimits=(-2, 3))
#     ax.yaxis.set_ticks_position('both')
#     # ax.grid(lw=0.2, ls='dashed', color='grey')
#     # plots_folder = 'test/paper_plots'
#     # savename = 'den_IC'
#     plt.savefig('../plots/test/dc_{0:03d}.png'.format(j), bbox_inches='tight', dpi=300)
#     plt.close()


# a_1 = 1.93
a_list = [a_0, a_1, a_2 , a_3]
den_list = [dc_0, dc_1, dc_2, dc_3]
vel_list = [v_0, v_1, v_2, v_3]

print(a_list)


#density plot
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'\texttt{sim\_k\_1\_11}', fontsize=16, y=0.9, usetex=True)

for i in range(4):
    ax[i].plot(x, np.log(den_list[i]), c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
    # ax[i].legend(fontsize=14, loc=1)
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.885, y=0.865)
    ax[i].set_ylabel(r'$\mathrm{log}\;(1+\delta)$', fontsize=20)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
plt.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/den_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)
# plt.savefig('../plots/test/den_ev.png', bbox_inches='tight', dpi=300)
plt.close()

#velocity plot
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1, 1]})
fig.suptitle(r'\texttt{sim\_k\_1\_11}', fontsize=16, y=0.9, usetex=True)

for i in range(4):
    ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
    # ax[i].legend(fontsize=14, loc=1)
    ax[i].set_title('a = {}'.format(np.round(a_list[i], 3)), x=0.885, y=0.865)
    ax[i].set_ylabel(r'$\bar{v}\;[\mathrm{km\,s}^{-1}]$', fontsize=20)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[i].yaxis.set_ticks_position('both')
fig.align_labels()
ax[3].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
plt.subplots_adjust(hspace=0)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/vel_ev.pdf', bbox_inches='tight', dpi=300, pad_inches=0.3)
# plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
plt.close()

# #density plot
# fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# for i in range(3):
#     ax[i].plot(x, np.log(den_list[i]+1), c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
#     ax[i].legend(fontsize=14, loc=1)
#     ax[i].set_ylabel(r'$\mathrm{log}\;(1+\delta)$', fontsize=18)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#     ax[i].yaxis.set_ticks_position('both')
# fig.align_labels()
# ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
# plt.subplots_adjust(hspace=0)
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/den_ev.pdf', bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/den_ev.png', bbox_inches='tight', dpi=300)
# plt.close()
#
#
# #velocity plot
# fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1, 1]})
# for i in range(3):
#     ax[i].plot(x, vel_list[i], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i], 3)))
#     ax[i].legend(fontsize=14)
#     ax[i].set_ylabel(r'$\bar{v}\;[\mathrm{km\,s}^{-1}]$', fontsize=18)
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#     ax[i].yaxis.set_ticks_position('both')
#
# fig.align_labels()
# ax[2].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
# plt.subplots_adjust(hspace=0)
# # plt.show()
# plt.savefig('../plots/test/new_paper_plots/vel_ev.pdf', bbox_inches='tight', dpi=300)
# # plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
# plt.close()
# fig, ax = plt.subplots(2, 2, figsize=(12,10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
#
# ax[0,0].plot(x, dc_in, c='b', lw=2, label=r'$\delta_{\mathrm{in}}$')
# ax[0,0].set_ylabel(r'$1+\delta(x)$', fontsize=20)
#
# ax[0,1].plot(x, dc_fi, c='b', lw=2, label=r'$1+\delta_{\mathrm{fi}}$')
# # ax[0,1].set_ylabel(r'$1+\delta(x)$', fontsize=20)
#
# ax[1,0].plot(x, v_in, c='b', lw=2, label=r'$v_{\mathrm{in}}$')
# ax[1,0].set_ylabel(r'$v(x)$', fontsize=20, )
#
# ax[1,1].plot(x, v_fi, c='b', lw=2, label=r'$v_{\mathrm{fi}}$')
# # ax[1,1].set_ylabel(r'$v(x)$', fontsize=20)
#
# for i in range(2):
#     ax[i,0].set_title(r'$a = {}$'.format(a_in), fontsize=20)
#     ax[i,1].set_title(r'$a = {}$'.format(a_fi), fontsize=20)
#     ax[i,1].yaxis.set_label_position('right')
#
#     for j in range(2):
#         ax[i,j].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=20)
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         ax[i,j].yaxis.set_ticks_position('both')
#
# # ax.ticklabel_format(scilimits=(-2, 3))
# # # ax.grid(lw=0.2, ls='dashed', color='grey')
# # plt.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plots_folder = 'test/paper_plots'
# savename = 'den_v_plots'
# plt.tight_layout()
# # plt.savefig('../plots/test/in.png', bbox_inches='tight', dpi=300)
# plt.subplots_adjust(hspace=0.25)
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()



# fig, ax = plt.subplots()
#
# ax.set_title(r'$a = {}$'.format(a), fontsize=20)
# ax.set_ylabel(r'$v(x)[\mathrm{km}\;\mathrm{s}^{-1}]$', fontsize=20)
# ax.plot(x, C1_nbody, lw=2, c='b')
# ax.minorticks_on()
# ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
# ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
# ax.ticklabel_format(scilimits=(-2, 3))
# # ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# plots_folder = 'test/paper_plots'
# savename = 'vel_IC'
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()
