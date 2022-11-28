#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functions import smoothing, spectral_calc, SPT_real_tr, read_density
from scipy.interpolate import interp1d
from zel import initial_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Nfiles = 51
# j = 0
def vel_ext(j, path):
    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    print('a = ', a)
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]
    m = int(v_nbody.size / 2)
    g = 6000
    # fig, ax = plt.subplots()
    # ax.set_title(r'$a = {}$'.format(a))
    # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
    # ax.set_ylabel(r'$v\;[\mathrm{km\,s}^{-1}]$', fontsize=14)
    #
    # # ax.scatter(x_nbody, v_nbody, c='k', s=0.01)#, label=r'$N-$body')
    # ax.plot(x_nbody[m-g:m+g], v_nbody[m-g:m+g], c='b')#, label=r'$N-$body')
    # vol = x_nbody[m+g] - x_nbody[m-g]
    # ax.text(0.57, 0.88, 'Size of halo = {} Mpc'.format(np.round(vol, 6)), transform=ax.transAxes)
    # # print(vol)
    #
    # # plt.legend()
    # ax.tick_params(axis='both', which='both', direction='in')
    # ax.ticklabel_format(scilimits=(-2, 3))
    # ax.grid(lw=0.2, ls='dashed', color='grey')
    # ax.yaxis.set_ticks_position('both')
    # ax.minorticks_on()
    #
    # plt.savefig('../plots/test/new_paper_plots/ps_1halo/phase_space_{0:03d}.png'.format(j+1), bbox_inches='tight', dpi=150)
    # plt.close()
    # # plt.show()
    return a, x_nbody[m-g:m+g], v_nbody[m-g:m+g]


# for j in range(Nfiles):
#     vel_ext(j, /path)
nums = [0, 13, 23, 35]
x_list, vel_list, a_list = [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]]

sol_0 = vel_ext(nums[0], path)
sol_1 = vel_ext(nums[1], path)
sol_2 = vel_ext(nums[2], path)
sol_3 = vel_ext(nums[3], path)

a_list = [[sol_0[0], sol_1[0]], [sol_2[0], sol_3[0]]]
x_list = [[sol_0[1], sol_1[1]], [sol_2[1], sol_3[1]]]
vel_list = [[sol_0[2], sol_1[2]], [sol_2[2], sol_3[2]]]

#velocity plot
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
fig.suptitle(r'\texttt{sim\_k\_1\_11}', fontsize=20, y=0.925, usetex=True)

for i in range(2):
    ax[i][0].set_ylabel(r'$v\;[\mathrm{km\,s}^{-1}]$', fontsize=18)
    ax[i][1].set_ylabel(r'$v\;[\mathrm{km\,s}^{-1}]$', fontsize=18)
    ax[i][1].yaxis.set_label_position('right')

    ax[1][i].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
    ax[i,1].tick_params(labelleft=False, labelright=True)
    for j in range(2):
        ax[i][j].plot(x_list[i][j], vel_list[i][j], c='b', lw=1.5, label=r'$a = {}$'.format(np.round(a_list[i][j], 3)))
        # ax[i].legend(fontsize=14, loc=1)
        ax[i][j].set_title('a = {}'.format(np.round(a_list[i][j], 3)), x=0.85, y=0.865, fontsize=16)
        ax[i][j].minorticks_on()
        ax[i][j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
        ax[i][j].yaxis.set_ticks_position('both')

fig.align_labels()
plt.subplots_adjust(hspace=0, wspace=0)
# plt.show()
plt.savefig('../plots/test/new_paper_plots/phase_space.pdf', bbox_inches='tight', dpi=300)#, pad_inches=0.3)
# plt.savefig('../plots/test/new_paper_plots/phase_space.png', bbox_inches='tight', dpi=150)
plt.close()

# plt.savefig('../plots/test/vel_ev.png', bbox_inches='tight', dpi=300)
