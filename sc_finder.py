#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib.pyplot import cm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# path = 'cosmo_sim_1d/sim_k_1_7/run1/'
# path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
path = 'cosmo_sim_1d/new_sim_k_1_11/run1/'

# path = 'cosmo_sim_1d/final_sim_k_1_11/run1/'

# path = 'cosmo_sim_1d/sim_k_1_15/run1/'

# a_list = []
# filename = path + '/particle_data.hdf5'
# file = h5py.File(filename, 'w')
# a_arr = file.create_group('Scalefactors')
#
# lag_coo = file.create_group('Lagrangian Coordinate')
# eul_coo = file.create_group('Eulerian Coordinate')
# vel = file.create_group('Velocities')
#
# for j in range(51):
#     nbody_filename = 'output_{0:04d}.txt'.format(j)
#     nbody_file = np.genfromtxt(path + nbody_filename)
#
#     a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
#     q = nbody_file[:,0]
#     x = nbody_file[:,-1]
#     v = nbody_file[:,2]
#
#     a_list.append(a)
#     lag_coo.create_dataset(str(j), data=q)
#     eul_coo.create_dataset(str(j), data=x)
#     vel.create_dataset(str(j), data=v)
#     print(a)
#
#
# a_arr.create_dataset('a', data=a_list)
# file.close()

filename = path + '/particle_data.hdf5'
file = h5py.File(filename, mode='r')
a_list = np.array(file['/Scalefactors']['a'])
eul_coo = file['/Eulerian Coordinate']
lag_coo = file['/Lagrangian Coordinate']
vel = file['/Velocities']

flips, flag_list = [], []
old_flip = 0
i1, i2 = 0, -1#204005, 204025 #245000, 255000 #

# for j in range(34, 36):
#     x2 = np.array(eul_coo[str(j)])[i1:i2]
#     print(x2[0], x2[1], x2[2])
#     mask = np.array([int((x2[i+1] > x2[i])) for i in range(x2.size-1)])
#     print(mask)

for j in range(51):#36):
    # print(j)
    x = np.array(eul_coo[str(j)])[i1:i2]
    q = np.array(lag_coo[str(j)])[i1:i2]
    # v = np.array(vel[str(j)])
    # print(x)
    mask = np.array([int((x[i+1] > x[i])) for i in range(x.size-1)])
    # print(mask)

    mask = [not((mask[l] == mask[l:l-8].any()) and (mask[l] == mask[l:l+8].any())) for l in range(x.size-1)]

    # fig, ax = plt.subplots()
    # ax.set_title(a_list[j])
    # ax.plot(mask)#, marker='o')
    # plt.show()
    # ax.plot(q, x)
    # plt.savefig('../plots/test/sc.png')
    # plt.close()

    new_flip = int((np.diff(mask)!=0).sum() / 2)
    print(j, a_list[j], new_flip)
    flips.append(new_flip)
    if old_flip - new_flip != 0:
        flag = True
    else:
        flag = False
    old_flip = new_flip
    flag_list.append(flag)

np.savetxt(fname=path+'/sc_flags.txt', X=flag_list, newline='\n')

# fig, ax = plt.subplots()
# ax.set_xlabel(r'$a$', fontsize=16)
# ax.set_ylabel(r'$x_{j}\;[h^{-1}\mathrm{Mpc}]$', fontsize=16)
# # ax.set_ylabel(r'$v_{j}\;[\mathrm{km}\,\mathrm{s}^{-1}]$', fontsize=16)
#
# m = 62500
# g = 500
# Nt = 51
# color = iter(cm.rainbow(np.linspace(0, 1, 2*g)))
# for m in range(m-g, m+g):
#     print(m)
#     # x0 = np.array([lag_coo[str(j)][m] for j in range(Nt)])
#     x0 = np.array([eul_coo[str(j)][m] for j in range(Nt)])
#     # v0 = np.array([vel[str(j)][m] for j in range(Nt)])
#
#     c = next(color)
#     ax.plot(a_list, x0, c=c, lw=2)
#     # ax.plot(x0, v0, c=c, lw=2)
#
# print(x0[0], x0[-1])
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
# plt.savefig('../plots/traj/v_vs_a.png', bbox_inches='tight', dpi=300)
# plt.close()
# # plt.show()
