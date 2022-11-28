#!/usr/bin/env python3
"""This file writes the Gadget ICs and stores them in hdf5 files.
author: @mandarmk9
"""
import time
t0 = time.time()
import multiprocessing as mp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools

from hdf5_write_ic import write
from functions import *

h = 0.7 #hubble constant in units of 100 km/s/Mpc

#box
L = 2 * np.pi #boxsize in Mpc
Nx = 256 #particle number (x)
dx = L / Nx #particle separation (x)
q = np.arange(0, L, dx) #lagrangian coordinate

#cosmology
a0 = 0.1 #initial scale factor
z = (1 / a0) - 1 #initial redshift
Omega0 = 1.0 #matter density at z=0
OmegaL = 0.0 #dark energy density at z=0
H0 = 100 #hubble constant in km/s/Mpc
rho_0 = 27.755

#the following calculation for mass only works with a 3D box
#populated with N**3 particles if N is the particle number in 1D
m = rho_0 * (dx**3)

# #ICs
# q = np.arange(0, L, dx) #lagrangian position

# d0 = 0.5
# den_in = -d0 * np.cos(2 * np.pi * q / L) #this is the initial overdensity \delta(q)
# Psi_q_old = d0 * (L / (2 * np.pi)) * np.sin(2 * np.pi * q / L)
# x_old = q + a0 * Psi_q_old
# v_old = H0 * np.sqrt(a0) * (Psi_q_old)#eul_vel(H0, q_mod, A, a0)
# A = [-0.5, 1, 0, 1]

# A = [-0.01, 1, -0.5, 11]
A = [-0.25, 1, 0, 11]

Psi_q = -Psi_q_finder(q, A)
Psi_t = a0 * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
x = q + Psi_t #eulerian position

#to ensure that the box is periodic
for j in range(Nx):
    if x[j] >= L:
        x[j] -= L
    elif x[j] < 0:
        if np.abs(x[j]) > 1e-10:
            x[j] += L

v = H0 * np.sqrt(a0) * (Psi_q) #peculiar velocity

# fig, ax = plt.subplots()
# title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a0, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom")
# # ax.set_ylim(-10, 10)
# ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
# ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
# # ax.plot(x_old, v_old, lw=2, color='k', ls='dashed', label='zel_old')
# ax.plot(x, v, color='b', lw=2, label='zel')
# plt.legend()
# plt.savefig('/vol/aibn31/data1/mandar/plots/test/test.png')
# plt.close()

u = v / np.sqrt(a0) #particle velocity for Gadget-2
Nt = np.array([0, Nx ** 3, 0, 0, 0, 0]) #total particle number
mass = np.array([0, m, 0, 0, 0, 0]) #mass array

# # initialise 3D box
# pos = []
# vel = []
# for i in range(Nx):
#     for j in range(Nx):
#         for k in range(Nx):
#             pos.append([x[i], q[j], q[k]])
#             vel.append([u[i], 0, 0])
#     print('writing {} of {}'.format(i, Nx))

# # write the positions and velocity according to the plane-parallel ICs into arrays
col1 = []
pcol2 = []
for i in range(Nx):
    col1.extend(list(np.repeat(x[i], Nx**2)))
    pcol2.extend(list(np.repeat(q[i], Nx)))
col2 = list(pcol2) * Nx
col3 = list(q) * Nx**2
pos_table = np.dstack((col1, col2, col3))[0]

vel_col1 = np.array(list(itertools.chain.from_iterable(itertools.repeat(i, Nx**2) for i in u)))
col23 = np.zeros(Nx**3)
vel_table = np.dstack((vel_col1, col23, col23))[0]

# # write to file
# #no. of hdf5 files per snapshot
NumFiles = 80
parts = int((Nx ** 3) / NumFiles)

params = [L, h, mass, Nt, OmegaL, Omega0, z, a0]

filedir = '/vol/aibn31/data1/mandar/gadget_runs/ICs/N480/'

def writer(j, pos_table, vel_table, params, parts, NumFiles, filedir):
    pp = pos_table[j * parts : (j + 1) * parts]
    vv = vel_table[j * parts : (j + 1) * parts]
    IDs = np.arange(int(j)*pp.shape[0], int(j+1)*pp.shape[0])  #initialise particle IDs
    NumPart_ThisFile = np.zeros(6, dtype=int)
    NumPart_ThisFile[1] = int(IDs.size)
    filename = 'n{}.{}.hdf5'.format(Nx, j)
    filepath = filedir + filename
    w = write(pp, vv, IDs, params[0], params[1], params[2], NumFiles, NumPart_ThisFile,
        params[3], params[4], params[5], params[6], params[7], filepath)
    w.write_file()

    print('writing part {} of {}'.format(j+1, NumFiles))
    print('Number of particles in this file: {}'.format(NumPart_ThisFile[1]))

processes = [mp.Process(target=writer, args=(j, pos_table, vel_table, params, parts, NumFiles, filedir)) for j in range(NumFiles)]
for p in processes:
    p.start()
for p in processes:
    p.join()

tn = time.time()
print('ICs are written to file, total time taken: {}s'.format(tn - t0))
