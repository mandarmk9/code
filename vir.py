#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functions import smoothing, spectral_calc, SPT_real_tr, read_density, read_sim_data
from scipy.interpolate import interp1d
from zel import initial_density
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# path = 'cosmo_sim_1d/phase_full_run1/'
# path = 'cosmo_sim_1d/sim_k_1 (copy)/run1/'
path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Nfiles = 51
H0 = 100
# Lambda = 3 * (2* np.pi)
# kind = 'sharp'
# j = 50
# a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
# plt.plot(x, tau_l)
# plt.show()


q_list, a_list, C1_list = [], [], []
for j in range(Nfiles):
    nbody_filename = 'output_{0:04d}.txt'.format(j)
    nbody_file = np.genfromtxt(path + nbody_filename)
    x_nbody = nbody_file[:,-1]
    v_nbody = nbody_file[:,2]

    moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
    moments_file = np.genfromtxt(path + moments_filename)
    a = moments_file[:,-1][0]
    x = moments_file[:,0]
    L = 1.0
    k = np.fft.ifftshift(2.0 * np.pi * np.arange(-x.size/2, x.size/2))
    M0 = moments_file[:,2]
    C0 = moments_file[:,3]
    M1 = moments_file[:,4]
    C1 = moments_file[:,5]
    M2 = moments_file[:,6]
    C2 = moments_file[:,7]

    #solve Poisson to get the potential \phi
    rhs = (3 * H0**2 / (2 * a)) * (M0-1) #using the hierarchy δ here
    phi = spectral_calc(rhs, L, o=2, d=1)
    grad_phi = spectral_calc(phi, L, o=1, d=0) #this is the gradient of the unsmoothed potential ∇ϕ

    m = int(x.size / 2)
    g = 6000 #set to 6000 for the innermost halo

    #define the kinetic energy 'scalar' inside the defined region
    T = sum(C2[m-g:m+g])
    V = sum((M0[m-g:m+g] * grad_phi[m-g:m+g]) * x[m-g:m+g]) / 2
    q = T / V
    q_list.append(q)
    a_list.append(a)
    C1_list.append(C1[m])#sum(C1[m-g:m+g]))
    print('a = ', a)

fig, ax = plt.subplots()
ax.plot(a_list, C1_list, lw=1.5, c='b', label=r'$q$')
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$C_{1}$')
ax.axhline(0, c='k', lw=1, ls='dashed', label='virialised')
plt.legend()
plt.show()


fig, ax = plt.subplots()
ax.plot(a_list, q_list, lw=1.5, c='b', label=r'$q$')
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$q = T/V$')
ax.axhline(1, c='k', lw=1, ls='dashed', label='virialised')
plt.legend()
plt.show()
