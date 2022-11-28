#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fnmatch
import camb

from camb import model, initialpower
from adaptive_ts_sch import *
from zel import eulerian_sampling

L = 2 * np.pi
Nx = 2**13
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

#parameters
h = 0.001
H0 = 100
rho_0 = 27.755
m = rho_0 * dx

a0 = 0.05
an = 10

dt_max = 1e-2 #specify the maximum allowed time step in a (the actual time step depends on this and the Courant factor)

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[1/a0 - 1], kmax=2.0)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1, maxkh=2.0, npoints = 10)
# print(np.log(k))
k = np.log(kh)
print(k[1] - k[0])
print(k[6] - k[5])

plt.loglog(k, pk[0,:], color='k')
plt.xlabel('k/h Mpc')
plt.ylabel(r'$P_{\mathrm{lin}}(k)$')

plt.title('Linear matter power spectrum at z = {}'.format(z[0]))
plt.savefig('/vol/aibn31/data1/mandar/plots/P_lin.png', bbox_inches='tight', dpi=120)

# nd = eulerian_sampling(x, a0, A)[1] + 1
# phi_v = phase(nd, k, H0, a0)
#
# psi = np.zeros(x.size, dtype=complex)
# psi = np.sqrt(nd) * np.exp(-1j * phi_v * m / h)

# N_out = 750 #the number of time steps after which an output file is written
# N_out2 = 200 #the number of time steps after which an output file is written after a=4
# C = 1 #the courant factor for the run
# loc2 = '/vol/aibn31/data1/mandar/data/sch_hfix_run35/'
#
# r_ind = 0 #the index of the file you want to restart the run from
# restart = 0
# try:
#     print(os.listdir(loc2))
#     restart_file = loc2 + fnmatch.filter(os.listdir(loc2), 'psi_*.hdf5')[r_ind]
# except IndexError:
#     r_ind = 0
#     print("No restart file found, starting from the initial condition...\n")
#     restart = 0
#
# if restart != 0:
#     with h5py.File(restart_file, 'r') as hdf:
#         ls = list(hdf.keys())
#         a0 = np.array(hdf.get(str(ls[1])))
#         psi = np.array(hdf.get(str(ls[3])))
#         print(a0)
#         assert a0 < an, "Final time cannot be less than the restart time"
# else:
#     pass
#
# print("The solver will run from a = {} to a = {}".format(a0, an))
#
# time_ev(psi, k, a0, an, dx, m, h, H0, dt_max, loc2, N_out, C, A, r_ind, L, N_out2)
#
# tn = time.time()
# print("Run finished in {}s".format(np.round(tn-t0, 5)))
