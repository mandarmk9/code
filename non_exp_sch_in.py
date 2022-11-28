#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
t0 = time.time()
import numpy as np
import h5py

from non_exp_sch import *
from zel import eulerian_sampling

L = 2 * np.pi
Nx = 2**12
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

#parameters
h = 0.01
H0 = 100
rho_0 = 27.755
m = rho_0 * dx

a0 = 0.5
dt = 1e-3
an = 20

A = [-0.01, 1, -0.5, 11] #the 0th and 2nd elements are the two amplitudes, 1st and 3rd are the frequencies

nd = eulerian_sampling(x, a0, A)[1] + 1
phi_v = phase(nd, k)

psi = np.zeros(x.size, dtype=complex)
psi = np.sqrt(nd) * np.exp(-1j * phi_v * m / h)
N_out = 100 #the number of time steps after which an output file is written
loc = '/vol/aibn31/data1/mandar/data/mz_run3/'

print('mean = ', np.mean(np.abs(psi**2) - 1))
time_ev(psi, k, a0, dt, an, m, h, H0, loc, N_out)

tn = time.time()
print("Run finished in {}s".format(np.round(tn-t0, 5)))
