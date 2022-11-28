#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import vla_solve as vp
import h5py as hp
from numpy.fft import fftshift, fft, ifft, fftfreq
from functions import phase, husimi, spectral_calc, write_vlasov_ic, Psi_q_finder, vla_ic_plotter
from zel import eulerian_sampling

Lx = 2 * np.pi
Nx = 8192
dx = Lx / Nx

x = np.arange(0, Lx, dx)
kx = fftfreq(x.size, dx) * 2.0 * np.pi

# Lv = 20
# Nv = 2049
# dv = Lv / Nx
# v = np.arange(-Lv, Lv, dv)

N_out = 1
a0 = 0.1
da = 1e-3
an = a0 + da#(100 * da)

h = 0.001
sigma_x = 10 * dx
sigma_p = h / (2 * sigma_x)

# print(sigma_x, 1/sigma_x)
# print(sigma_p, 1/sigma_p)
# print(sigma_x * sigma_p)

H0 = 100
rho_0 = 27.755
m = rho_0 * dx
# p = m * a0 * v
p = np.sort(kx * h)
v = p / (m * a0)

A = [-0.25, 1, 0, 11]
nd = eulerian_sampling(x, a0, A)[1] + 1
nd /= np.mean(nd)

phi_v = phase(nd, kx, H0, a0)

psi = np.zeros(Nx, dtype=complex)
psi[:] = np.sqrt(nd[:]) * np.exp(-1j * phi_v[:] * m / h)

X, V = np.meshgrid(x, p)
f = husimi(psi, X, V, sigma_x, h, Lx)

Psi_q = -Psi_q_finder(x, A)
v_zel = H0 * np.sqrt(a0) * (Psi_q) #peculiar velocity

run = '/vlasov_run1/'
vla_ic_plotter(f, x, v, v_zel, a0, run)
# loc = '/vol/aibn31/data1/mandar/data/vlasov_run1/'
# # write_vlasov_ic(f, x, p, H0, m, a0, da, an, loc, N_out, A)
# # vp.time_step(f, x, p, H0, m, a0, da, an, loc, N_out)

tn = time.time()
print('Finished run. Total time taken: {}s'.format(tn - t0))
