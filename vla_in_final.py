#!/usr/bin/env python3
"""
This code tests Vlasov solvers using known analytical solutions / ICs for which the behaviour of the solution is known
"""
import time
tcode_0 = time.time()
import h5py
import numpy as np
import vla_solve_final as vp

import matplotlib.pyplot as plt

loc = '/vol/aibn31/data1/mandar/data/vlasov_tests/'
from functions import phase, husimi, spectral_calc, write_vlasov_ic, Psi_q_finder, vla_ic_plotter
from zel import eulerian_sampling

Lx = 2 * np.pi
Nx = 512
dx = Lx / Nx

x = np.arange(0, Lx, dx)
kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

da = 1e-2
a = np.arange(0.1, 0.5, da)
a0 = a[0]

h = 0.03
sigma_x = np.sqrt(h / 2) #* 2 #* dx
sigma_p = h / (2 * sigma_x)
H0 = 100

print(sigma_x, 1/sigma_x)
print(sigma_p, 1/sigma_p)
# print(sigma_x * sigma_p)

rho_0 = 27.755
m = rho_0 * dx

# p_max = 5
# dp = 2 * p_max / (Nx)
p = np.sort(kx*h)#np.arange(-p_max, p_max, dp)
dp = p[1] - p[0]
v = p / (m * a0)
A = [-0.5, 1, 0, 11]
nd = eulerian_sampling(x, a0, A)[1] + 1
# nd /= np.mean(nd)

phi_v = phase(nd, kx, H0, a0)

psi = np.zeros(Nx, dtype=complex)
psi[:] = np.sqrt(nd[:]) * np.exp(-1j * phi_v[:] * m / h)

X, P = np.meshgrid(x, p)
f0 = husimi(psi, X, P, sigma_x, h, Lx)
f0 /= np.mean(np.trapz(f0, dx=dp, axis=0))
f0 /= np.trapz(np.trapz(f0, dx=dp, axis=0), dx=dx, axis=0)

Psi_q = -Psi_q_finder(x, A)
v_zel = H0 * np.sqrt(a0) * (Psi_q) #peculiar velocity
x_zel = x + a0 * Psi_q


M0 = vp.moment(f0, P, dp, 0)
M1 = vp.moment(f0, P, dp, 1)
v_vla = M1 / M0 / (m * a0)
M0 /= np.mean(M0)
plt.plot(x, nd, label='zel')
plt.plot(x, M0, label='vla')
plt.legend()
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/M0_0.png')
plt.close()


plt.plot(x_zel, v_zel, label='zel')
plt.plot(x, v_vla, label='vla')
plt.legend()
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/M1_0.png')
plt.close()

Lv = 120
fig, ax = plt.subplots()
ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a0, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
plot2d_2 = ax.pcolormesh(x_zel, v, f0, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'$f_{V}$', fontsize=20)

ax.plot(x_zel, v_zel, c='r', lw=1.5, ls='dashed', label='Zel')
ax.set_ylim(-Lv, Lv)
ax.legend(loc='upper right')
legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
frame = legend.get_frame()
plt.tight_layout()
frame.set_edgecolor('white')
frame.set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/IC.png')
plt.close()


fn = vp.time_step(f0, x, p, H0, m, a[0], a[-1], da)
v_zel = H0 * np.sqrt(a[-1]) * (Psi_q) #peculiar velocity
x_zel = x + a[-1] * Psi_q
fig, ax = plt.subplots()
title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a[-1], 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
plot2d_2 = ax.pcolormesh(x_zel, v, fn, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
ax.plot(x_zel, v_zel, c='r', lw=1.5, ls='dashed', label='Zel')

c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'$f_{V}$', fontsize=20)

ax.set_ylim(-Lv, Lv)
ax.legend(loc='upper right')
legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
frame = legend.get_frame()
plt.tight_layout()
frame.set_edgecolor('white')
frame.set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/fn.png')
plt.close()


M0_n = vp.moment(fn, P, dp, 0)
M1_n = vp.moment(fn, P, dp, 1)
v_vla_n = M1_n / M0_n / (m * a[-1])
nd_n = eulerian_sampling(x, a[-1], A)[1] + 1
nd_n /= np.mean(nd_n)
M0_n /= np.mean(M0_n)

plt.plot(x_zel, nd_n, label='zel')
plt.plot(x_zel, M0_n, label='vla')
plt.legend()
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/M0_n.png')
plt.close()

plt.plot(x_zel, v_zel, label='zel')
plt.plot(x_zel, v_vla_n, label='vla')
plt.legend()
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/M1_n.png')
plt.close()

tcode_n = time.time()
print('Done! This file took {}s to run.'.format(tcode_n-tcode_0))
