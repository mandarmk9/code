#!/usr/bin/env python3
"""
This code tests Vlasov solvers using known analytical solutions / ICs for which the behaviour of the solution is known
"""
import time
tcode_0 = time.time()
import h5py
import numpy as np
import vla_solve_static as vp
import vla_solve_old as vp_old

import matplotlib.pyplot as plt

loc = '/vol/aibn31/data1/mandar/data/vlasov_tests/'
from functions import phase, husimi, spectral_calc, write_vlasov_ic, Psi_q_finder, vla_ic_plotter
from zel import eulerian_sampling

Lx = 2 * np.pi
Nx = 512
dx = Lx / Nx

x = np.arange(0, Lx, dx)
kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

# Lv = 20
# Nv = 2049
# dv = Lv / Nx
# v = np.arange(-Lv, Lv, dv)

N_out = 1e10
H0 = 100
t0 = 2.1e-4
dt_max = 1e-5
tn = t0 + 250*dt_max
a0 = (3*H0*t0/2) ** (2/3)

h = 0.05
sigma_x = np.sqrt(h / 2)#10 * dx
sigma_p = h / (2 * sigma_x)

print(sigma_x, 1/sigma_x)
print(sigma_p, 1/sigma_p)
# print(sigma_x * sigma_p)

rho_0 = 27.755
m = rho_0 * dx

p_max = 10
dp = 2 * p_max / (Nx)
p = np.arange(-p_max, p_max, dp)
v = p / (m * a0)
A = [-0.1, 1, 0, 11]
nd = eulerian_sampling(x, a0, A)[1] + 1
nd /= np.mean(nd)

phi_v = phase(nd, kx, H0, a0)

psi = np.zeros(Nx, dtype=complex)
psi[:] = np.sqrt(nd[:]) * np.exp(-1j * phi_v[:] * m / h)

X, P = np.meshgrid(x, p)
f0 = husimi(psi, X, P, sigma_x, h, Lx)
f0 /= np.mean(np.trapz(f0, dx=dp, axis=0))

Psi_q = -Psi_q_finder(x, A)
v_zel = H0 * np.sqrt(a0) * (Psi_q) #peculiar velocity

M0 = vp.moment(f0, P, 0)
M1 = vp.moment(f0, P, 1)
v_vla = M1 / M0 / (m * a0)

plt.plot(x, v_zel, label='zel')
plt.plot(x, v_vla, label='vla')
plt.legend()
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/IC_comp.png')
# Lx = 2 * np.pi
# Nx = 512
# dx = Lx / Nx
#
# x = np.arange(0, Lx, dx)
# kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
#
# Lv = 5
# Nv = 512
# dv = Lv / Nx
# v = np.arange(-Lv, Lv, dv)
#
# X, P = np.meshgrid(x, v)
#
# a0 = 0.1
# da = 1/8#0.075
# an = 5
# N_out = int((an - a0) / da)

# A = 0.05
# n = 0.5
# T_rr = 2 * np.pi * dv / n
# # f0 = (A / np.sqrt(2 * np.pi)) * np.exp(- (P**2) / 2) * np.sin(n*X) #free-streaming
# # f0 = np.exp(- (P**2) / 2) * (1 + A*np.sin(n*X)) #landau damping
# f0 = ((2 * np.pi)**(-1/2)) * (P**2) * np.exp(-((P)**2) / 2) * (1 + A*np.cos(n*(X)))
Lv = 75
fig, ax = plt.subplots()
ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a0, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
plot2d_2 = ax.pcolormesh(x, v, f0, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'$f_{V}$', fontsize=20)

ax.plot(x, v_zel, c='r', lw=1.5, ls='dashed', label='Zel')
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

vp.time_step(loc, f0, x, v, H0=H0, m=m, t0=t0, dt_max=dt_max, tn=tn, N_out=N_out, save_dist=True)

# fn = (A / np.sqrt(2*np.pi)) * np.exp(- (P**2) / 2) * (np.cos(n*X) + n*an*np.sin(n*x))
j = 0
with h5py.File(loc + 'dist_{0:05d}.hdf5'.format(j), 'r') as hdf:
   ls = list(hdf.keys())
   print(ls)
   a = np.array(hdf.get(str(ls[0])))
   f = np.array(hdf.get(str(ls[1])))
   print('a = ', a)

v_zel = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity

fig, ax = plt.subplots()
title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
plot2d_2 = ax.pcolormesh(x, v, f, shading='auto', cmap='inferno')
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
ax.plot(x, v_zel, c='r', lw=1.5, ls='dashed', label='Zel')

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
plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/fn.png'.format(j))
plt.close()

# t = np.arange(a0, an+(da), da)
# f_old = vp_old.time_ev(f0, x, v, t)
#
# fig, ax = plt.subplots()
# ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
# ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
# title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(t[-1], 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
# plot2d_2 = ax.pcolormesh(x, v, f_old, shading='auto', cmap='inferno')
#
# ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
# c = fig.colorbar(plot2d_2, fraction=0.15)
# c.set_label(r'$f_{V}$', fontsize=20)
#
# ax.set_ylim(-Lv, Lv)
# ax.legend(loc='upper right')
# legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
# frame = legend.get_frame()
# plt.tight_layout()
# frame.set_edgecolor('white')
# frame.set_facecolor('black')
# for text in legend.get_texts():
#     plt.setp(text, color = 'w')
# plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/fn_old.png'.format(j))
# plt.close()



# #reading the test results
# Nfiles = 360
# rho_list = np.zeros(Nfiles)
# a_list = np.zeros(Nfiles)
# for j in range(1, Nfiles+1):
#    with h5py.File(loc + 'moments_{0:05d}.hdf5'.format(j), 'r') as hdf:
#       ls = list(hdf.keys())
#       M0 = np.array(hdf.get(str(ls[0])))
#       a = np.array(hdf.get(str(ls[-2])))
#       print('a = ', a)
#
#    rho_list[j-1] = np.max(M0)
#    a_list[j-1] = a
# fig, ax = plt.subplots()
# ax.set_title('Recurrence in the free-streaming Vlasov equation')
# ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
# ax.plot(a_list, rho_list, c='b', lw=2)
# ax.set_ylabel(r'$\rho_{\mathrm{max}}$')
# ax.set_yscale('log')
# ax.set_xlabel('t')
# ax.axvline(T_rr, c='k', ls='dashed', lw=2)
# plt.savefig('/vol/aibn31/data1/mandar/plots/vlasov_tests/free_streaming.png')


tcode_n = time.time()
print('Done! This file took {}s to run.'.format(tcode_n-tcode_0))
