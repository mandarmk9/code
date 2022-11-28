#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py

from functions import * #, kde_gaussian, stencil, neighbors
# from adaptive_ts_sch import *
# from zel import eulerian_sampling
# from scipy.ndimage import gaussian_filter1d

# plt.style.use('simple_plt')


# loc2 = '/vol/aibn31/data1/mandar/data/sup_14/'  #sch_4096_high_amp_long_mode/
loc2 = '/vol/aibn31/data1/mandar/data/sch_multi_k/'  #sch_4096_high_amp_long_mode/

Nfiles = 61
a_list = np.empty(Nfiles)
for i in range(Nfiles):
    with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
        ls = list(hdf.keys())
        print(ls)
        a = np.array(hdf.get(str(ls[1])))
        print(a)
        a_list[i] = a

j = 12
gadget_files = '/vol/aibn31/data1/mandar/data/N64/'
file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
pos = np.array(file['/Positions'])
vel = np.array(file['/Velocities'])
header = file['/Header']
a_gadget = header.attrs.get('a')
N = int(header.attrs.get('Nx'))
file.close()


a_diff = np.abs(a_list - a_gadget)
a_ind = np.where(a_diff == np.min(a_diff))[0][0]
print(a_list[a_ind], a_gadget)

# a_ind = 0
# Nfiles = 1105
# for a_ind in range(1000, Nfiles, 20):
with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(a_ind), 'r') as hdf:
    ls = list(hdf.keys())
    a = np.array(hdf.get(str(ls[1])))
    psi = np.array(hdf.get(str(ls[3])))

L = 2 * np.pi
Nx = psi.size
dx = L / Nx

x = np.arange(0, L, dx)
k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

#parameters
h = 0.03
sigma_x = 0.025
H0 = 100
rho_0 = 27.755
m = rho_0 * dx
p = np.sort(k * h)
# print('m = ', a)
v_sch = p / (m * a)
X, P = np.meshgrid(x, p)
f_H = husimi(psi, X, P, sigma_x, h, L)
f_H /= np.max(f_H)

A = [-0.01, 1, -0.5, 11]
Psi_q = -Psi_q_finder(x, A, L)
Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
x_eul = x + Psi_t #eulerian position

#to ensure that the box is periodic
for l in range(N):
    if x_eul[j] >= L:
        x_eul[j] -= L
    elif x_eul[j] < 0:
        x_eul[j] += L

v_zel = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity

fig, ax = plt.subplots()
ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=20)
ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=20)
title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
plot2d_2 = ax.pcolormesh(x, v_sch, f_H, shading='auto', cmap='inferno')
ax.scatter(pos, vel, color='w', alpha=0.7, s=5, label='N-body')
ax.plot(x_eul, v_zel, color='r', ls='dashed', label='Zel')

ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
c = fig.colorbar(plot2d_2, fraction=0.15)
c.set_label(r'$f_{H}$', fontsize=20)

ax.set_ylim(-20, 20)
ax.legend(loc='upper right')
legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
frame = legend.get_frame()
plt.tight_layout()
frame.set_edgecolor('white')
frame.set_facecolor('black')
for text in legend.get_texts():
    plt.setp(text, color = 'w')
plt.savefig('/vol/aibn31/data1/mandar/plots/schr/ps/ps_{0:03d}.png'.format(a_ind))
plt.close()
# plt.show()

print('saving {} of {}'.format(a_ind, Nfiles))
