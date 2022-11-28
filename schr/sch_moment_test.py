#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings

from functions import *
from SPT import *
from zel import eulerian_sampling as es

loc2 = '/vol/aibn31/data1/mandar/data/sch_hfix_run17/'
filename_SPT = loc2 + 'spt_kern.hdf5'

Nfiles = 378

a_list = np.empty(Nfiles)
for i in range(Nfiles):
   with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
       ls = list(hdf.keys())
       a = np.array(hdf.get(str(ls[1])))
       a_list[i] = a

filename = 'sch_a_list.hdf5'
with h5py.File(filename, mode='w') as hdf:
    hdf.create_dataset('a_list', data=a_list)

print("Collecting Schroedinger scalefactors...")
with h5py.File(filename, mode='r') as hdf:
    ls = list(hdf.keys())
    a_list = np.array(hdf.get(str(ls[0])))

print("Matching to Gadget scalefactors...")
Nfiles_gad = 60

for j in range(17, Nfiles_gad-7):
    gadget_files = '/vol/aibn31/data1/mandar/code/N420/'
    file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
    pos = np.array(file['/Positions'])
    vel = np.array(file['/Velocities'])
    header = file['/Header']
    a_gadget = header.attrs.get('a')
    vel /= np.sqrt(a_gadget)
    N = int(pos.size)
    file.close()
    a_diff = np.abs(a_list - a_gadget)
    a_ind = np.where(a_diff == np.min(a_diff))[0][0]

    print(a_list[a_ind], a_gadget)

    with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(a_ind), 'r') as hdf:
        ls = list(hdf.keys())
        A = np.array(hdf.get(str(ls[0])))
        print(A)
        a = np.array(hdf.get(str(ls[1])))
        L, h, m, H0 = np.array(hdf.get(str(ls[2])))
        psi = np.array(hdf.get(str(ls[3])))
        print('a =', a, '\n')

    Nx = psi.size
    dx = L / Nx

    x = np.arange(0, L, dx)
    k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

    q = np.arange(0, L, L/N)
    k_pos = np.fft.fftfreq(q.size, q[1]-q[0]) * 2 * np.pi

    sigma_x = 100 * dx
    sigma_p = h / (2 * sigma_x)
    sm = 1 / (4 * (sigma_x**2))

    dist = x - 0
    dist[dist < 0] += L
    dist[dist > L/2] = - L + dist[dist > L/2]

    W_x = np.exp(-sm * ((dist)**2))
    W_k_an = np.exp(- (k ** 2) / (4 * sm))
    W_k_pos = np.exp(- (k_pos ** 2) / (4 * sm))

    #Husimi Moments
    psi_star = np.conj(psi)
    grad_psi = spectral_calc(psi, k, o=1, d=0)
    grad_psi_star = spectral_calc(np.conj(psi), k, o=1, d=0)
    lap_psi = spectral_calc(psi, k, o=2, d=0)
    lap_psi_star = spectral_calc(np.conj(psi), k, o=2, d=0)

    MW_0 = np.abs(psi ** 2)
    MH_0_k = np.fft.fft(MW_0) * W_k_an
    MH_0 = np.real(np.fft.ifft(MH_0_k))

    MW_1 = (1j * h) * ((psi * grad_psi_star) - (psi_star * grad_psi))
    MH_1_k = np.fft.fft(MW_1) * W_k_an
    MH_1 = np.real(np.fft.ifft(MH_1_k))

    MW_2 = - ((h**2) / 2) * ((lap_psi * psi_star) - (2 * grad_psi * grad_psi_star) + (psi * lap_psi_star))
    MH_2_k = np.fft.fft(MW_2) * W_k_an
    MH_2 = np.real(np.fft.ifft(MH_2_k))

    CH_1 = MH_1 / MH_0
    v_pec = CH_1 / (m * a)

    Psi_q = -Psi_q_finder(x, A)
    Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
    x_eul = x + Psi_t #eulerian position
    v_zel_us = H0 * np.sqrt(a) * Psi_q #peculiar velocity
    v_zel = np.real(np.fft.ifft(np.fft.fft(v_zel_us) * W_k_an))
    vel2 = np.real(np.fft.ifft(np.fft.fft(vel) * W_k_pos))

    fig, ax = plt.subplots()
    ax.set_title(r'a = {}'.format(str(np.round(a, 3))))
    ax.grid(linewidth=0.15, color='gray', linestyle='dashed')

    ax.plot(x, v_pec, c='b', lw=2, label='Sch')
    ax.plot(x_eul, v_zel, c='k', ls='dashed', lw=2, label='Zel')
    ax.scatter(q, vel2, c='r', s=25, label='N-body')


    ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
    ax.set_xlabel(r'x$\,$[$h^{-1} \; \mathrm{Mpc}$]', fontsize=12)
    plt.legend(fontsize=10, loc=2, bbox_to_anchor=(1,1))

    print('saving...')
    plt.savefig('/vol/aibn31/data1/mandar/plots/test/vel_{0:03d}.png'.format(a_ind), bbox_inches='tight', dpi=120)
    plt.close()

tn = time.time()
print("Done! Finished in {}s".format(np.round(tn-t0, 3)))
