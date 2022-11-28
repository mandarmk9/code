#!/usr/bin/env python3
import time
t0 = time.time()

import h5py
import numpy as np
import matplotlib.pyplot as plt

from functions import *

Nfiles = 3 #number of data files
Nx = 420 #particle number along each dimension
fileext = '.hdf5'
data_filedir = '/vol/aibn31/data1/mandar/code/N{}/'.format(Nx)
Nfiles = 25
for j in range(20, Nfiles):
    print("saving snapshot {}".format(str(j+1)))
    # data_filename = 'ICs' #.format(j)
    data_filename = 'data_{0:03d}'.format(j)
    data_file = data_filedir + data_filename + fileext

    file = h5py.File(data_file, mode='r')
    pos = np.array(file['/Positions'])
    header = file['/Header']
    a = header.attrs.get('a')
    IDs = file['/IDs']
    L = int(header.attrs.get('L'))
    vel = np.array(file['/Velocities']) / np.sqrt(a)
    file.close()

    H0 = 100
    N_zel = 1024
    dx = L / N_zel
    A = [-0.01, 1, -0.5, 11]
    q = np.arange(0, 2*np.pi, dx)
    Psi_q = -Psi_q_finder(q, A)
    Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
    x = q + Psi_t #eulerian position
    v = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity

    # Lambda = 6
    # sm = (Lambda ** 2) / 2

    # dist = x - 0
    # dist[dist < 0] += L
    # dist[dist > L/2] = - L + dist[dist > L/2]

    # W_k_an = (np.sqrt(np.pi / sm)) * np.exp(- (k ** 2) / (4 * sm))

    # den_nbody = kde_gaussian(q, pos, sm, L)
    #
    # den_zel = es(x, a, A)[1]
    # dk_zel = np.fft.fft(den_zel) * dx
    # dk_zel *= W_k_an
    # dk2_zel = np.abs(dk_zel) ** 2 #/ (a**2)
    # # dx_zel = np.real(np.fft.ifft(dk_zel))
    #
    # dk_nbody = np.fft.fft(den_nbody) * ((L / N)) #factor of L/N for fourier space comparisons; use den_nbody when comparing in real space
    #
    # dk2_nbody = np.abs(dk_nbody) ** 2 #/ (a**2)
    #
    # print()

    fig, ax = plt.subplots()
    ax.set_title('a = {}'.format(str(np.round(a, 3))))
    # title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom")
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=12)
    ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=12)
    ax.plot(x, v, color='k', lw=1.5, ls='dashed', label='zel')
    ax.scatter(pos, vel, s=15, color='b', label='N-body')
    plt.legend()
    plt.savefig('/vol/aibn31/data1/mandar/plots/test/ps_{0:03d}.png'.format(j))
    plt.close()
