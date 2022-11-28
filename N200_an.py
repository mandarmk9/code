#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
import matplotlib.pyplot as plt

from functions import *

filedir = 'N200'
fileext = '.hdf5'
j = 0

for j in range(0, 107, 20):
    filename = 'data_{0:03d}'.format(j)
    filepath = '/vol/aibn31/data1/mandar/code/'+filedir+'/'+filename+fileext
    print("saving snapshot {}".format(str(j)))

    file = h5py.File(filepath, mode='r')
    pos = np.array(file['/Positions'])
    vel = np.array(file['/Velocities'])
    header = file['/Header']
    a = header.attrs.get('a')
    L = header.attrs.get('L')
    N = int(header.attrs.get('Nx'))
    file.close()

    #Cosmology​
    H0 = 100
    dx = L / N
    q = np.arange(0, L, dx)
    d0 = 0.5
    den_in = -d0 * np.cos(2*np.pi*q/L)
    nd = np.abs((1 - a*den_in)**(-1))
    Phi_q = d0*(L/(2*np.pi))*np.sin(2*np.pi*q/L)
    Phi_t = a*Phi_q
    x = q + Phi_t
    for l in range(N):
        if x[l] >= L:
            x[l] -= L
        elif x[l] < 0:
            x[l] += L
    v = H0 * (np.sqrt(a)) * (Phi_q) #peculiar velocity
    z = (1 / a) - 1
    # x = eul_pos(q, A, a)
    # v = eul_vel(H0, q, A, a) #peculiar velocity

    fig, ax = plt.subplots()
    title = ax.text(0.05, 0.9, 'z = {}'.format(str(np.round(z, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom")
    ax.scatter(pos, vel, s=5, color='b', label='N-body')
    ax.plot(x, v, color='k', ls='dashed', label='zel')
    plt.legend()
    plt.savefig('/vol/aibn31/data1/mandar/plots/N200/vel_{}.png'.format(j))
    plt.close()
    #plt.show()
