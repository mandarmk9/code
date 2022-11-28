#!/usr/bin/env python3
"""
This code solves the Schrodinger-Poisson system
created: Thu Jan 23 15:00
@author: mandar
"""

import numpy as np
import h5py as hp
import os

from functions import spectral_calc

def pot(psi, V, dt, h, m):
    """Potential Half-Step"""
    return np.exp(-1j * m * V * dt / h) * psi

def kin(psi, k, dt, h, m):
    """Kinetic Step"""
    return np.fft.ifft(np.exp(-1j * (k ** 2) * h * dt / (4 * m)) * np.fft.fft(psi))

def poisson_fft(psi, k):
    """Poisson solver"""
    H0 = 100
    den = 3 * (H0**2) * ((np.abs(psi) ** 2) - 1) / 2
    V = np.fft.fft(den)
    V[0] = 0
    V[1:] /= -k[1:] ** 2
    return np.fft.ifft(V)

def phase(nd, k):
    H0 = 100
    V = np.fft.fft(nd)
    V[0] = 0
    V[1:] /= -k[1:] ** 2
    return (np.fft.ifft(V)) * H0

def time_ev(psi, k, t0, dt, tn, m, h, H0, loc, N_out=100):
    """The Propagator"""
    x = np.arange(0, 2*np.pi, (2*np.pi)/k.size)
    t = t0
    name = 0
    count = 0
    flag = 1
    write_out(loc, name, t, psi)

    while flag == 1:

        #kinetic half-step; eta increased by Δη/4
        psi = kin(psi, k, dt, h, m)
        t += dt / 4
        V = poisson_fft(psi, k)

        #potential half-step; eta increased by Δη/2 (in total by Δη/4 + Δη/2)
        psi = pot(psi, V, dt, h, m)
        t += dt / 2

        #kinetic half-step; eta increased by Δη/4 (in total by Δη)
        psi = kin(psi, k, dt, h, m)
        t += dt / 4

        count += 1
        if count == N_out:
            name += 1
            flag = write_out(loc, name, t, psi)
            count = 0
        if t > tn:
            flag = 0

        print('Solved for t = {}'.format(t))
        print('The last time step was {} \n'.format(dt))
        print('\nmean density in the box is = {}'.format(np.mean(np.abs(psi**2) - 1)))

        if flag == 0:
            print('Stopping run...')
            write_out(loc, name, t, psi)
            print('Done!')

def flagger(loc):
    if os.path.exists(str(loc) + 'stop'):
        os.remove(str(loc) + 'stop')
        return 0
    else:
        return 1

def write_out(loc, name, t, psi):
    print('Writing output file for t = {}'.format(t))
    filename = str(loc) + 'psi_{0:05d}.hdf5'.format(name)
    with hp.File(filename, 'w') as hdf:
        hdf.create_dataset('t', data=t)
        hdf.create_dataset('psi', data=psi)
    flag_loc = flagger(loc)
    if flag_loc == 0:
        return flag_loc
    else:
        return 1
