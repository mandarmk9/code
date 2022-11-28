#!/usr/bin/env python3
"""
This code solves the Schrodinger-Poisson system
created: Thu Jan 23 15:00
@author: mandar
"""

import numpy as np
import h5py as hp
import os

def pot(psi, V, dt):
    """Potential Half-Step"""
    return np.exp(-1j * V * dt) * psi

def kin(psi, k, dt, kap):
    """Kinetic Step"""
    return np.fft.ifft(np.exp(-1j * (k ** 2) * kap * dt / 4) * np.fft.fft(psi))

def poisson_fft(psi, k, kap):
    """Poisson solver"""
    den = 3 * ((np.abs(psi) ** 2) - 1) / (2 * kap)
    V = np.fft.fft(den)
    V[0] = 0
    V[1:] /= -k[1:] ** 2
    return np.fft.ifft(V)

def phase(nd, k, H0, a0):
    V = np.fft.fft(nd)
    V[0] = 0
    V[1:] /= -k[1:] ** 2
    return (np.fft.ifft(V) / a0) * (H0 * (a0 ** (3 / 2)))

def contrast(dc_in, a0):
    return np.abs(1 - a0 * dc_in) ** (- 1)

def kappa(h, m, H0, a):
    return h / (m * H0 * np.sqrt(a))

def time_ev(psi, k, a0, an, m, h, H0, dt, L=2*np.pi):
    """The Propagator"""
    a = a0
    eta = np.log(a)
    flag = 1
    dx = L / k.size
    x = np.arange(0, L, dx)
    while flag == 1:

        #kinetic half-step; eta increased by Δη/4
        psi = kin(psi, k, dt, kappa(h, m, H0, a))
        eta += dt / 4
        a = np.exp(eta)
        V = poisson_fft(psi, k, kappa(h, m, H0, a))

        #potential half-step; eta increased by Δη/2 (in total by Δη/4 + Δη/2)
        psi = pot(psi, V, dt)
        eta += dt / 2
        a = np.exp(eta)

        #kinetic half-step; eta increased by Δη/4 (in total by Δη)
        psi = kin(psi, k, dt, kappa(h, m, H0, a))
        eta += dt / 4
        a = np.exp(eta)

        if a > an:
            flag = 0

        print('Solved for a = {}'.format(a))
        print('mass was {}'.format(mass(psi, x)))
        print('The last time step was {} \n'.format(dt))
        if flag == 0:
            print('Stopping run...')
            print('Done!')

    return psi, a

def mass(psi, x):
    ma = 27.755 * np.trapz(np.abs(psi), x=x)
    return ma

def write_out(loc, name, a, psi, params, A):
    print('Writing output file for a = {}'.format(a))
    filename = str(loc) + 'psi_{0:05d}.hdf5'.format(name)
    with hp.File(filename, 'w') as hdf:
        hdf.create_dataset('a', data=a)
        hdf.create_dataset('psi', data=psi)
        hdf.create_dataset('params', data=params)
        hdf.create_dataset('ICs', data=A)
    flag_loc = flagger(loc)
    if flag_loc == 0:
        return flag_loc
    else:
        return 1
