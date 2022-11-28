#!/usr/bin/env python3
"""
This code solves the Schrodinger-Poisson system
created: Thu Jan 23, 2021 15:00
@author: mandar
"""

import numpy as np
import h5py as hp
import os

from functions import spectral_calc

def pot(psi, V, dt):
    """Potential Half-Step"""
    return np.exp(-1j * V * dt) * psi

def kin(psi, k, dt, kap):
    """Kinetic Step"""
    return np.fft.ifft(np.exp(-1j * (k ** 2) * kap * dt / 4) * np.fft.fft(psi))

def poisson_fft(psi, k, L, kap):
    """Poisson solver"""
    den = 3 * ((np.abs(psi) ** 2) - 1) / (2 * kap)
    V = spectral_calc(den, L, o=2, d=1)
    return V

def kappa(h, H0, a):
    return h / (H0 * np.sqrt(a))

def time_ev(psi, L, a0, an, h, H0, dt_max, loc, N_out, C, A, r_ind=0, N_out2=50):
    """The Propagator"""
    a = a0
    eta = np.log(a)
    dt = dt_max
    count = 0
    name = r_ind
    flag = 1
    flag2 = 0
    params = [L, h, H0]
    Nx = psi.size
    x = np.arange(0, L, L/Nx)
    k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    write_out(loc, name, a, psi, params, A)

    while flag == 1:
        #kinetic half-step; eta increased by Δη/4
        psi = kin(psi, k, dt, kappa(h, H0, a))
        # eta += dt / 4
        # a = np.exp(eta)
        V = poisson_fft(psi, k, L, kappa(h, H0, a))

        #potential half-step; eta increased by Δη/2 (in total by Δη/4 + Δη/2)
        psi = pot(psi, V, dt)
        # eta += dt / 2
        # a = np.exp(eta)

        #kinetic half-step; eta increased by Δη/4 (in total by Δη)
        psi = kin(psi, k, dt, kappa(h, H0, a))
        eta += dt
        a = np.exp(eta)

        if a >= 1:
            N_out = N_out2
            count *= flag2
            flag2 = 1

        # dt_V = (np.pi / (np.max(V))).real
        # dt_T = (np.pi / (4 * (np.max(k) **2) * kappa(h, H0, a))).real
        dt = dt_max#C * np.min([dt_V, dt_T, dt_max])

        print(N_out)
        count += 1
        print(count)
        if count == N_out:
            name += 1
            flag = write_out(loc, name, a, psi, params, A)
            count = 0
            print(count)
        if a > an:
            flag = 0

        print('Solved for a = {}'.format(a))
        print('The last time step was {}'.format(dt))
        print('mean density in the box is = {}\n'.format(np.mean(np.abs(psi**2) - 1)))

        if flag == 0:
            print('Stopping run...')
            write_out(loc, name, a, psi, params, A)
            print('Done!')

def flagger(loc):
    if os.path.exists(str(loc) + 'stop'):
        os.remove(str(loc) + 'stop')
        return 0
    else:
        return 1

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

def energy(psi, V, k, dx):
    """Energy Calculation"""
    psi_r = psi
    psi_k = np.fft.fft(psi_r)
    psi_c = np.conj(psi_r)
    E_k = 0.5 * psi_c * np.fft.ifft((k ** 2) * psi_k)
    E_r = psi_c * V * psi_r
    E_f = sum(E_r + E_k).real
    E = E_f * dx
    return E
