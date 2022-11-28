#!/usr/bin/env python3
"""
This code solves the Schrodinger-Poisson system
created: Thu Jan 23 15:00
@author: mandar
"""

import numpy as np
import h5py as hp
import os
import time

def pot(psi, V, dt):
    """Potential Half-Step"""
    return np.exp(-1j * V * dt / 2) * psi

def kin(psi, k, dt, kappa):
    """Kinetic Step"""
    return np.fft.ifft(np.exp(-1j * (k ** 2) * kappa * dt / 2) * np.fft.fft(psi))

def poisson_fft(psi, k, kappa):
    """Poisson solver"""
    # k[0] = 0
    den = 3 * ((np.abs(psi) ** 2) - 1) / (2 * kappa)
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

def time_ev(psi, k, a0, an, dx, m, h, H0, dt_max, loc, N_max, C, A, L):
    """The Propagator"""
    ti = time.process_time()
    a = a0
    eta = np.log(a)
    dt = dt_max
    dt_arr = []
    count = 0
    name = 0
    flag = 1
    params = [h, H0, m, L]
    filename = str(loc) + 'psi_{0:05d}.hdf5'.format(name)
    write_psi(filename, a, psi, params, A)

    while flag == 1:
        kappa = h / (m * H0 * np.sqrt(a))
        rho = np.abs(psi ** 2) - 1
        norm = np.trapz(rho, dx=dx, axis=0)
        V = poisson_fft(psi, k, kappa)
        # e = energy(psi, V, k, dx)

        psi = pot(psi, V, dt)
        psi = kin(psi, k, dt, kappa)
        psi = pot(psi, V, dt)

        # dt_V = (np.pi / (2 * np.max(V))).real
        # dt_T = (np.pi / (2 * (np.max(k) **2) * kappa)).real
        # dt = C * np.min([dt_V, dt_T, dt_max])
        dt = dt_max
        eta += dt

        a = np.exp(eta)
        dt_arr.append(a)
        count += 1

        if count == N_max:
            filename = str(loc) + 'psi_{0:05d}.hdf5'.format(name)
            name += 1
            count = 0
            write_psi(filename, a, psi, params, A) #e, norm, psi)
            print('writing a = {}'.format(a))
            flag = flagger(loc)
            if a > an:
                flag = 0
            else:
                None
        else:
            None

        print('\nsolved for a = {}'.format(a))
        print('\nmean density in the box is = {}'.format(np.mean(np.abs(psi**2) - 1)))
        print('the norm is {}'.format(norm))
        print('\nthe last time step was {}'.format(dt))

        # print('the energy is {}, norm is {}'.format(e, norm))

        if flag == 0:
            print('\nstopping run')

    tn = time.process_time()
    print('\nrun has finished. total run time = {}s'.format(tn))
    return dt_arr

def flagger(loc):
    if os.path.exists(str(loc) + 'stop'):
        os.remove(str(loc) + 'stop')
        return 0
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

def write_psi(filename, a, psi, params, A):
    with hp.File(filename, 'w') as hdf:
        hdf.create_dataset('A', data=A)
        hdf.create_dataset('a', data=a)
        hdf.create_dataset('psi', data=psi)
        hdf.create_dataset('params', data=params)
