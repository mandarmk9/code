#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions import spectral_calc
import matplotlib.pyplot as plt
import h5py as hp
import os

def spatial_advection(f, kx, v, dt):
    return np.real(np.fft.ifft(np.fft.fft(f, axis=1) * np.exp(-1j * kx[None, :] * v[:, None] * dt / 2), axis=1))

def velocity_advection(f, kv, E, dt):
    return np.real(np.fft.ifft(np.fft.fft(f, axis=0) * np.exp(1j * kv[:, None] * E[None, :] * dt), axis=0))

def poisson_solver(rho, kx):
    kx[0] = 1
    V = np.fft.fft(rho)
    V[0] = 0
    V /= - kx ** 2
    return np.real(np.fft.ifft(V))

def field_solver(f, kx, dv, kappa):
    f_dp = np.trapz(f, dx=dv, axis=0)
    rho = kappa * (f_dp - 1)#np.mean(f_dp))
    E = np.real(spectral_calc(rho, kx, o=1, d=1))
    return E

def time_step(f0, x, p, H0, m, a0, an, da_max):
    dx = x[1] - x[0]
    dv = p[1] - p[0]
    kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
    kv = np.fft.fftfreq(p.size, dv) * 2.0 * np.pi
    a = a0
    j = 0
    fn = f0
    flag = True
    while flag == True:
        print('\na = ', a)
        kappa = (3 * (H0) * m) / (2 * np.sqrt(a))
        vel = p / (m * H0 * (a ** (3 / 2)))
        # da_int = dx / np.max(vel)
        da = da_max #np.min([da_int, da_max])

        fn = spatial_advection(fn, kx, vel, da)
        E = field_solver(fn, kx, dv, kappa)
        fn = velocity_advection(fn, kv, E, da)
        fn = spatial_advection(fn, kx, vel, da)

        norm = np.mean(np.trapz(fn, dx=dv, axis=0))
        f_dx_dp = np.trapz(np.trapz(fn, dx=dv, axis=0), dx=dx, axis=0)
        # fn /= f_dx_dp
        print('<f_dp> = ', norm)
        print('f_dx_dp = ', f_dx_dp)

        a += da
        j += 1

        if a >= an:
            flag = False

    return fn

def moment(F, P, dp, n):
    F1 = F * (P ** n)
    M = np.trapz(F1, dx=dp, axis=0)
    return M
