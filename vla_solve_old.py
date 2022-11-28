#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from functions import spectral_calc

def generate2d(x1, x2, f, indexing='xy'):
    """Generates a 2D distribution from two 1D arrays, with the
    functional form specified by f. Indexing (str): xy (default) for cartesian,
    and ij for matrix.
    """
    [X1, X2] = np.meshgrid(x1, x2, indexing=indexing)
    F = f(X1, X2)
    return F

def spatial_shift(f, v, kx, Nx, Nv, dt):
    """f is a 2D matrix of shape (M, N); kx represent the fourier modes of the variable
    x, while v is the other variable. Note that Nv = len(v), Nx = len(x). This function takes
    the matrix f into the fourier space of x, and interpolates f to a shifted position
    x' : (x - x') = v*dt for a full step, or = v*dt/2 for a half step. The result is taken back
    to real space.
    """
    f_kx = np.zeros([Nv, Nx], dtype=float)
    for i in range(0, Nv):
        f_kx[i] = np.real(np.fft.ifft(np.fft.fft(f[i,:])*np.exp(-1j * kx * v[i] * dt)))
    return f_kx

def field_solver(f, kx, dx, dv, Nx, H0, a, ax=0):
    """Obtains the mass/number density by cumulative integration of f. Integrates the
    resulting poisson equation to obtain the field E. ax=0 for cartesian indexing in f,
    and ax=1 for matrix indexing (cumulative integration along the v-axis).
    """
    kappa = 1 #4 * np.pi #(3 * H0**2) / 2
    rho = -kappa * (np.trapz(f, dx=dv, axis=0))# - 1)
    phi = spectral_calc(rho, kx, o=1, d=1)
    return phi

def velocity_shift(f, phi, kx, kv, Nx, Nv, dt):
    """f is a 2D matrix of shape (M, N); kv represent the fourier modes of the variable
    v, while x is the other variable. Note that M = len(v), N = len(x). This function takes
    the matrix f into the fourier space of v, and interpolates f to a shifted position
    v' : (v - v') = -E*dt. The result is taken back to real space.
    """
    f_kv = np.zeros([Nv, Nx], dtype=float)
    # E = np.real(np.fft.ifft(1j*kx*np.fft.fft(phi)))
    E = phi
    for j in range(0, Nx):
        f_kv[:, j] = np.real(np.fft.ifft(np.fft.fft(f[:, j])*np.exp(1j * kv *  E[j] * dt)))
    return f_kv

def time_ev(f0, x, v, t):
    """Simulates the time evolution of f0.
    """
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
    kv = np.fft.fftfreq(v.size, dv) * 2.0 * np.pi
    Nt = t.size
    Nx = x.size
    Nv = v.size
    # t = (2 * (a ** (3 / 2))) / (3 * H0)
    m = 1
    H0 = 1
    f_sp = f0
    for i in range(Nt-1):
        dt = t[i+1] - t[i]
        f_sp = spatial_shift(f_sp, v, kx, Nx, Nv, dt/2)
        phi = field_solver(f_sp, kx, dx, dv, Nx, H0, t[i])
        f_sp = velocity_shift(f_sp, phi, kx, kv, Nx, Nv, dt)
        f_sp = spatial_shift(f_sp, v, kx, Nx, Nv, dt/2)
        print('t = {}'.format(t[i]))
    fn = f_sp
    return fn
