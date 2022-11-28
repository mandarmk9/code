#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
tsim_0 = time.time()
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
    V = np.fft.fft(rho)
    V[0] = 0
    V[1:] /= - kx[1:] ** 2
    return np.real(np.fft.ifft(V))

def field_solver(f, kx, dp, kappa):
    f_dp = np.trapz(f, dx=dp, axis=0)
    # f_dp_bar = np.mean(f_dp)
    rho = kappa * (f_dp - 1)
    # phi = poisson_solver(rho, kx)
    E = spectral_calc(rho, kx, o=1, d=1)
    return E

def write_moments(loc, name, x, M0, M1, M2, a):
    print('Writing output file (M0, M1, M2) for a = {}'.format(a))
    filename = str(loc) + 'moments_{0:05d}.hdf5'.format(name)

    with hp.File(filename, 'w') as hdf:
        hdf.create_dataset('M0', data=M0)
        hdf.create_dataset('M1', data=M1)
        hdf.create_dataset('M2', data=M2)
        hdf.create_dataset('x', data=x)
        hdf.create_dataset('a', data=np.array(a))

def write_dist(loc, name, x, p, fn, a):
    print('Writing output file (dist) for a = {}'.format(a))
    filename = str(loc) + 'dist_{0:05d}.hdf5'.format(name)
    with hp.File(filename, 'w') as hdf:
        hdf.create_dataset('f', data=fn)
        hdf.create_dataset('x', data=x)
        hdf.create_dataset('p', data=p)
        hdf.create_dataset('a', data=np.array(a))

def moment(F, P, n):
    F1 = F * (P ** n)
    M = np.sum(F1, axis=0)
    return M

def flagger(loc):
    if os.path.exists(str(loc) + 'stop'):
        os.remove(str(loc) + 'stop')
        return 1
    else:
        return 0

def time_step(loc, f0, x, p, H0=100, m=1, t0=0.1, dt_max=1e-2, tn=1, N_out=10, save_dist=False):
    """This function evolves the Vlasov distribution from an initial value f0.
    loc : the directory where the output is saved,
    f0 : the initial Vlasov distribution, with size Nx * Np,
    (x,p) : the phase-space grid,
    H0 : the Hubble constant in units of h km/s/Mpc,
    m : the mean mass in a cell in units of 1e10 M_sun,
    t0 : the initial time (the scalefactor in a cosmological simulation),
    dt_max : the maximum time-step,
    tn : the stopping time,
    N_out : the number of time-steps after which an output is written,
    save_dist : if True, the code saves the full distribution after every N_out time-steps,
    """
    #grid-spacings in x-, p- space
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    #Fourier wavevector for x and p
    kx = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi
    kp = np.fft.fftfreq(p.size, dp) * 2.0 * np.pi

    #create a 2D grid on which the distribution is evolved
    X, P = np.meshgrid(x, p)

    #set the initial distribution and scalefactor; these are updated in the main loop
    fn = f0
    t = t0
    a = (3*H0*t/2) ** (2/3)
    #flag controls the main loop, count decides when to write an output file, and name tracks the output file index
    flag, count, name = 0, 0, 0

    #main loop
    while flag == 0:
        a = (3*H0*t/2) ** (2/3)
        kappa = (3 * (H0**2)) / (2 * a)
        vel = p / (m * a**2)

        #dt_in defines an internal criteria for time-stepping
        dt_int = dx / np.max(vel)
        print(dt_max, dt_int)
        dt = np.min([dt_max, dt_int])

        #first, the first half-step in x
        fn = spatial_advection(fn, kx, vel, dt)

        # calculation of the potential, followed by the step in p
        E = field_solver(fn, kx, dp, kappa)
        fn = velocity_advection(fn, kp, m*E, dt)

        #finally, the second half-step in x
        fn = spatial_advection(fn, kx, vel, dt)
        fn /= np.mean(np.trapz(fn, dx=dp, axis=0))

        # #to remove filamentation as soon as it appears, we set all negative values to zeros
        # fn[fn < 0] = 0

        #update time and count
        t += dt
        count += 1

        #save file if the save conditions are met
        if t > tn:
            print('last time-step reached...')
            flag = 1
        else:
            print('checking flag status...')
            flag = flagger(loc)
            print('flag = {}'.format(flag))

        if count == N_out:
            name += 1

            if save_dist == True:
                write_dist(loc, name, x, p, fn, a)
            else:
                M0 = moment(fn, P, 0)
                M1 = moment(fn, P, 1)
                M2 = moment(fn, P, 2)
                write_moments(loc, name, x, M0, M1, M2, a)

            count = 0

        print('Solved for a = {}'.format(a))
        print('The last time-step was dt = {} \n'.format(dt))

        if flag == 1:
            print('Stopping run...')
            if save_dist == True:
                write_dist(loc, name, x, p, fn, a)
            else:
                M0 = moment(fn, P, 0)
                M1 = moment(fn, P, 1)
                M2 = moment(fn, P, 2)
                write_moments(loc, name, x, M0, M1, M2, a)
            tsim_n = time.time()
            print('Done! The solver took {}s.'.format(tsim_n-tsim_0))
