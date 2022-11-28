#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from functions import plotter, dn, smoothing, read_density
from scipy.interpolate import interp1d
from SPT import SPT_final
from zel import *
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/nbody_gauss_run_3/'
plots_folder = 'test' #_k7_L3'

zero = 0
Nfiles = 39
mode = 1
sm = 1
Lambda = 100 * (2 * np.pi)
kind = 'sharp'
# kind = 'gaussian'

#define lists to store the data
a_list = np.zeros(Nfiles)

#the densitites
P_nb = np.zeros(Nfiles)
P_lin = np.zeros(Nfiles)
P_1l = np.zeros(Nfiles)
P_2l = np.zeros(Nfiles)

def SPT(dc_in, k, L, Nx, a):
  """Returns the SPT PS upto 2-loop order"""
  F = dn(5, k, L, dc_in)
  d1k = (np.fft.fft(F[0]) / Nx)
  d2k = (np.fft.fft(F[1]) / Nx)
  d3k = (np.fft.fft(F[2]) / Nx)
  d4k = (np.fft.fft(F[3]) / Nx)
  d5k = (np.fft.fft(F[4]) / Nx)

  P11 = (d1k * np.conj(d1k)) * (a**2)
  P12 = ((d1k * np.conj(d2k)) + (d2k * np.conj(d1k)))  * (a**3)
  P22 = (d2k * np.conj(d2k)) * (a**4)
  P13 = ((d1k * np.conj(d3k)) + (d3k * np.conj(d1k))) * (a**4)
  P14 = ((d1k * np.conj(d4k)) + (d4k * np.conj(d1k))) * (a**5)
  P23 = ((d2k * np.conj(d3k)) + (d3k * np.conj(d2k))) * (a**5)
  P33 = (d3k * np.conj(d3k)) * (a**6)
  P15 = ((d1k * np.conj(d5k)) + (d5k * np.conj(d1k))) * (a**6)
  P24 = ((d2k * np.conj(d4k)) + (d4k * np.conj(d2k))) * (a**6)

  P_lin = P11
  P_1l = P_lin + P12 + P13 + P22
  P_2l = P_1l + P14 + P15 + P23 + P24 + P33
  return np.real(P_lin), np.real(P_1l), np.real(P_2l)


for j in range(1, Nfiles):

    # # dk_par, a1, dx = read_density(path, 0)
    # # dc_in = np.real(np.fft.ifft(dk_par))
    # x = np.arange(0, 1.0, L/Nx)
    # A = [-0.5, 1, -0.05, 11]
    # dc_in = initial_density(x, A, L)
    # # initial_file = np.genfromtxt(path + 'output_initial.txt')
    # # q = initial_file[:,0]
    # # Psi = initial_file[:,1]
    # # L = 1.0
    # # Nx = q.size
    # # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    # # dc_in = -spectral_calc(Psi, L, o=1, d=0) / a
    #

    # dk_par, a, dx = read_density(path, j)
    # L = 1.0
    # Nx = dk_par.size
    # k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-Nx/2, Nx/2))
    # amp = 5 #np.max(k)
    # P_in = np.zeros(k.size)
    # P_in[1:] = amp*np.abs(k[1:])**(0.5)
    # print(P_in)
    # dk_in = np.sqrt(P_in) * np.exp(2j* np.pi*np.random.uniform(0, 0.01, k.size))
    # dc_in = np.real(np.fft.ifft(dk_in))

    # P_lin_a, P_1l_a, P_2l_a = SPT(dc_in, k, L, Nx, a)

    # dk_par *= dk_par.size
    # P_nb_a = np.real(dk_par * np.conj(dk_par))

    #we now extract the solutions for a specific mode
    P_nb[j] = P_nb_a[mode] #/ a**2
    # P_lin[j] = P_lin_a[mode] #/ a**2
    # P_1l[j] = P_1l_a[mode] #/ a**2
    # P_2l[j] = P_2l_a[mode] #/ a**2
    a_list[j] = a

    print('a = ', a, '\n')

# print(P_nb, P_1l, P_2l)
xaxis = a_list
yaxes = [P_nb / a_list**2]#, P_1l / a_list**2]#, P_2l / a_list**2]
colours = ['b', 'brown', 'magenta']
labels = [r'$N-$body', 'SPT: 1-loop', 'SPT: 2-loop']
linestyles = ['solid', 'dashed', 'dashed']
title = r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode)

xlabel = r'$a$'
ylabel = r'$P(k=1, a) \; / a^{2}$'
savename = 'nbody_spectra_k{}'.format(mode)
a_sc = 1
plotter(mode, Lambda, xaxis, yaxes, xlabel, ylabel, colours, labels, linestyles, plots_folder, savename, a_sc, title_str=title)
