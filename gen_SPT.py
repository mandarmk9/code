#!/usr/bin/env python3
import numpy as np

j = 0
path = 'cosmo_sim_1d/nbody_gauss_run_4/'

moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
moments_file = np.genfromtxt(path + moments_filename)
a = moments_file[:,-1][0]
x_cell = moments_file[:,0]
M0_par = moments_file[:,2]-1 #the -1 is for δ

Nx = x_cell.size
L = 1.0
k = np.fft.ifftshift(2.0 * np.pi * np.arange(-Nx/2, Nx/2))
k_half = k[int(Nx/2):]

k_half = np.array([1, 2, 3, 4, 5])
delta_2 = np.zeros((k_half.size,2))
for i in range(k_half.size):
    k_half_shifted = np.roll(k_half, i)
    print(k_half, k_half_shifted)
    delta_2[i] = [k_half[i], k_half_shifted[i]]
print(delta_2)
