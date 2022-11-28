#!/usr/bin/env python3
import numpy as np
import h5py

from functions import kde_gaussian

loc2 = '/vol/aibn31/data1/mandar/data/sch_hfix_run5/'

print("Collecting Schroedinger scalefactors...")
filename = 'sch_a_list.hdf5'

with h5py.File(filename, mode='r') as hdf:
    ls = list(hdf.keys())
    a_list_sch = np.array(hdf.get(str(ls[0])))

print("Matching to Gadget scalefactors...")

num_files = 39
# a_list_gadget = np.empty(num_files)

for j in range(num_files):
    gadget_files = '/vol/aibn31/data1/mandar/code/N300/'
    file = h5py.File(gadget_files + 'data_{0:03d}.hdf5'.format(j), mode='r')
    pos = np.array(file['/Positions'])
    header = file['/Header']
    a_gadget = header.attrs.get('a')
    N = int(header.attrs.get('Nx'))
    file.close()
    # a_list_gadget[j] = a_gadget

    a_diff = np.abs(a_list_sch - a_gadget)
    a_ind = np.where(a_diff == np.min(a_diff))[0][0]
    diff = a_diff[a_ind] * 100 / a_gadget

    with h5py.File(loc2 + 'psi_{0:05d}.hdf5'.format(a_ind), 'r') as hdf:
        ls = list(hdf.keys())
        A = np.array(hdf.get(str(ls[0])))
        a = np.array(hdf.get(str(ls[1])))
        L, h, m, H0 = np.array(hdf.get(str(ls[2])))
        psi = np.array(hdf.get(str(ls[3])))

    Nx = psi.size
    dx = L / Nx

    x = np.arange(0, L, dx)
    k = np.fft.fftfreq(x.size, dx) * 2.0 * np.pi

    q = np.arange(0, L, L/N)
    k_pos = np.fft.fftfreq(q.size, q[1]-q[0]) * 2 * np.pi

    Lambda = 6
    sm = (Lambda ** 2) / 2

    dist = x - 0
    dist[dist < 0] += L
    dist[dist > L/2] = - L + dist[dist > L/2]

    W_k_an = (np.sqrt(np.pi / sm)) * np.exp(- (k ** 2) / (4 * sm))

    den_nbody = kde_gaussian(q, pos, sm, L)

    den_sch = np.abs(psi)**2 - 1 #'unsmoothed' schrodinger overdensity ('' because hbar is still there)
    dk_sch = np.fft.fft(den_sch) #* dx
    dk_sch *= W_k_an

    dk_nbody = np.fft.fft(den_nbody) * Nx / N #* ((L / N))

    dk2_nbody = np.abs(dk_nbody) ** 2 #/ (a**2)
    dk2_sch = np.abs(dk_sch) ** 2 #/ (a**2)

    err = (dk2_sch[11] - dk2_nbody[11]) * 100 / dk2_nbody[11]

    print(a_list_sch[a_ind])#, a_gadget)
    print('den error = {}%'.format(err))
    # print('time error = {}%'.format(diff))
