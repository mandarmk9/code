#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np

from functions import *

positions = []
velocities = []
PIDs = []

N = 300
Nfiles = 1
Nsnaps = 30

data_filedir = '/vol/aibn31/data1/mandar/data/sup_14_n300_run2/'.format(N)
filedir = '/vol/aibn31/data1/mandar/gadget_runs/sup_14_n{}_run2/output/'.format(N)
fileext = '.hdf5'
print('writing combined snapshot...')

# for j in range(Nfiles):
# j = 0
for l in range(Nsnaps):
    j = '037'
    print('collected snapshot {} of {}...'.format(l+1, Nsnaps))
    filename = 'snapshot_{}.{}'.format(j, str(l))
    filepath = filedir + filename + fileext

    file = h5py.File(filepath, mode='r')
    pos = np.array(file['/PartType1/Coordinates'])
    vel = np.array(file['/PartType1/Velocities'])
    IDs = np.array(file['/PartType1/ParticleIDs'])
    header = file['/Header']
    z = header.attrs.get('Redshift')
    file.close
    positions.extend(pos)
    velocities.extend(vel)
    PIDs.extend(IDs)

data_filename = 'data_' + j
data_filepath = data_filedir + data_filename + fileext
with h5py.File(data_filepath, 'w') as hdf:
    hdf.create_group('Header')
    header = hdf['Header']
    header.attrs.create('L', 2*np.pi, dtype=float)
    header.attrs.create('Nx', N, dtype=float)
    header.attrs.create('a', (1/(1+z)), dtype=float)
    hdf.create_dataset('Positions', data=np.array(positions))
    hdf.create_dataset('Velocities', data=np.array(velocities))
    hdf.create_dataset('IDs', data=np.array(PIDs))
    print('done!')
