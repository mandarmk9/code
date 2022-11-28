#!/usr/bin/env python3
import h5py
import numpy as np

from functions import *

positions = []
velocities = []
PIDs = []

data_filedir = '/vol/aibn31/data1/mandar/data/multi_128_run/'
filedir = '/vol/aibn31/data1/mandar/gadget_runs/sup_14_n128_multi/output/'
fileext = '.hdf5'
print('writing combined snapshot...')

Nfiles = 1
Nsnaps = 15
# for j in range(Nfiles):
j = 15
for l in range(Nsnaps):
    print('collected snapshot {} of {}...'.format(l+1, Nsnaps))
    filename = 'snapshot_{0:03d}.{0:01d}'.format(j, l)
    filepath = filedir + filename + fileext

    file = h5py.File(filepath, mode='r')
    pos = np.array(file['/PartType1/Coordinates'])
    vel = np.array(file['/PartType1/Velocities'])
    IDs = np.array(file['/PartType1/ParticleIDs'])
    header = file['/Header']
    z = header.attrs.get('Redshift')
    file.close()

    positions.extend(pos)
    velocities.extend(vel)
    PIDs.extend(IDs)

print('writing snapshot {} of {} to data...'.format(j+1, Nfiles))
Nx = len(PIDs) #total particle number
a = (1 / (1 + z))
H0 = 100
L = 2*np.pi
N = 128
dx = L/N
A = [-0.01, 1, -0.5, 11]
q = np.arange(0, L, dx)
Psi_q = -Psi_q_finder(q, A)
Psi_t = a * Psi_q  #this is the displacement field \Psi_{t} = a(t) \times \int(-\delta(q) dq)
x = q + Psi_t #eulerian position
v = H0 * np.sqrt(a) * (Psi_q) #peculiar velocity

x_gadget = np.array(positions)[:, 0]
v_gadget = np.array(velocities)[:, 0] * np.sqrt(a)

fake_id = (N**2)*np.arange(N)
indices = []
for id in fake_id:
    try:
        indices.append(np.where(IDs==id)[0][0])
    except IndexError:
        print('couldn\'t find index {}...'.format(id))
        pass

rho_0 = 27.755
m = rho_0*(dx**3)

p2 = []
v2 = []
for index in indices:
    p2.append(x_gadget[index])
    v2.append(v_gadget[index])

pos = np.array(p2)
vel = np.array(v2)

print(len(set(pos)))
# data_filename = 'data_{0:03d}'.format(j)
# data_file = data_filedir + data_filename + fileext
# with h5py.File(data_file, 'w') as hdf:
#     hdf.create_group('Header')
#     header = hdf['Header']
#     header.attrs.create('L', 2*np.pi, dtype=float)
#     header.attrs.create('Nx', 128, dtype=float)
#     header.attrs.create('a', (1/(1+z)), dtype=float)
#     hdf.create_dataset('Positions', data=pos)
#     hdf.create_dataset('Velocities', data=vel)
#     print('snapshot written!')
