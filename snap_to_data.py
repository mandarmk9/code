#!/usr/bin/env python3
"""This file collects the 3D data cubes from the Gadget snapshot files and writes a compact hdf5 file
    containing the positions, velocities, and particles IDs of the 1D perturbed particles.
"""
import time
t0 = time.time()

import h5py
import numpy as np
import matplotlib.pyplot as plt

fileext = '.hdf5'
err = 0 #useful flag for monitoring errors in the code
NumFilesPerSnapshot = 60 #number of files each snapshot is split into
NumSnaps = 48 #total number of snapshots
Nx = 480 #number of particles along each dimension
data_filedir = '/vol/aibn31/data1/mandar/data/N{}/'.format(Nx) #where to save the compact data file

#We start by writing the IDs of the particles we are interested in
in_IDs = (Nx ** 2) * np.arange(Nx)

for l in range(NumSnaps): #l dictates the number of snapshots
    print("Reading snapshot {} of {}...\n".format(l+1, NumSnaps))
    positions, velocities, PIDs = [], [], [] #initialise empty lists for the positions, velocities & particles IDs
    l_str = '{0:03d}'.format(l) #necessary step; otherwise the next for loop doesn't iterate over j for the snap_filename

    for j in range(NumFilesPerSnapshot):
        #where to read the snaps from
        snap_filename = 'snapshot_{}.{}'.format(l_str, j)
        snap_filepath = '/vol/aibn31/data1/mandar/gadget_runs/sup_14_n420_run1/output/' + snap_filename + fileext

        # snap_filename = 'n{}.{}'.format(Nx, j)
        # snap_filepath = '/vol/aibn31/data1/mandar/gadget_runs/ICs/test/' + snap_filename + fileext

        #open the individual snap parts
        file = h5py.File(snap_filepath, mode='r')
        print("File {}: part {} of {}...".format(snap_filename, j+1, NumFilesPerSnapshot))
        pos = np.array(file['/PartType1/Coordinates'])
        vel = np.array(file['/PartType1/Velocities'])
        IDs = np.array(file['/PartType1/ParticleIDs'])
        header = file['/Header']
        N = header.attrs.get('NumPart_ThisFile')[1]
        z = header.attrs.get('Redshift')
        file.close()

        #collect the perturbed particles
        for i in range(in_IDs.size):
            try:
                match = np.where(IDs - in_IDs[i] == 0)[0][0]
                positions.append(pos[match, 0])
                velocities.append(vel[match, 0])
                PIDs.append(IDs[match])
            except IndexError:
                pass

    PIDs = np.array(PIDs)
    PIDs, positions, velocities = (list(t) for t in zip(*sorted(zip(PIDs, positions, velocities))))

    if len(positions) == Nx:
        print("\nCollected the perturbed particles, writing to data file...")
        #write the collected particles for each snap into a data file
        data_filename = 'data_{0:03d}'.format(l)
        data_filepath = data_filedir + data_filename + fileext
        with h5py.File(data_filepath, 'w') as hdf:
            hdf.create_group('Header')
            header = hdf['Header']
            header.attrs.create('L', 2*np.pi, dtype=float)
            header.attrs.create('a', (1/(1+z)), dtype=float)
            hdf.create_dataset('Positions', data=np.array(positions))
            hdf.create_dataset('Velocities', data=np.array(velocities) * (1/(1+z))) #we save the peculiar velocity, not the particle velocity
            hdf.create_dataset('IDs', data=np.array(PIDs))
            print('Writing done!\n')
    else:
        print("\nSomething went wrong, check the collection of perturbed particles.\n Expected {} particles, received {}. Exiting...\n".format(Nx, len(positions)))
        err = 1
        break

tn = time.time()
if err == 0:
    print("Task finished successfully, took {}s".format(np.round(tn-t0, 3)))
if err == 1:
    print("Task finished with errors, took {}s".format(np.round(tn-t0, 3)))
