#!/usr/bin/env python3
"""This module calculates the N-body moment hierarchy and writes to a hdf5 output file."""

import numpy as np
import h5py
from functions import is_between

def calc_hierarchy(path, file_num, dx_grid = 2.5e-4):
   print("Reading file {}".format(file_num))

   nbody_file = np.genfromtxt(path + 'output_{0:04d}.txt'.format(file_num))
   try:
      a = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(file_num))
      print('a = ', a)
   except:
      a = None

   v_nbody = nbody_file[:,2]
   x_nbody = nbody_file[:,-1]

   x_grid = np.arange(0, 1, dx_grid)
   M0 = np.zeros(x_grid.size)
   C1 = np.zeros(x_grid.size)
   M2 = np.zeros(x_grid.size)

   print("Calculating hierarchy...\n")

   for j in range(x_grid.size-1):
      s = is_between(x_nbody, x_grid[j], x_grid[j+1])
      try:
         vels = v_nbody[s[0]]
      except IndexError:
         print("No particles found in cell {}...\n".format(j))
         pass
      M0[j] = s[1].size
      C1[j] = sum(vels) / len(vels)
      M2[j] = -((sum(vels**2)  / v_nbody.size) - (C1[j]**2))

   M0[-1] = M0[0]
   M0 /= np.mean(M0)
   C1[-1] = C1[0]
   # M2 *= M0
   M2[-1] = M2[0]
   M2 -= np.min(M2)
   M1 = C1 * M0
   C2 = M2 - (M1**2 / M0)
   C0 = np.log(M0)

   return M0, M1, M2, C0, C1, C2, a, dx_grid

def write_hierarchy(path, file_num):
   M0, M1, M2, C0, C1, C2, a, dx_grid = calc_hierarchy(path, file_num)
   print("Writing file {}".format(file_num))
   filename = path + 'moments/hierarchy_{0:03d}.hdf5'.format(file_num)
   file = h5py.File(filename, 'w')
   file.create_group('Header')
   header = file['Header']
   header.attrs.create('a', a, dtype=float)
   header.attrs.create('dx', dx_grid, dtype=float)

   moments = file.create_group('Moments')
   moments.create_dataset('M0', data=M0)
   moments.create_dataset('M1', data=M1)
   moments.create_dataset('M2', data=M2)

   cumulants = file.create_group('Cumulants')
   cumulants.create_dataset('C0', data=C0)
   cumulants.create_dataset('C1', data=C1)
   cumulants.create_dataset('C2', data=C2)

   file.close()
   print("Done!\n")

   return filename

# path = 'cosmo_sim_1d/EFT_nbody_long/'
# path = 'cosmo_sim_1d/EFT_nbody_run7/'
# for file_num in range(175, 176):
#    write_hierarchy(path, file_num)

def read_hierarchy(path, file_num):
   filename = path + 'hierarchy_{0:03d}.hdf5'.format(file_num)
   file = h5py.File(filename, mode='r')
   header = file['/Header']
   a = header.attrs.get('a')
   dx = header.attrs.get('dx')

   moments = file['/Moments']
   M0 = np.array(moments['M0'])
   M1 = np.array(moments['M1'])
   M2 = np.array(moments['M2'])

   cumulants = file['/Cumulants']
   C0 = np.array(cumulants['C0'])
   C1 = np.array(cumulants['C1'])
   C2 = np.array(cumulants['C2'])

   file.close()

   return M0, M1, M2, C0, C1, C2, a, dx
