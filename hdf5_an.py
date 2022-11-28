#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import h5py
import numpy as np
import matplotlib.pyplot as plt

from functions import *

Nx = 420
j = 0

filedir = '/vol/aibn31/data1/mandar/data/N{}/'.format(Nx)
filename = 'data_{0:03d}'.format(j)
fileext = '.hdf5'

filepath = filedir + filename +fileext

file = h5py.File(filepath, mode='r')
pos = np.array(file['/Positions'])
vel = np.array(file['/Velocities'])
IDs = np.array(file['/IDs'])
header = file['/Header']
a = header.attrs.get('a')
file.close()

print(IDs, IDs.size)
