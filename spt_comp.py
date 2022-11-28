#!/usr/bin/env python3

from SPT import *

n = 3
filename = './spt_kernels/F2.hdf5'
F = kernel_sym(n)
print(F)
# spt_write_to_hdf5(filename, F)
# G = spt_read_from_hdf5(filename)
#
# print(G)
