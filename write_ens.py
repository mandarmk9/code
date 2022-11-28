#!/usr/bin/env python3
import time
import numpy as np
import multiprocessing as mp
from functions import write_sim_data

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
A = [-0.05, 1, -0.5, 11]

n_runs = 24
mode = 1
kinds = ['sharp', 'gaussian']
kind_txts = ['sharp cutoff', 'Gaussian smoothing']
# which = 0

tmp_st = time.time()
# n = 5
zero, Nfiles = 0, 25
def write(j, Lambda, path, A, kind, mode, run, folder_name=''):
    path = path[:-2] + '{}/'.format(run)
    write_sim_data(j, Lambda, path, A, kind, mode, folder_name)

for which in range(0, 2):
    kind = kinds[which]
    kind_txt = kind_txts[which]
    for Lambda in range(2,7):
        print('Lambda = {} ({})'.format(Lambda, kind_txt))
        # folder_name = '/new_data_{}/L{}'.format(kind, Lambda)
        Lambda *= (2*np.pi)
        for j in range(zero, Nfiles):#, Nfiles):
            print('Writing {} of {}'.format(j+1, Nfiles))
            tasks = []
            for run in range(1, 1+n_runs):
                p = mp.Process(target=write, args=(j, Lambda, path, A, kind, mode, run,))
                tasks.append(p)
                p.start()
            for task in tasks:
                p.join()
tmp_end = time.time()
print('multiprocessing takes {}s'.format(np.round(tmp_end-tmp_st, 3)))
#
