#!/usr/bin/env python3

import os
import numpy as np
loc = 'cosmo_sim_1d/sim_k_1_11/run1/data_sharp/L6/'
files = os.listdir(loc)
change_index = 1

for j in range(len(files)):
    # print(files[j])
    if 'sol_' in files[j]:
        try:
            old_filename = loc + files[j]
            old_index_string = files[j][-8:-5]
            if int(old_index_string) > 22:
                new_index = int(old_index_string) - 1
                new_index_string = '{0:03d}'.format(new_index)
                # print(old_index_string, new_index_string)
                new_filename = old_filename.replace(old_index_string, new_index_string)
                print(old_filename, new_filename)
                os.rename(old_filename, new_filename)

        except Exception as e: print(e)

    # old_index = int(old_index_string)
    # if old_index > 528:
    # new_index = old_index + change_index
    # print(old_filename)#, new_filename)
        # break
