#!/usr/bin/env python3

import os
import numpy as np
loc = 'cosmo_sim_1d/nbody_hier/moments'
files = os.listdir(loc)
change_index = 1

# print(files)
for j in range(len(files)):
    if 'M0_' in files[j]:
        try:
            old_filename = loc + files[j]
            old_index_string = files[j][3:-5]
            # print(old_index_string)

            new_index = int(old_index_string) + 1
            # print(new_index)
            new_index_string = '{0:04d}'.format(new_index)
            # print(old_index_string, new_index_string)
            new_filename = old_filename.replace(old_index_string, new_index_string)
            print(old_filename, new_filename)
            os.rename(old_filename, new_filename)
        except: pass
    # old_index = int(old_index_string)
    # if old_index > 528:
    # new_index = old_index + change_index
    # print(old_filename)#, new_filename)
        # break
