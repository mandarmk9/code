#!/usr/bin/env python3
import time
t0 = time.time()
import numpy as np
import h5py
import matplotlib.pyplot as plt

loc = '/vol/aibn31/data1/mandar/'

def mean_calc(Nfiles, run, loc='/vol/aibn31/data1/mandar/'):
   mean, a_list = [], []
   for i in range(Nfiles):
      with h5py.File(loc + 'data' + run + 'psi_{0:05d}.hdf5'.format(i), 'r') as hdf:
         ls = list(hdf.keys())
         psi = np.array(hdf.get(str(ls[3])))
         t = np.array(hdf.get(str(ls[1])))
         print('t = ', t)
         a_list.append(t)
         M = np.sum(np.abs(psi**2)) #* 2 * np.pi / psi.size
         mean.append(M)
      print('true data structure: ', ls)
      # break
   return mean, a_list


Nfiles1 = 378
Nfiles2 = 4094
Nfiles3 = 4095

run1 = '/sch_hfix_run17/'
# run2 = '/sch_hfix_run14/'
# run3 = '/sch_hfix_run14/'
# run4 = '/mz_run15/'

dt1 = 1e-2
dt2 = 1e-3
dt3 = 0.1
dt4 = 1

mean1, a1 = mean_calc(Nfiles1, run1)
# mean2, a2 = mean_calc(Nfiles2, run2)
# mean3, a3 = mean_calc(Nfiles3, run3)
# # mean4, a4 = mean_calc(Nfiles, run4)
#
fig, ax = plt.subplots(figsize=(8, 5))
ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
ax.set_xlabel('a', fontsize=12)
ax.set_ylabel(r'$\frac{M(a) - M(0)}{M(0)}$', fontsize=12)
# ax.set_ylabel(r'$\frac{\rho(a) - \rho(0)}{\rho(0)}$', fontsize=12)

print(mean1)
mass_err = (mean1 - mean1[0]) / mean1[0]

ax.plot(a1, mass_err, c='b')#, label='dt = {}'.format(dt1))
# ax.plot(mean2, c='k', label='dt = {}'.format(dt2))
# ax.plot(a3, mean3, c='r', label='h = {}'.format(dt3))
# ax.plot(a4, mean4, c='brown', label='h = {}'.format(dt4))

plt.legend()

print('saving...')
plt.savefig(loc + 'plots/mz_runs/total_mass_ev.png', bbox_inches='tight', dpi=120)
plt.close()

tn = time.time()
print('This took {}s'.format(np.round(tn-t0, 3)))
