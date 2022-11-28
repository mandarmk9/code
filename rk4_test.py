#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

Nfiles = 51

#define lists to store the data
x = np.arange(0, 2*np.pi, 2*np.pi/Nfiles)

f = np.zeros(Nfiles, dtype='complex')
g = np.zeros(Nfiles, dtype='complex')
p = np.zeros(Nfiles, dtype='complex')
q = np.zeros(Nfiles, dtype='complex')
r = np.zeros(Nfiles, dtype='complex')

p = -np.sin(x)
q = 1 + np.cos(x)
r = np.ones(Nfiles)

#ICs
f[0] = 1
g[0] = 0

f_half, r_half = 0, 0
#loop for RK4
for j in range(Nfiles-1):
    h = (x[j+1] - x[j])
    p_half = -np.sin((x[j+1] - x[j]) / 2)
    q_half = 1 + np.cos((x[j+1] - x[j]) / 2)
    k1 = -(p[j]*g[j] + q[j]*f[j] + r[j])
    k2 = -(p_half*(g[j]+(h*k1/2)) + q_half*f_half + r_half)
    k3 = -(p_half*(g[j]+(h*k2/2)) + q_half*f_half + r_half)
    k4 = -(p[j+1]*g[j+1] + q[j+1]*f[j+1] + r[j+1])

    g[j+1] = g[j] + (k1 + 2*k2 + 2*k3 + k4)/6
    f[j+1] = np.trapz(g[:j], x[:j])
    f_half = np.cos((x[j+1]-x[j])/ 2)#(f[j+1] - f[j]) / 2
    r_half = (r[j+1] - r[j]) / 2


f_an = np.cos(x)
g_an = -np.sin(x)
fig, ax = plt.subplots()
ax.set_xlabel('a')
ax.plot(x, g_an, c='k', lw=2, label='analytical')
ax.plot(x, g, c='b', ls='dashed', lw=2, label='RK4')
plt.legend()
# plt.savefig('../plots/test/greens_test.png', dpi=150)
plt.show()
