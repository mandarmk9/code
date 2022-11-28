#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from functions import spectral_calc

A = np.random.uniform(0, 1, (3,3))
B = np.random.uniform(0, 1, (3,3))
C = np.random.uniform(0, 1, (3,3))

ABC = np.linalg.inv((A.dot(B)).dot(C))
ABC_ = np.linalg.inv(C).dot(np.linalg.inv(B).dot(np.linalg.inv(A)))
 #np.dot(np.dot(A, B), C)
print(ABC - ABC_)
# print(ABC_)

# L = 1#2 * np.pi
# N = 2**10
# dx = L / N
# x = np.arange(0, L, dx)
# k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
#
# k1 = 1
# k2 = 2
#
# f1 = np.sin(2 * np.pi * k1 * x / L)
# f2 = np.cos(2 * np.pi * k2 * x / L)
# prod = f1 * f2
# d_prod_dir = spectral_calc(f1 * f2, k, L, o=1, d=0)
# d_prod_rule = (f1 * spectral_calc(f2, k, L, o=1, d=0)) + (f2 * spectral_calc(f1, k, L, o=1, d=0))
#
# fig, ax = plt.subplots()
# ax.plot(x, d_prod_dir, c='k', lw=2, label='direct')
# ax.plot(x, d_prod_rule, c='b', lw=2, ls='dashed', label='rule')
# plt.savefig('../plots/cosmo_sim/k_test.png', bbox_inches='tight', dpi=120)
