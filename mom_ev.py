#!/usr/bin/env python3
"""A script for reading and plotting snapshots from cosmo_sim_1d"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
from matplotlib.pyplot import cm

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# Nfiles = 51
# M0_list = np.zeros(Nfiles)
# a_list = np.zeros(Nfiles)
# moments = np.zeros(shape=(12, Nfiles))
# for j in range(Nfiles):
#     nbody_filename = 'output_{0:04d}.txt'.format(j)
#     nbody_file = np.genfromtxt(path + nbody_filename)
#     x_nbody = nbody_file[:,-1]
#     v_nbody = nbody_file[:,2]
#
#     moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
#     moments_file = np.genfromtxt(path + moments_filename)
#     a = moments_file[:,-1][0]
#     print('a = {}'.format(a))
#     x = moments_file[:,0]
#     g = int(x.size/3) - 1
#     for l in range(2,14):
#         moments[l-2][j] = moments_file[:,l][g]
#     a_list[j] = a
#     # print(g, x[g])
#
# df = pandas.DataFrame(data=moments)
# df_a = pandas.DataFrame(data=a_list)
#
# pickle.dump(df, open("./data/sim_k_1_11_moment_evolution_{}.p".format(g), "wb"))
# # pickle.dump(df_a, open("./data/sim_k_1_11_a_list.p", "wb"))


# g = int(125000/3) - 1
# moments = pickle.load(open("./data/sim_k_1_11_moment_evolution_{}.p".format(g), "rb" ))
xg = 0.5
moments = pickle.load(open("./data/sim_k_1_11_moment_evolution.p", "rb" ))

a_list = np.array(pickle.load(open("./data/sim_k_1_11_a_list.p", "rb" )))

labels = ['M0', 'C0', 'M1', 'C1', 'M2', 'C2', 'M3', 'C3', 'M4', 'C4', 'M5', 'C5']

# j = 11
colours =['k','b','g','r'] # list of basic colors
linestyles = ['-','--','-.',':'] # list of basic linestyles

fig, ax = plt.subplots()
# ax.set_title(r'$a = {}$'.format(a))
ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)

ax.set_ylabel(r'$C_{{i}}(x={})$'.format(xg), fontsize=14)
ax.set_xlabel(r'$a$', fontsize=14)
ax.axvline(a_list[12], c='grey', lw=1)
ax.axvline(a_list[35], c='grey', lw=1, label=r'$a_{\mathrm{sc}}$')
# cycle = np.arange(1, 13, 2)
# color = iter(cm.Dark2(np.linspace(0, 1, cycle.size)))
for j in range(3, 4):
    c = colours[j // 4]
    ls = linestyles[j % 4]
    # c = next(color)
    field = np.array(moments.iloc[j,:])
    # if j > 5:
    # field = np.abs(field)**(1/int(labels[j][-1]))
    norm = 1
    ax.plot(a_list, field / norm, c=c, lw=2, ls=ls, label=labels[j])

ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.minorticks_on()

plt.legend()
# plt.savefig('../plots/sim_k_1_11/ci_ev.png', bbox_inches='tight', dpi=150)
# plt.close()

plt.show()
