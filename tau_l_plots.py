#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from matplotlib.pyplot import cm
from functions import read_sim_data

path = 'cosmo_sim_1d/new_sim_k_1_11/'
A = []
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
plots_folder = 'new_sim_k_1_11/tau/'
mode = 1
nruns = 24
color = iter(cm.rainbow(np.linspace(0, 1, nruns)))
ls = iter([':', '-.', '-', '--']*6)

taus = []
fig, ax = plt.subplots()
ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
for run in range(1, nruns+1):
    sim_path = path + 'run{}/'.format(run)
    j = 13
    sol = read_sim_data(sim_path, Lambda, kind, j)
    if run == 1:
        a = sol[0]
        x = sol[1]
        tau_l = sol[-3]
    # elif run == 7:
    #     pass
    else:
        a, tau_l = sol
        x = np.arange(0, 1, 1/tau_l.size)
    taus.append(tau_l)
    ax.plot(x, tau_l, c=next(color), ls=next(ls), lw=2, label='run{}'.format(run))

mean_tau = sum(np.array(taus)) / len(taus)
ax.plot(x, tau_l, c='k', lw=2, label='mean')

ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
ax.legend(fontsize=12, bbox_to_anchor=(1,1))
ax.yaxis.set_ticks_position('both')
plt.show()
# plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, l), bbox_inches='tight', dpi=150)
# plt.close()
# plt.savefig('../plots/{}/{}_{}.png'.format(plots_folder, field, j), bbox_inches='tight', dpi=150)
#
