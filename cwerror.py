#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind = 'gaussian'


results = pd.read_csv('../phase_runs_params_{}.csv'.format(kind))
a = results['a']
cs2 = results['cs2']
cv2 = results['cv2']
err_cs2 = results['err_cs2']
err_cv2 = results['err_cv2']
ctot2 = cs2 + cv2
err_ctot2 = np.sqrt(err_cs2**2 + err_cv2**2)




xlabel = r'$a$'
ylabel = r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$'
yaxes_sharp = [sharp3, sharp2, sharp1]
yaxes_gauss = [gauss3, gauss2, gauss1]
colours = ['orange', 'cyan', 'k']
labels = [r'$\mathrm{B^{+12}}$', 'M&W', r'from fit to $\left<\tau_{l}\right>$']
linestyles = ['solid', 'solid', 'solid']
markers = ['v', '*', 'o']

xaxis = a_list
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})
ax[0].set_xlabel(xlabel, fontsize=18, x=1)

ax[0].set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (Gaussian)'.format(int(Lambda/(2*np.pi))), fontsize=18)
ax[1].set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ (sharp)'.format(int(Lambda/(2*np.pi))), fontsize=18)

# ax[0].errorbar(xaxis, yaxes_gauss[0], yerr=yerror_gauss, c=colours[0], ls=linestyles[0], lw=2, label=labels[0], marker='o')

for i in range(len(yaxes_sharp)):
    ax[0].plot(xaxis, yaxes_gauss[i], c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker=markers[i])
    ax[1].plot(xaxis, yaxes_sharp[i], c=colours[i], ls=linestyles[i], lw=2, label=labels[i], marker=markers[i])

# ax[0].plot(xaxis, yaxes_gauss[2], c=colours[2], ls=linestyles[2], lw=2, label=labels[2], marker='*')
# ax[0].set_ylim(0, 1.5)

# ax[1].errorbar(xaxis, yaxes_sharp[0], yerr=yerror_sharp, c=colours[0], ls=linestyles[0], lw=2, label=labels[0], marker='o')
# ax[1].plot(xaxis, yaxes_sharp[1], c=colours[1], ls=linestyles[1], lw=2, label=labels[1], marker='v')
# ax[1].plot(xaxis, yaxes_sharp[1], c=colours[2], ls=linestyles[2], lw=2, label=labels[2], marker='*')


for i in range(2):
    ax[i].set_ylabel(ylabel, fontsize=18)
    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=13)
    ax[i].ticklabel_format(scilimits=(-2, 3))
    ax[i].yaxis.set_ticks_position('both')
    ax[i].axvline(a_sc, c='g', lw=1, label=r'$a_{\mathrm{sc}}$')

ax[1].yaxis.set_label_position('right')

plt.subplots_adjust(wspace=0)
ax[1].tick_params(labelleft=False, labelright=True)
ax[1].legend(fontsize=13)#, loc=2, bbox_to_anchor=(1,1))
# plt.savefig('../plots/test/ctot.png', bbox_inches='tight', dpi=150)

plots_folder = 'test/ctot2/4v8'#/paper_plots'
# plots_folder = 'nbody_gauss_run4/'#/paper_plots'

savename = 'abs_sig_ctot2_ev_L{}_4'.format(int(Lambda/(2*np.pi)))
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
plt.close()
