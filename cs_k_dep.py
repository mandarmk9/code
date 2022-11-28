#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

from EFT_nbody_solver import *

#define directories, file parameteres
path = 'cosmo_sim_1d/nbody_hier/'

Lambda = 5
file_num = 49
sol = param_calc(file_num, Lambda, path)
a, x, k = sol[0:3]
dx = x[1] - x[0]
sep = np.arange(0, x.size * dx, dx)
ctot2, ctot2_2, cs2, cv2 = sol[-4:]
# ctot2 = cs2 + cv2
# print(np.mean(ctot2))
# print(ctot2[0])
# print(np.median(ctot2))
#
# ctot2_k = np.fft.fft(ctot2) / ctot2.size
# print(np.real(ctot2_k[:2]))
fig, ax = plt.subplots()

ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
# ax.set_ylabel(r'$c^{{2}}\;[\mathrm{{km}}^{{2}}\mathrm{{s}}^{{-2}}]$', fontsize=14)
ax.set_ylabel(r'$P(x)$', fontsize=14)

ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=14)
# ax.set_xlabel(r'$k\;[h\;\mathrm{Mpc}^{-1}]$', fontsize=14)
# ax.set_xlim(0.46, 0.48)
# ax.set_ylim(-1.25e3, 3e3)

# ax.plot(x, cs2, c='k', lw=2, label=r'$c^{2}_{\mathrm{s}}$')
# ax.plot(x, cv2, c='b', lw=2, label=r'$c^{2}_{\mathrm{v}}$')
# ax.plot(x, ctot2, c='r', lw=2, label=r'$c^{2}_{\mathrm{tot}}$')

# ax.plot(sep, cv2, c='r', lw=2, label=r'$P_{A\delta}$')
# ax.plot(sep, cv2, c='b', lw=2, label=r'$P_{A\Theta}$')
# ax.plot(sep, cs2, c='r', lw=2, label=r'$P_{A\Theta}\partial^{2}P_{\delta\Theta} - P_{A\delta}\partial^{2}P_{\Theta\Theta}$')
# ax.plot(sep, cv2, c='b', lw=2, label=r'$(\partial^{2}P_{\delta\Theta})^{2} - \partial^{2}P_{\delta\delta}\partial^{2}P_{\Theta\Theta}$')
ax.plot(sep, ctot2, c='k', lw=2, label=r'$c^{2}_{\mathrm{s}} / a^{2}$')
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.grid(lw=0.2, ls='dashed', color='grey')
ax.yaxis.set_ticks_position('both')
ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
plt.savefig('../plots/EFT_nbody/c2_spat.png'.format(Lambda), bbox_inches='tight', dpi=120)
# print(ctot2[])
