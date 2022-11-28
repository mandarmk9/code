#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import h5py
import numpy as np

from functions import plotter2, read_density
from scipy.interpolate import interp1d


def spec_nbody(path, Nfiles, mode):
    a_list = np.zeros(Nfiles)
    P_nb = np.zeros(Nfiles)
    print('\npath = {}'.format(path))
    for j in range(Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        x_cell = moments_file[:,0]
        dk_par, a, dx = read_density(path, j)
        L = 1.0
        x = np.arange(0, L, dx)

        M0_par = np.real(np.fft.ifft(dk_par))
        M0_par /= np.mean(M0_par)
        f_M0 = interp1d(x, M0_par, fill_value='extrapolate')
        M0_par = f_M0(x_cell)

        M0_k = np.fft.fft(M0_par - 1) / M0_par.size
        P_nb[j] = (np.real(M0_k * np.conj(M0_k)))[mode]
        a_list[j] = a
        print('a = {}'.format(a))

    return a_list, P_nb / a_list**2 / 1e-4

mode = 1

path = 'cosmo_sim_1d/nbody_new_run2/'
Nfiles = 33
a_list_1, P_nb = spec_nbody(path, Nfiles, mode)

path = 'cosmo_sim_1d/nbody_phase_inv/'
Nfiles = 51
a_list_2, P_nb_phase_inv = spec_nbody(path, Nfiles, mode)

#for plotting the spectra
plots_folder = 'test'
xaxes = [a_list_1, a_list_2]
yaxes = [P_nb, P_nb_phase_inv]
colours = ['b', 'k']
labels = ['sim_k_1_11', 'sim_k_1_11_phase_inv']
savename = 'nbody_spec'
linestyles = ['solid', 'dashed']
fig, ax = plt.subplots()

ax.set_title(r'$k = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(mode), fontsize=12)
ax.set_xlabel(r'$a$', fontsize=14)
ax.set_ylabel(r'$a^{-2}P(k=1, a) \times 10^{4}$', fontsize=14)

for i in range(len(yaxes)):
    ax.plot(xaxes[i], yaxes[i], c=colours[i], ls=linestyles[i], lw=2.5, label=labels[i])
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in')
ax.ticklabel_format(scilimits=(-2, 3))
ax.yaxis.set_ticks_position('both')
ax.legend(fontsize=11)#, loc=2, bbox_to_anchor=(1,1))
# plt.savefig('../plots/{}/{}.png'.format(plots_folder, savename), bbox_inches='tight', dpi=150)
# plt.savefig('../plots/{}/{}.pdf'.format(plots_folder, savename), bbox_inches='tight', dpi=300)
# plt.close()
plt.show()
