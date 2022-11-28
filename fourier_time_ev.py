#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import numpy as np

j = 0
Nfiles = 595
# mode = 1
densities = np.empty(shape=(2, Nfiles))
a_list = np.empty(Nfiles)
a_list_gad = np.empty(Nfiles)

for mode in [1, 2, 10, 11, 12]:
    for j in range(Nfiles):
        filename = '/vol/aibn31/data1/mandar/data/modes/all_modes_sch_spt/dk2_{0:03d}.hdf5'.format(j)

        with h5py.File(filename, 'r') as hdf:
            ls = list(hdf.keys())
            a = np.array(hdf.get(str(ls[0])))
            dk2_spt = np.array(hdf.get(str(ls[1])))
            dk2_sch = np.array(hdf.get(str(ls[2])))
            # dk2_nbody = np.array(hdf.get(str(ls[2])))
            # dk2_zel = np.array(hdf.get(str(ls[4])))
            k = np.array(hdf.get(str(ls[3])))
            Lambda = np.array(hdf.get(str(ls[4])))

        den = np.array([dk2_spt[mode], dk2_sch[mode]]) # dk2_nbody[mode], dk2_zel[mode]]) #/ (np.array([a, a_gad, a, a])**2)
        a_list[j] = a
        # a_list_gad[j] = a_gad
        densities[:, j] = den[:]
    #errors wrt Zel
    # err_spt = (densities[0] - densities[3]) * 100 / (densities[3])
    # err_nbody = (densities[1] - densities[3]) * 100 / (densities[3])
    # err_sch = (densities[2] - densities[3]) * 100 / (densities[3])

    #erros wrt Sch
    err_spt = (densities[0] - densities[1]) * 100 / (densities[1])
    # err_nbody = (densities[1] - densities[2]) * 100 / (densities[2])


    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
    ax[0].set_title('k = {} [$h \; \mathrm{{Mpc}}^{{-1}}$]'.format(k[mode]))
    ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
    ax[1].set_xlabel(r'$a$', fontsize=14)

    ax[0].plot(a_list, densities[0], color='k', label='3SPT')
    # ax[0].plot(a_list_gad, densities[1], color='r', label='N-body')
    ax[0].plot(a_list, densities[1], color='b', label='Sch')
    # ax[0].plot(a_list, densities[3], color='yellow', ls='dashed', label='Zel')

    ax[1].axhline(0, color='b')
    ax[1].plot(a_list, err_spt, color='k')
    # ax[1].plot(a_list_gad, err_nbody, color='r')
    # ax[1].plot(a_list, err_sch, color='b')

    # ax[1].set_xlim(0, 30)
    # ax[1].set_ylim(-1, 5)
    # ax[1].minorticks_on(direction='in')
    ax[1].set_ylabel('% err', fontsize=14)

    for i in range(2):
        ax[i].tick_params(axis='both', which='both', direction='in')
        ax[i].ticklabel_format(scilimits=(-2, 3))
        ax[i].grid(lw=0.2, ls='dashed', color='grey')
        ax[i].yaxis.set_ticks_position('both')
        # ax[i].axvline(2, color='g', ls='dashed', label=r'a$_{\mathrm{sc}}$')

    ax[0].legend(fontsize=14, loc=2, bbox_to_anchor=(1,1))

    # plt.show()
    plt.savefig('/vol/aibn31/data1/mandar/plots/sch_hfix_run10/sch_vs_spt/k_{}.pdf'.format(mode), bbox_inches='tight', dpi=120)
