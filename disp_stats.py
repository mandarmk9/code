#!/usr/bin/env python3
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt

# paths = ['cosmo_sim_1d/sim_k_1_7/run1/', 'cosmo_sim_1d/sim_k_1_11/run1/', 'cosmo_sim_1d/sim_k_1_15/run1/']
# paths = ['cosmo_sim_1d/amps_sim_k_1_11/run1/', 'cosmo_sim_1d/amps2_sim_k_1_11/run1/']
# paths = ['cosmo_sim_1d/multi_k_sim/run1/']
# path = 'cosmo_sim_1d/amp_ratio_test/run1/'
paths = ['cosmo_sim_1d/sim_k_1_11/run1/']
Nfiles = 51

def k_NL_ext(path, Nfiles, ind='mean'):
    print(path)
    a_list, k_NL = [], []
    for j in range(Nfiles):
        moments_filename = 'output_hierarchy_{0:04d}.txt'.format(j)
        nbody_filename = 'output_{0:04d}.txt'.format(j)
        moments_file = np.genfromtxt(path + moments_filename)
        a = moments_file[:,-1][0]
        nbody_file = np.genfromtxt(path + nbody_filename)
        Psi = nbody_file[:,1]
        x_nbody = nbody_file[:,-1]
        v_nbody = nbody_file[:,2]

        Psi *= (2*np.pi)
        if ind == 'mean':
            k_NL.append(1/np.mean(np.abs(Psi)))
        elif ind == 'median':
            k_NL.append(1/np.median(np.abs(Psi)))

        a_list.append(a)
        # std = np.std([Psi], ddof=1)
        # print('sigma = ', std)
        # print('mean(|Psi|)', np.mean(np.abs(Psi)))
        print('a = {}, k_NL = {}\n'.format(np.round(a,4), np.round(1/np.mean(np.abs(Psi)), 3)))
    return a_list, k_NL

# k_NL_lists = []
# # ind = 'mean'
# ind = 'median'
#
# for j in range(len(paths)):
#     lists = k_NL_ext(paths[j], Nfiles, ind)
#     a_list = lists[0]
#     k_NL_lists.append(lists[1])
#     df = pandas.DataFrame(data=k_NL_lists)
#     pickle.dump(df, open('./{}/k_NL_lists_{}.p'.format(paths[j], ind), 'wb'))

# path = paths[0]
# a_list = np.zeros(Nfiles)
# for j in range(Nfiles):
#     a_list[j] = np.genfromtxt(path + 'aout_{0:04d}.txt'.format(j))
#     print('a = ', a_list[j])
# df = pandas.DataFrame(data=a_list)
# pickle.dump(df, open('./{}/a_list.p'.format(paths[0]), 'wb'))

# df = pandas.DataFrame(data=k_NL_lists)
# pickle.dump(df, open('./data/k_NL_lists_{}.p'.format(ind), 'wb'))

data = pickle.load(open("./{}/k_NL_lists_{}.p".format(paths[0], 'mean'), "rb" ))
data_median = pickle.load(open("./{}/k_NL_lists_{}.p".format(paths[0], 'median'), "rb" ))
a_list = np.array(pickle.load(open('./{}/a_list.p'.format(paths[0]), "rb" ))[0])

# A = [-0.05, 1, -0.04, 2, -0.03, 3, -0.02, 4, -0.01, 5, -1, 11]
A = [-0.1, 1, -0.5, 11]

from zel import initial_density
x = np.arange(0, 1, 1/1000)
a_sc = 1 / np.max(initial_density(x, A, 1))
# # print(a_sc)
k_NL_mean = np.array(data)[0]
k_NL_median = np.array(data_median)[0]

# a_list, k_NL_mean = k_NL_ext(path, Nfiles, ind='mean')
# a_list, k_NL_median = k_NL_ext(path, Nfiles, ind='median')


k_NL_lists = [k_NL_mean, k_NL_median]
labels = ['mean', 'median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']
# labels = [r'\texttt{sim\_multi\_k}: mean', r'\texttt{sim\_multi\_k}: median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']
# labels = [r'\texttt{amp\_ratio\_test}: mean', r'\texttt{amp\_ratio\_test}: median'] #[r'\texttt{sim\_k\_1\_7}', r'\texttt{sim\_k\_1\_11}', r'\texttt{sim\_k\_1\_15}']

colours = ['b', 'r', 'k']
linestyles = ['solid', 'dashdot', 'dotted']
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots()
plt.rcParams.update({"text.usetex": True})

ax.set_xlabel(r'$a$', fontsize=18)
ax.set_ylabel(r'$k_{\mathrm{NL}}$', fontsize=18)
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
ax.set_title(r'\texttt{sim\_k\_1\_11}', fontsize=18)
ax.axhline(1, c='brown', ls='dashed', label=r'$k=1$', lw=1)
ax.axhline(3, c='k', ls='dashed', label=r'$k=3$', lw=1)
ax.axhline(9, c='magenta', ls='dashed', label=r'$k=9$', lw=1)
ax.axvline(a_sc, c='seagreen', ls='dashed', label=r'$a_{\mathrm{sc}}$', lw=1)
# ax.axhline(5, c='brown', ls='dashed', label=r'$k=5$', lw=1)
ax.minorticks_on()
ax.yaxis.set_ticks_position('both')

for j in range(2):
    ax.plot(a_list, k_NL_lists[j], c=colours[j], lw=1.5, ls=linestyles[j], label=labels[j])

plt.legend(fontsize=13)
# # plt.savefig('../plots/test/paper_plots/k_NL_ev_med.png', dpi=300)
plt.savefig('../plots/test/new_paper_plots/k_NL_ev.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('../plots/multi_k_sim/k_NL_multi.png', dpi=150, bbox_inches='tight')
# plt.savefig('../plots/ratio_test/k_NL_multi.png', dpi=150, bbox_inches='tight')
plt.close()

# plt.show()

# data = pickle.load(open("./data/k_NL_lists.p", "rb" ))
# data_median = pickle.load(open("./data/k_NL_lists_median.p", "rb" ))
# a_list = np.array(pickle.load(open("./data/sim_k_1_11_a_list.p", "rb" ))[0])
# # k_NL_7 = [data[j][0] for j in range(data.shape[1])]
# k_NL_11 = [data[j][1] for j in range(data.shape[1])]
# k_NL_11_median = [data_median[j][1] for j in range(data_median.shape[1])]
# k_NL_15 = [data[j][2] for j in range(data.shape[1])]
#
# # k_NL_11 = [data[j][1] for j in range(data.shape[1])]
# # k_NL_11_median = [data_median[j][1] for j in range(
# # k_NL_lists = [k_NL_11, k_NL_11_median] #[k_NL_7, k_NL_11, k_NL_15]



# ax.set_ylabel(r'$N$')
# ax.set_xlabel(r'$\Psi\;[(2\pi h)^{-1}\;\mathrm{Mpc}]$')
#
# ax.hist(Psi, bins=50)
# # ax.hist(x_nbody, bins=50)
#
# ax.set_title('a = {}'.format(a))
# ax.set_title(title_txt, fontsize=18)
