#!/usr/bin/env python3
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing as mp
from scipy.optimize import curve_fit
from functions import write_sim_data, read_sim_data, param_calc_ens, sub_find, AIC
import pickle
import pandas
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# path = 'cosmo_sim_1d/sim_k_1_7/run1/'
# A = [-0.05, 1, -0.5, 7]

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# A = []#[-0.025, 1, -0.0, 11]

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# A = [-0.05, 1, -0.5, 11]

# path = 'cosmo_sim_1d/sim_k_1_15/run1/'
# A = [-0.05, 1, -0.5, 15]
#
# # path = 'cosmo_sim_1d/sim_k_1/run1/'
# # A = []
# # # flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
#

# path = 'cosmo_sim_1d/new_sim_k_1_11/run1/'
# A = [-0.05, 1, -0.5, 11]

# path = 'cosmo_sim_1d/final_sim_k_1_11/run1/'
# A = [-0.05, 1, -0.5, 11]
#
# n_runs = 16
# mode = 1
# kinds = ['sharp', 'gaussian']
# kind_txts = ['sharp cutoff', 'Gaussian smoothing']
# which = 0
#
# tmp_st = time.time()
# zero = 0
# Nfiles = 101
# def write(j, Lambda, path, A, kind, mode, run, folder_name=''):
#     path = path[:-2] + '{}/'.format(run)
#     write_sim_data(j, Lambda, path, A, kind, mode, folder_name)
#
# kind = kinds[which]
# kind_txt = kind_txts[which]
# for Lambda in range(2,7):
#     print('Lambda = {} ({})'.format(Lambda, kind_txt))
#     # folder_name = '/new_data_{}/L{}'.format(kind, Lambda)
#     Lambda *= (2*np.pi)
#     for j in range(zero, Nfiles):#, Nfiles):
#         print('Writing {} of {}'.format(j+1, Nfiles))
#         tasks = []
#         for run in range(1, 1+n_runs):
#             p = mp.Process(target=write, args=(j, Lambda, path, A, kind, mode, run,))
#             tasks.append(p)
#             p.start()
#         for task in tasks:
#             p.join()
# tmp_end = time.time()
# print('multiprocessing takes {}s'.format(np.round(tmp_end-tmp_st, 3)))

# # # # # print(flags)
# # # # j = 0
# # # a, corr12, corr21 = [], [], []
# # # for j in range(1):
# # #     j = 4
# # #     Lambda = 5*(np.pi*2)
# # #     kind = 'sharp'
# # #     sol = param_calc(j, Lambda, path, A, mode, kind, nruns)
# # #     # print('a = ', sol[0])
# # #     # a.append(sol[0])
# # #     # corr12.append(sol[1][1,2])
# # #     # corr21.append(sol[1][2,1])
# # #
# # # # print(sol[1])
# # # # import seaborn as sns
# # # # lab = [r'$C_{0}$', r'$C_{1}$', r'$C_{2}$']
# # # # fig, ax = plt.subplots()
# # # # ax.set_title('Correlation matrix, a = {}, $\Lambda = {}$'.format(sol[0], int(Lambda/(2*np.pi))))
# # # # sns.set(font_scale=1.5)
# # # # hm = sns.heatmap(sol[1], cbar=True, annot=True, square=True, fmt='.4f', annot_kws={'size': 12}, cmap='coolwarm', xticklabels=lab, yticklabels=lab)
# # # # plt.show()
# #
# # # plt.plot(a, corr12, c='k', lw=2)
# # # plt.show()
# #
# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, err1_list, err2_list, chi_list = [], [], [], [], [], [], []
# Nfiles = 61
# cs2_list = np.zeros(Nfiles)
# cv2_list = np.zeros(Nfiles)
# a_list = np.zeros(Nfiles)
#
# # for j in [2, 11, 30, 48]:#range(Nfiles):
# #     # j = 0
# #
# #     for Lambda in range(2,8):
# #         print(j, Lambda)
# #         Lambda *= 2*np.pi
#
# Lambda = 3 * (2*np.pi)
# # file_num = 25
# mode = 1
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # taus = []
#
# for file_num in range(51):
#     a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb_a, P_1l_a_tr, d1k = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs=8, n_use=10)
#
#     plots_folder = 'sim_k_1_11/sm/{}_tau'.format(kind)
#     fig, ax = plt.subplots()
#     ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=12)
#     ax.plot(x, tau_l, c='b', lw=2, label=r'$[\tau]_{\Lambda}$')
#     ax.plot(x, fit, c='k', ls='dashed', lw=2, label=r'fit to $[\tau]_{\Lambda}$') #label=r'$\left<[\tau]_{\Lambda}\right>$ (fit)')
#     # # ax.plot(x, fit_bau, c='y', ls='dotted', lw=2, label=r'fit from $B^{+12}$')
#
#     ax.fill_between(x, tau_l-yerr, tau_l+yerr, color='gray', alpha=0.5)
#     # ax.errorbar(x, tau_l, yerr=yerr, ecolor='r')#, errorevery=10000, ecolor='r')
#     # stoch = (tau_l - fit)
#     # ax.plot(x, stoch, c='r', lw=2, label=r'$[\tau]_{\Lambda} - \left<[\tau]_{\Lambda}\right>$')
#
#     # ax.plot(x, tau_d2, c='k', ls='dashed', lw=2, label=r'$[\tau]^{\partial^{2}}_{\Lambda}$')
#
#     ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=12)
#     ax.set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=12)
#     ax.minorticks_on()
#     ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
#     ax.ticklabel_format(scilimits=(-2, 3))
#     # ax.grid(lw=0.2, ls='dashed', color='grey')
#     ax.legend(fontsize=12, bbox_to_anchor=(1,1))
#     ax.yaxis.set_ticks_position('both')
#     # plt.show()
#     plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, file_num), bbox_inches='tight', dpi=150)
#     plt.close()


# taus.append(tau_l)
# #we already save the \tau from run1 in the last line. the next loop should run from 2 to 5 (or 9)
# tau_l_0 = read_sim_data(path, Lambda, kind, file_num)[5]
# taus.append(tau_l_0)
# for run in range(2, 9):
#     print(run)
#     path = path[:-2] + '{}/'.format(run)
#     taus.append(read_sim_data(path, Lambda, kind, file_num)[1])
#
# print(len(taus))
# errors = [(tau - tau_l) * 100 / tau_l for tau in taus]
# linestyles = ['solid', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)), 'dashdot', 'dashed', 'dotted']
# colors = ['brown', 'darkcyan', 'dimgray', 'violet', 'orange', 'cyan', 'b', 'r', 'k']
# labels = [r'$\left<[\tau]_{\Lambda}\right>$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
# print(len(linestyles), len(colors), len(labels), len(errors))
#
# plots_folder = 'final_runs/mean_tau'
# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
# ax[0].set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$'.format(a, int(Lambda/(2*np.pi))), fontsize=14)
#
# ax[0].plot(x, tau_l, c=colors[0], lw=2.5, ls=linestyles[0], label=labels[0])
#
# # for i in range(8):
# #     ax[0].plot(x, taus[i], c=colors[i+1], lw=2.5, ls=linestyles[i+1], label=labels[i+1])
# #     ax[1].plot(x, errors[i], c=colors[i+1], lw=2.5, ls=linestyles[i+1])
#
# ax[0].set_ylabel(r'$[\tau]_{\Lambda}\;\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=14)
# ax[1].set_ylabel('% err', fontsize=14)
# ax[1].set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$', fontsize=14)
# ax[0].legend(fontsize=12, bbox_to_anchor=(1,1))
#
# ax[1].axhline(0, c='brown', lw=2.5)
# for i in range(2):
#     ax[i].minorticks_on()
#     ax[i].tick_params(axis='both', which='both', direction='in', labelsize=12)
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].yaxis.set_ticks_position('both')
# plt.savefig('../plots/{}/tau_{}.png'.format(plots_folder, j), bbox_inches='tight', dpi=150)
# plt.close()
# plt.show()


# plt.plot(a_list, cs2_list, lw=2, c='k')
# plt.plot(a_list, cv2_list, lw=2, c='b')
# plt.show()



# Nfiles = 61
# a_list, err1_list, err2_list, C1_list, C2_list = [], [], [], [], []
# fig, ax = plt.subplots()
# for j in range(Nfiles):
#     sol = param_calc(j, Lambda, path, A, mode, kind, nruns)
#     a_list.append(sol[0])
#     err1_list.append(sol[2])
#     err2_list.append(sol[3])
#     C1_list.append(sol[4])
#     C2_list.append(sol[5])
#     print('a = ', sol[0])
#
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
#
# a_list = np.array(a_list)
# C1_list = np.array(C1_list)
# C2_list = np.array(C2_list) * (100 / np.sqrt(a_list))
# err1_list = np.array(err1_list)
# err2_list = np.array(err2_list)
#
# plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'])
#
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.set_xlabel(r'$a$', fontsize=16)
#
# ax.fill_between(a_list, C1_list+err1_list, C1_list-err1_list, color='midnightblue', alpha=0.55, zorder=2, label=r'$\sigma_{1}$')
# ax.fill_between(a_list, C2_list+err1_list, C2_list-err1_list, color='darkslategray', alpha=0.55, zorder=2, label=r'$\sigma_{2}$')
# ax.plot(a_list, C1_list, lw=2, c='b', zorder=3, marker='o', label=r'$C_{1}$')
# ax.plot(a_list, C2_list, lw=2, c='k', ls='dashed', zorder=3, marker='v', label=r'$C_{2}$')
#
# ax.set_ylabel(r'$C_{i}$', fontsize=16)
#
# plt.legend(fontsize=11)
# plt.subplots_adjust(hspace=0)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# ax.yaxis.set_ticks_position('both')
#
# plt.show()
# # plt.savefig('../plots/test/ctot2/err/ctot2_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.close()
#


#
# Nfiles = 61
# a_list, err1_list, err2_list, C1_list, C2_list = [], [], [], [], []
# fig, ax = plt.subplots(2, 1, sharex=True)
# for j in range(Nfiles):
#     sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use)
#     a_list.append(sol[0])
#     err1_list.append(sol[2])
#     err2_list.append(sol[3])
#     C1_list.append(sol[4])
#     C2_list.append(sol[5])
#     print('a = ', sol[0])
#
#     if flags[j] == 1:
#         for i in range(2):
#             sc_line = ax[i].axvline(a_list[j], c='teal', lw=1, zorder=1)
#
# C1_list = np.array(C1_list)
# C2_list = np.array(C2_list)
# err1_list = np.array(err1_list)
# err2_list = np.array(err2_list)
#
# plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'])
#
# ax[0].set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax[1].set_xlabel(r'$a$', fontsize=16)
#
# ax[0].fill_between(a_list, C1_list+err1_list, C1_list-err1_list, color='midnightblue', alpha=0.55, zorder=2, label=r'$1\sigma$ err on $C_{1}$')
# ax[0].scatter(a_list, C1_list, s=40, c='b', zorder=3)
# ax[0].plot(a_list, C1_list, c='b', lw=1.5, zorder=4)
# ax[0].set_ylabel(r'$C_{1}$', fontsize=16)
#
# ax[1].fill_between(a_list, C2_list+err1_list, C2_list-err1_list, color='darkslategray', alpha=0.55, zorder=2, label=r'$1\sigma$ err on $C_{2}$')
# ax[1].scatter(a_list, C2_list, s=40, c='k', zorder=3)
# ax[1].plot(a_list, C2_list, c='k', lw=1.5, zorder=4)
# ax[1].set_ylabel(r'$C_{2}$', fontsize=16)
#
# plt.subplots_adjust(hspace=0)
#
# for i in range(2):
#     ax[0].minorticks_on()
#     ax[0].tick_params(axis='both', which='both', direction='in')
#     ax[0].yaxis.set_ticks_position('both')
# # plt.show()
# plt.savefig('../plots/test/ctot2/err/ctot2_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# plt.close()

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# mode = 1
# A = [-0.05, 1, -0.5, 11]
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
# # kind = 'gaussian'
# # kind_txt = 'Gaussian smoothing'
# n_runs = 8
# n_use = 10
# Lambda_list = np.arange(2, 6)
# nums = [[0, 12], [33, 50]]
# ctot2_0_list, ctot2_1_list, ctot2_2_list, error_list = [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]]
# a_list = [[[], []], [[], []]]
#
# for i1 in range(2):
#     print(i1)
#     for i2 in range(2):
#         print(i2)
#         file_num = nums[i1][i2]
#         c0, c1, c2, err = [], [], [], []
#         for i3 in range(Lambda_list.size):
#             print(Lambda_list[i3])
#             folder_name = ''#'/new_data_{}/L{}'.format('sharp', Lambda_list[i3])
#             Lambda = Lambda_list[i3] * (2*np.pi)
#             sol = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs, n_use, folder_name)
#             a_list[i1][i2] = float(sol[0])
#             c0.append(float(sol[2]))
#             c1.append(float(sol[3]))
#             c2.append(float(sol[4]))
#             err.append(float(sol[14]))
#             print('a = ', float(sol[0]))
#
#         ctot2_0_list[i1][i2] = c0
#         ctot2_1_list[i1][i2] = c1
#         ctot2_2_list[i1][i2] = c2
#         error_list[i1][i2] = err
#
# error_list = np.array(error_list)
# ctot2_0_list = np.array(ctot2_0_list)
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
# for i in range(2):
#     print(i)
#     # ax[0,0].set_ylabel(r'$c^{2}_{v}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     # ax[0,0].set_ylabel(r'$c^{2}_{s}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     ax[0,0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', y=-0.1, fontsize=16)
#     ax[1,1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), x=0, fontsize=16)
#     ax[i,1].tick_params(labelleft=False, labelright=True)
#     for j in range(2):
#         ax[i,j].set_title(r'$a = {}$'.format(np.round(a_list[i][j], 3)), x=0.25, y=0.9)
#         print(ctot2_0_list[i][j])
#         # ax[i,j].fill_between(Lambda_list, ctot2_0_list[i][j]-error_list[i][j], ctot2_0_list[i][j]+error_list[i][j], color='darkslategray', alpha=0.35, rasterized=True)
#
#
#         ax[i,j].plot(Lambda_list, ctot2_1_list[i][j], c='cyan', lw=1.5, marker='v', label=r'M&W')
#         ax[i,j].plot(Lambda_list, ctot2_2_list[i][j], c='orange', lw=1.5, marker='*', label=r'$B^{+12}$')
#         ax[i,j].errorbar(Lambda_list, ctot2_0_list[i][j], yerr=error_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
#         # ax[i,j].plot(Lambda_list, ctot2_0_list[i][j], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')
#         ax[i,j].minorticks_on()
#         ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#         ax[i,j].yaxis.set_ticks_position('both')
#
#     # if kind == 'sharp':
#     #     ax[0,0].set_ylim(0.201, 0.25)
#     #     ax[1,0].set_ylim(-0.25, 1.8)
#     #     ax[0,1].set_ylim(0.3, 2.5)
#     #     ax[1,1].set_ylim(0, 1.3)
#     #
#     # elif kind == 'gaussian':
#     #     # ax[1,0].set_ylim(0, 1.7)
#     #     ax[0,1].set_ylim(-0.35, 1.5)
#
#
#     else:
#         pass
#
#
# plt.legend(fontsize=12, bbox_to_anchor=(1,1))
# fig.align_labels()
# plt.subplots_adjust(hspace=0, wspace=0)
# # plt.savefig('../plots/test/ctot2/Lam_dep/ctot2_lambda_dep_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# # plt.savefig('../plots/test/ctot2/Lam_dep/ctot2_lambda_dep_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.savefig('../plots/sim_k_1_11/ctot2_lambda_dep_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.close()
# # plt.show()

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
mode = 1
A = [-0.05, 1, -0.5, 11]
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
n_runs = 8
n_use = n_runs-1
Lambda_list = np.arange(3, 7)
# nums = [10, 15, 23]
nums = [0, 14, 23]

nbins_x, nbins_y, npars = 20, 20, 3
fm = 'curve_fit'

ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, error_list = [[], [], []],  [[], [], []], [[], [], []], [[], [], []], [[], [], []]
a_list = []

# for j in range(len(nums)):
#     c0, c1, c2, c3, err = [], [], [], [], []
#     file_num = nums[j]
#     for i in range(Lambda_list.size):
#         Lambda = Lambda_list[i] * (2*np.pi)
#         sol = param_calc_ens(file_num, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
#         a = float(sol[0])
#         c0.append(float(sol[2]))
#         c1.append(float(sol[3]))
#         c2.append(float(sol[4]))
#         c3.append(float(sol[-2]))
#         err.append(float(sol[-1]))
#     print('a = ', a)
#
#     a_list.append(a)
#     ctot2_0_list[j] = c0
#     ctot2_1_list[j] = c1
#     ctot2_2_list[j] = c2
#     ctot2_3_list[j] = c3
#
#     error_list[j] = err
#
# error_list = np.array(error_list)
#
# file = open("./{}/ctot2_lambda_dep_{}.p".format(path, kind), 'wb')
# df = pandas.DataFrame(data=[a_list, ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, error_list])
# pickle.dump(df, file)
# file.close()

file = open("./{}/ctot2_lambda_dep_{}.p".format(path, kind), 'rb')
read_file = pickle.load(file)
a_list, ctot2_0_list, ctot2_1_list, ctot2_2_list, ctot2_3_list, error_list = np.array(read_file)
file.close()

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

ax[0].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', fontsize=20)
ax[1].set_xlabel(r'$\Lambda\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(kind_txt), fontsize=20)
ax[2].set_ylabel(r'$c^{2}_{\mathrm{tot}}\;[\mathrm{km^{2}\,s}^{-2}]$', fontsize=20)
ax[2].yaxis.set_label_position('right')
for i in range(3):

    ax[i].set_title(r'$a = {}$'.format(np.round(a_list[i], 3)), fontsize=20)

    ctot2_line, = ax[i].plot(Lambda_list, ctot2_1_list[i], c='c', lw=1.5, marker='v')#, label=r'M\&W')
    ctot2_2_line, = ax[i].plot(Lambda_list, ctot2_2_list[i], c='orange', lw=1.5, marker='*')#, label=r'$B^{+12}$')
    ctot2_3_line, = ax[i].plot(Lambda_list, ctot2_0_list[i], c='k', lw=1.5, marker='o')#, label=r'fit to $[\tau]_{\Lambda}$')
    # ctot2_4_line, = ax[i].plot(Lambda_list, ctot2_3_list[i], c='xkcd:dried blood', lw=1.5, marker='+')#, label=r'FDE')
    # ctot2_4_err = ax[i].fill_between(Lambda_list, ctot2_3_list[i]-error_list[i], ctot2_3_list[i]+error_list[i], color='darkslategray', alpha=0.55, rasterized=True)

    # ax[i].errorbar(Lambda_list, ctot2_0_list[i], yerr=error_list[i], c='k', lw=1.5, marker='o', label=r'fit to $[\tau]_{\Lambda}$')

    ax[i].minorticks_on()
    ax[i].tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax[i].yaxis.set_ticks_position('both')
    #
    # if kind == 'sharp':
    #     ax[0,0].set_ylim(0.201, 0.25)
    #     ax[1,0].set_ylim(-0.25, 1.8)
    #     ax[0,1].set_ylim(0.3, 2.5)
    #     ax[1,1].set_ylim(0, 1.3)
    #
    # elif kind == 'gaussian':
    #     # ax[1,0].set_ylim(0, 1.7)
    #     ax[0,1].set_ylim(-0.35, 1.5)
    #
    #
    # else:
    #     pass

# print(type(ctot2_line), type(ctot2_4_err))
# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line, (ctot2_4_line, ctot2_4_err)], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line, ctot2_4_line], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=3, framealpha=1)

# plt.legend(handles=[ctot2_3_line, ctot2_line, ctot2_2_line], labels=[r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=4, framealpha=1)

# plt.legend(fontsize=14, bbox_to_anchor=(1,1))
fig.align_labels()
plt.subplots_adjust(wspace=0.2)
plt.savefig('../plots/test/new_paper_plots/ctot2_lambda_dep_{}_no_fde.png'.format(kind), bbox_inches='tight', dpi=150)
# plt.savefig('../plots/test/new_paper_plots/ctot2_lambda_dep_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
# plt.close()
# plt.show()


# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# # path = 'cosmo_sim_1d/final_sim_k_1_11/run1/'
#
# A = [-0.05, 1, -0.5, 11]
# mode = 1
# Lambda = (2*np.pi) * 3
# file_nums = [0, 13, 23]
# # file_nums = [7, 15, 23]
#
# # file_nums = [8, 9, 10]
#
# n_runs = 8
# n_use = n_runs-1
# kinds = ['sharp', 'gaussian']
# kinds_txt = ['sharp cutoff', 'Gaussian smoothing']
# # fm = 'WLS'
# # fm = 'curve_fit'
# # fm = 'lmfit'
# fm = ''
# nbins_x, nbins_y, npars = 10, 10, 6
#
# for j in range(1, 2):
#     kind = kinds[j]
#     kind_txt = kinds_txt[j]
#
#     sol_0 = param_calc_ens(file_nums[0], Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
#     sol_1 = param_calc_ens(file_nums[1], Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
#     sol_2 = param_calc_ens(file_nums[2], Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=fm, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
#
#     x = sol_0[1]
#     a_list = [sol_0[0], sol_1[0], sol_2[0]]
#     tau_list = [sol_0[12], sol_1[12], sol_2[12]]
#     fit_list = [sol_0[13], sol_1[13], sol_2[13]]
#     yerr_list = [sol_0[11], sol_1[11], sol_2[11]]
#     chi_list = [sol_0[10], sol_1[10], sol_2[10]]
#     taus = [sol_0[-2], sol_1[-2], sol_2[-2]]
#     x_binned = [sol_0[-1], sol_1[-1], sol_2[-1]]
#
#     # C_list = [sol_0[4], sol_1[4], sol_2[4]]
#
#     plt.rcParams.update({"text.usetex": True})
#     plt.rcParams.update({"font.family": "serif"})
#
#     fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})
#
#     fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=20)
#     ax[0].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
#     ax[2].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=22)
#     ax[2].yaxis.set_label_position('right')
#     # ax[2].tick_params(labelleft=False, labelright=True)
#     ax[1].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=20)
#     for j in range(3):
#         err_plus = np.array(taus[j])+np.array(yerr_list[j])
#         err_minus = np.array(taus[j])-np.array(yerr_list[j])
#         print(len(x_binned[j]))
#         ax[j].set_title(r'$a = {}$'.format(np.round(a_list[j], 3)), x=0.15, y=0.9, fontsize=18)
#         ax[j].plot(x, tau_list[j], c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
#         # ax[j].fill_between(x, tau_list[j]-yerr_list[j], tau_list[j]+yerr_list[j], color='darkslategray', alpha=0.35, rasterized=True)
#         # ax[j].fill_between(x, (np.array(tau_list[j]) - np.array(yerr_list[j])), (np.array(tau_list[j]) + np.array(yerr_list[j])), color='darkslategray', alpha=0.35, rasterized=True)
#         # ax[j].fill_between(np.array(x_binned[j]), err_minus, err_plus, color='darkslategray', alpha=0.35, rasterized=True)
#         ax[j].errorbar(np.array(x_binned[j]), err_minus, err_plus, color='darkslategray', marker='o')#, alpha=0.35, rasterized=True)
#
#         ax[j].plot(x, fit_list[j], c='k', lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
#         ax[j].minorticks_on()
#         ax[j].tick_params(axis='both', which='both', direction='in', labelsize=15)
#         ax[j].yaxis.set_ticks_position('both')
#
#         # chi_str = r'$\chi^{{2}}/{{\mathrm{{d.o.f.}}}} = {}$'.format(np.round(chi_list[j], 3))
#         # ax[j].text(0.35, 0.05, chi_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[j].transAxes)
#
#
#     # for tick in ax[1,1].yaxis.get_majorticklabels():
#     #     tick.set_horizontalalignment("right")
#
#     plt.legend(fontsize=18, bbox_to_anchor=(1, 1.25))
#     fig.align_labels()
#     plt.subplots_adjust(wspace=0.17)
#     plt.savefig('../plots/test/new_paper_plots/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
#     plt.close()
#     # plt.show()
#     # plt.savefig('../plots/test/new_paper_plots/fit_test/tau_fits_{}_delsq.png'.format(kind), bbox_inches='tight', dpi=150)
#     # plt.close()



# path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
# n_runs = 24
# A = [-0.05, 1, -0.5, 11]

# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# n_runs = 8
# A = [-0.05, 1, -0.5, 11]
#
# # path = 'cosmo_sim_1d/sim_k_1/run1/'
#
# # A = [-0.05, 1, -0.0, 11]
#
# mode = 1
# Lambda = (2*np.pi) * 3
# file_nums = [0, 12, 23, 35]
# # file_nums = [0, 12, 23, 23]
# # file_nums = [11, 12, 13, 14]
#
# # file_nums = [0, 10, 15, 23]
# # file_nums = [0, 1, 2, 3]
#
# modes = True #set this to false only if looking a single sim (instead of an ensemble)
# n_use = n_runs-1
# kinds = ['sharp', 'gaussian']
# kinds_txt = ['sharp cutoff', 'Gaussian smoothing']
# # fitting_method = 'lmfit'
# fitting_method = 'curve_fit'
#
# # # fitting_method = 'WLS'
# # sub = np.random.choice(62500, n_use)
# # # sub = [55833, 59515, 24295, 26639, 1935, 40438, 30062]
# # # n_ev = int(62500 / 6)
# # # sub = np.arange(0, 62500)[0::n_ev]
# sub = sub_find(7, 62500)
#
# folder_name = ''#'/new_data_{}/L{}'.format('sharp', int(Lambda/(2*np.pi)))
# for j in range(1):
#     kind = kinds[j]
#     kind_txt = kinds_txt[j]
#
#     sol_0 = param_calc_ens(file_nums[0], Lambda, path, A, mode, kind, n_runs, n_use, folder_name, modes, fitting_method, sub=sub)
#     sol_1 = param_calc_ens(file_nums[1], Lambda, path, A, mode, kind, n_runs, n_use, folder_name, modes, fitting_method, sub=sub)
#     sol_2 = param_calc_ens(file_nums[2], Lambda, path, A, mode, kind, n_runs, n_use, folder_name, modes, fitting_method, sub=sub)
#     sol_3 = param_calc_ens(file_nums[3], Lambda, path, A, mode, kind, n_runs, n_use, folder_name, modes, fitting_method, sub=sub)
#
#     x = sol_0[1]
#     a_list = [[sol_0[0], sol_1[0]], [sol_2[0], sol_3[0]]]
#     tau_list = [[sol_0[12], sol_1[12]], [sol_2[12], sol_3[12]]]
#     fit_list = [[sol_0[13], sol_1[13]], [sol_2[13], sol_3[13]]]
#     yerr_list = [[sol_0[11], sol_1[11]], [sol_2[11], sol_3[11]]]
#     chi_list = [[sol_0[10], sol_1[10]], [sol_2[10], sol_3[10]]]
#     # C_list = [[sol_0[4], sol_1[4]], [sol_2[4], sol_3[4]]]
#     plt.rcParams.update({"text.usetex": True})
#     plt.rcParams.update({"font.family": "serif"})
#
#     fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
#     fig.suptitle(r'$\Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16, x=0.5, y=0.92)
#     for i in range(2):
#         ax[i,0].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#         ax[i,1].set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#         ax[i,1].yaxis.set_label_position('right')
#
#         ax[1,i].set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
#         ax[i,1].tick_params(labelleft=False, labelright=True)
#         ax[1,i].set_title(r'$a = {}$'.format(np.round(a_list[1][i], 3)), x=0.15, y=0.9)
#         ax[0,i].set_title(r'$a = {}$'.format(np.round(a_list[0][i], 3)), x=0.15, y=0.9)
#         for j in range(2):
#             # ax[i,j].errorbar(x, tau_list[i][j], yerr=yerr_list[i][j], ecolor='r', errorevery=10000, c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].plot(x, tau_list[i][j], c='b', lw=1.5, label=r'$\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].fill_between(x, tau_list[i][j]-yerr_list[i][j], tau_list[i][j]+yerr_list[i][j], color='darkslategray', alpha=0.35, rasterized=True)
#             ax[i,j].plot(x, fit_list[i][j], c='k', lw=1.5, ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
#             ax[i,j].minorticks_on()
#             ax[i,j].tick_params(axis='both', which='both', direction='in', labelsize=13.5)
#             # err_str = r'$C_{{0}} = {}$'.format(np.round(C_list[i][j][0], 3)) + '\n' + r'$C_{{1}} = {}$'.format(np.round(C_list[i][j][1], 3)) + '\n' + r'$C_{{2}} = {}$'.format(np.round(C_list[i][j][2], 3))
#             # ax[i,j].text(0.35, 0.05, err_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i,j].transAxes)
#             chi_str = r'$\chi^{{2}}/{{\mathrm{{d.o.f.}}}} = {}$'.format(np.round(chi_list[i][j], 3))
#             ax[i,j].text(0.35, 0.05, chi_str, bbox={'facecolor': 'white', 'alpha': 0.75}, usetex=True, fontsize=12, transform=ax[i,j].transAxes)
#
#             ax[i,j].yaxis.set_ticks_position('both')
#
#     # for tick in ax[1,1].yaxis.get_majorticklabels():
#     #     tick.set_horizontalalignment("right")
#
#     plt.legend(fontsize=12, bbox_to_anchor=(0.975, 2.2))
#     fig.align_labels()
#     plt.subplots_adjust(hspace=0, wspace=0)
#     # plt.savefig('../plots/test/new_paper_plots/tau_fits_{}.pdf'.format(kind), bbox_inches='tight', dpi=300)
#     # plt.close()
#     plt.show()
#     # plt.savefig('../plots/test/tau_fits/dcl2_tau_fits_{}.png'.format(kind), bbox_inches='tight', dpi=150)
#     # plt.savefig('../plots/sim_k_1_11/tau_fits_{}.png'.format(kind), bbox_inches='tight', dpi=150)
#     #
#     # plt.close()

# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, err1_list, err2_list, chi_list, t_err = [], [], [], [], [], [], [], []
# # path = 'cosmo_sim_1d/sim_k_1_11/run1/'
# # path = 'cosmo_sim_1d/sim_k_1_11/run1/'
#
# A = [-0.05, 1, -0.5, 11]
# Nfiles = 51
# Lambda = 3 * (2*np.pi)
# kind = 'sharp'
# kind_txt = 'sharp cutoff'
#
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'
#
# n_runs = 8
# n_use = n_runs-2
# mode = 1
# # path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
# # path = 'cosmo_sim_1d/final_sim_k_1_11/run1/'
# path = 'cosmo_sim_1d/sim_k_1_11/run1/'
#
#
# flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')
# folder_name = '' #'/new_data_{}/L{}'.format('sharp', int(Lambda/(2*np.pi)))
# # path = 'cosmo_sim_1d/another_sim_k_1_11/run1/'
#
# f1 = 'curve_fit'
# # f1 = ''
# nbins_x = 15
# nbins_y = 15
# npars = 6
#
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# # plt.rcParams.update({'font.weight': 'bold'})
# fig, ax = plt.subplots(figsize=(10, 6))
# for j in range(Nfiles):
#     sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, folder_name, fitting_method=f1, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars)
#     a_list.append(sol[0])
#     ctot2_list.append(sol[2])
#     ctot2_2_list.append(sol[3])
#     ctot2_3_list.append(sol[4])
#
#     err1_list.append(sol[6])
#     err2_list.append(sol[7])
#     chi_list.append(sol[10])
#     t_err.append(sol[-7])
#
#     aic = AIC(npars, sol[-1])
#     print('a = ', sol[0], j, aic)
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
#
# chi_list = np.array(chi_list)
# t_err = np.array(t_err)
# # ind = np.where(np.max(chi_list) == chi_list)[0][0]
# # chi_list[ind] = chi_list[ind-1]
#
#
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
# ax.set_xlabel(r'$a$', fontsize=20)
# # ax.fill_between(a_list, ctot2_list-t_err, ctot2_list+t_err, color='darkslategray', alpha=0.55, zorder=2)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)
#
#
# # ctot2_line, = ax.plot(a_list, ctot2_list, c='k', lw=1.5, zorder=4, marker='o')
# ctot2_2_line, = ax.plot(a_list, ctot2_2_list, c='brown', lw=1.5, marker='*', zorder=2)
# ctot2_3_line, = ax.plot(a_list, ctot2_3_list, c='orange', lw=1.5, marker='v', zorder=3)
# # plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
#
# # plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W'], fontsize=14, loc=1, framealpha=1)
#
# # plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'], fontsize=14)
#
# # obj = ax.scatter(a_list, ctot2_list, c=chi_list, s=40, cmap='rainbow', norm=colors.LogNorm(vmin=chi_list.min(), vmax=chi_list.max()), zorder=4)#, label=r'$c^{2}_{\mathrm{tot}}$')
# # cbar = fig.colorbar(obj, ax=ax)#, title=r'$\mathrm{log}(\chi^{2}/\mathrm{d.o.f.}$')
# # cbar.ax.set_ylabel(r'$\chi^{2}/\mathrm{d.o.f.}$', fontsize=18)
#
# # ax.errorbar(a_list, ctot2_list, yerr=t_err, c='k', linestyle="None")  #, marker='o', markerfacecolor='b', markeredgecolor='b', markersize=5)#
# # ax.set_ylim(0,1.5)
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
#
# # ax.legend(fontsize=11)
# # plt.show()
# # # plt.savefig('../plots/test/ctot2/err/ctot2_{}_all.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.savefig('../plots/test/ctot2/err/ctot2_{}_all.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# # plt.savefig('../plots/test/ctot2/err/ctot2_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# # plt.savefig('../plots/sim_k_1_11/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# #
# # plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# # # plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.close()
# plt.show()

# path = 'cosmo_sim_1d/sim_k_1_15/run2/'
# A = [-0.05, 1, -0.5, 11]
# mode = 1
# Lambda = (2*np.pi) * 5
# n_runs = 8
# n_use = 10
# kinds = ['sharp', 'gaussian']
# kinds_txt = ['sharp cutoff', 'Gaussian smoothing']
# Nfiles = 51
#
# for j in range(2):
#     kind = kinds[j]
#     kind_txt = kinds_txt[j]
#     for n in range(Nfiles):
#         # sol = param_calc_ens(n, Lambda, path, A, mode, kind, n_runs, n_use)
#         # x = sol[1]
#         # a = sol[0]
#         # tau = sol[12]
#
#         moments_filename = 'output_hierarchy_{0:04d}.txt'.format(n)
#         moments_file = np.genfromtxt(path + moments_filename)
#         a = moments_file[:,-1][0]
#         x = moments_file[:,0]
#         tau = read_sim_data(path, Lambda, kind, j)[1]
#
#         print('a = ', a)
#         fig, ax = plt.subplots()
#         ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
#         ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
#
#         ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(a, int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
#         ax.plot(x, tau, c='b', lw=1.5)#, label=r'$\left<[\tau]_{\Lambda}\right>$')
#         ax.minorticks_on()
#         ax.tick_params(axis='both', which='both', direction='in')
#         ax.yaxis.set_ticks_position('both')
#
#         # plt.legend(fontsize=12)
#         fig.align_labels()
#         plt.subplots_adjust(hspace=0, wspace=0)
#         plt.savefig('../plots/sim_k_1_15/tau_run2/{}/tau_{}.png'.format(kind, n), bbox_inches='tight', dpi=150)
#         plt.close()
