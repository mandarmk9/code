#!/usr/bin/env python3

#import libraries
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from functions import read_sim_data, AIC, sub_find, binning, spectral_calc, param_calc_ens
import os
import pickle
import pandas
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



A = [-0.05, 1, -0.5, 11]
Nfiles = 50
Lambda = 3 * (2*np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'

# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

path, n_runs = 'cosmo_sim_1d/sim_k_1_11/run1/', 8
# path, n_runs = 'cosmo_sim_1d/final_sim_k_1_11/run1/', 16
# path, n_runs = 'cosmo_sim_1d/new_sim_k_1_11/run1/', 24
n_use = n_runs-2
mode = 1

flags = np.loadtxt(fname=path+'/sc_flags.txt', delimiter='\n')

f1 = 'curve_fit'
# f1 = ''
nbins_x = 20
nbins_y = 20
npars = 3

fde_method = 'algorithm'
# fde_method = 'percentile'
per = 20

# for per in range(10, 101, 10):
a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err1_list, err2_list, chi_list, t_err, a_list4, err4_list = [], [], [], [], [], [], [], [], [], [], []
for j in range(Nfiles):
    sol = param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, fitting_method=f1, nbins_x=nbins_x, nbins_y=nbins_y, npars=npars, fde_method=fde_method, per=per)
    a_list.append(sol[0])
    ctot2_list.append(sol[2])
    ctot2_2_list.append(sol[3])
    ctot2_3_list.append(sol[4])
    ctot2_4_list.append(sol[-2])
    err4_list.append(sol[-1])
    print('err', sol[-1])
    err1_list.append(sol[6])
    err2_list.append(sol[7])
    chi_list.append(sol[10])
    t_err.append(sol[-8])
    print('a = ', sol[0], 'ctot2: ', sol[2], ',', sol[-2])

ctot2_4_list = np.array(ctot2_4_list)
err4_list = np.array(err4_list)
a_list = np.array(a_list)

df = pandas.DataFrame(data=[a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list])
file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'wb')
pickle.dump(df, file)
file.close()

# file = open("./{}/ctot2_plot_{}_L{}.p".format(path, kind, int(Lambda/(2*np.pi))), 'rb')
# read_file = pickle.load(file)
# a_list, ctot2_list, ctot2_2_list, ctot2_3_list, ctot2_4_list, err4_list = np.array(read_file)
# file.close()
#
#
# ctot2_4, a_list4, err4 = [], [], []
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams.update({"font.family": "serif"})
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_title(r'$\Lambda = {} \;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=18, y=1.01)
# ax.set_xlabel(r'$a$', fontsize=20)
# ax.set_ylabel('$c_{\mathrm{tot}}^{2}[\mathrm{km}^{2}s^{-2}]$', fontsize=20)
#
# # print(ctot2_4_list)
# for j in range(len(ctot2_list)):
#     # if j == 0:
#     #     cond = False
#     #     distance = 0
#     # else:
#     #     distance = np.abs(ctot2_4_list[j] - ctot2_4_list[j-1])
#     #     cond = distance < 1
#     cond = True
#     if cond:
#         print(j, a_list[j], ctot2_4_list[j], ctot2_3_list[j])#, distance)
#         ctot2_4.append(ctot2_4_list[j])
#         a_list4.append(a_list[j])
#         err4.append(err4_list[j])
#     if flags[j] == 1:
#         sc_line = ax.axvline(a_list[j], c='teal', lw=1, zorder=1)
#
#     else:
#         # sc_line = ax.axvline(0.5, c='teal', lw=1, zorder=1)
#         # print('boo')
#         pass
#
# ctot2_4_list = np.array(ctot2_4)
# a_list4 = np.array(a_list4)
# err4_list = np.array(err4)
#
# # ax.fill_between(a_list, ctot2_list-t_err, ctot2_list+t_err, color='darkslategray', alpha=0.55, zorder=2)
# ctot2_line, = ax.plot(a_list, ctot2_list, c='k', lw=1.5, zorder=4, marker='o') #from tau fit
# ctot2_2_line, = ax.plot(a_list, ctot2_2_list, c='cyan', lw=1.5, marker='*', zorder=2) #M&W
# ctot2_3_line, = ax.plot(a_list, ctot2_3_list, c='orange', lw=1.5, marker='v', zorder=3) #B+12
# ctot2_4_line, = ax.plot(a_list4, ctot2_4_list, c='xkcd:dried blood', lw=1.5, marker='+', zorder=1) #FDE
# ctot2_4_err = ax.fill_between(a_list4, ctot2_4_list-err4_list, ctot2_4_list+err4_list, color='darkslategray', alpha=0.55, zorder=2)
#
#
# # plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, (ctot2_4_line, ctot2_4_err)], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE'], fontsize=14, loc=3, framealpha=1)
# plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line, ctot2_4_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $[\tau]_{\Lambda}$', r'M\&W', r'$\mathrm{B^{+12}}$', r'FDE: {}\%, median'.format(per)], fontsize=14, loc=3, framealpha=1)
#
# # plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# # plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{B^{+12}}$'], fontsize=14, loc=1, framealpha=1)
# # plt.legend(handles=[sc_line, ctot2_2_line, ctot2_3_line], labels=[r'$a_\mathrm{sc}$', r'M\&W', r'$\mathrm{C^{+12}}$'], fontsize=14, loc=1, framealpha=1)
#
# # plt.legend(handles=[sc_line, ctot2_line, ctot2_2_line], labels=[r'$a_\mathrm{sc}$', r'from fit to $\langle[\tau]_{\Lambda}\rangle$', r'M\&W'], fontsize=14, loc=1, framealpha=1)
# # plt.legend(handles=[sc_line], labels=[r'$a_\mathrm{sc}$'], fontsize=14)
#
#
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
# ax.yaxis.set_ticks_position('both')
#
# # plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.pdf'.format(kind), bbox_inches='tight', dpi=300) #ctot2/err/lined/
# # plt.savefig('../plots/test/new_paper_plots/ctot2_ev_{}.png'.format(kind), bbox_inches='tight', dpi=150)
# # plt.savefig('../plots/test/new_paper_plots/fde_per_test_median/ctot2_ev_{}_{}.png'.format(kind, int(per)), bbox_inches='tight', dpi=150)
#
# # plt.savefig('../plots/test/new_paper_plots/fit_comps/curve_fit/{}_{}_ctot2_ev_{}.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_ctot2_ev_{}.png'.format(nbins_x, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.savefig('../plots/test/new_paper_plots/fit_comps/{}_{}_ctot2_ev_{}_median_w_binned_fit.png'.format(n_runs, data_cov, kind), bbox_inches='tight', dpi=150) #ctot2/err/lined/
# # plt.close()
# plt.show()
