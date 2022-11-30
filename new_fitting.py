#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from functions import read_sim_data
from scipy.interpolate import interp2d, griddata
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 10
a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
# print(a)
dv_l *= -np.sqrt(a) / 100
tau_l -= np.mean(tau_l)
# plt.plot(x, tau_l)

def dir_der_o1(X, tau_l, ind):
    """Calculates the first-order directional derivative of tau_l along the vector X."""
    x1 = np.array([X[0][ind], X[1][ind]])
    x2 = np.array([X[0][ind+1], X[1][ind+1]])
    v = (x2 - x1)
    D_v_tau = (tau_l[ind+1] - tau_l[ind]) / v[0] #/ v[0]
    return v, D_v_tau

def dir_der_o2(X, tau_l, ind):
    """Calculates the second-order directional derivative of tau_l along the vector X."""
    #calculate the first-order directional derivatives at two different points
    v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
    v2, D_v_tau2 = dir_der_o1(X, tau_l, ind+2)
    x1 = np.array([X[0][ind], X[1][ind]])
    x2 = np.array([X[0][ind+2], X[1][ind+2]])
    v = (x2 - x1)
    D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
    return v, D2_v_tau

def new_param_calc(dc_l, dv_l, tau_l, dist):
    ind = np.argmin(dc_l**2 + dv_l**2)
    X = np.array([dc_l, dv_l])
    params_list = []
    for j in range(-dist//2, dist//2 + 1):
        v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
        v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
        C_ = [tau_l[ind], dtau1, dtau1_o2/2]
        params_list.append(C_)


    params_list = np.array(params_list)
    C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
    C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
    C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))
    C_ = [C0_, C1_, C2_]
    return C_

dist = 20
C_ = new_param_calc(dc_l, dv_l, tau_l, dist)

# # tau_l -= np.mean(tau_l)
# # ind = np.argmin(dc_l**2 + dv_l**2)
# ind = np.argmin(tau_l**2)
# # print(ind, ind2)
# # print(tau_l[ind], tau_l[ind2])
#
# last = 10000
# # print(ind)
# tau_l_sp = tau_l[ind-last:ind+last]
# dc_l_sp = dc_l[ind-last:ind+last]
# dv_l_sp = dv_l[ind-last:ind+last]
#
# # tau_l_sp = tau_l[0::800]
# # dc_l_sp = dc_l[0::800]
# # dv_l_sp = dv_l[0::800]
#
# N = tau_l_sp.size
# taus_calc = []
# for j in range(N-3):
#     # print(j)
#     C_ = new_param_calc(dc_l_sp, dv_l_sp, tau_l_sp, dist, j)
#     print(C_)
#     tau_calc = C_[0] + C_[1]*dc_l[j] + C_[2]*dc_l[j]**2
#     taus_calc.append(tau_calc)
# # print(taus_calc[int((N-3)/2)])
# taus_calc = np.array(taus_calc)

# plt.rcParams.update({"text.usetex": True})
# fig, ax = plt.subplots()
# ax.scatter(tau_l_sp[:-3], taus_calc, c='b', s=2)
# ax.set_xlabel(r'$\tau_{l}$')
# ax.set_ylabel(r'$\Delta \tau_{l}$')
#
# plt.savefig('../plots/test/new_paper_plots/tau_diff.png', bbox_inches='tight', dpi=150)
# plt.close()


##plotting
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots()
###

# nvlines, nhlines = 25, 25 #60, 60
# min_dc, max_dc = dc_l.min(), dc_l.max()
# dc_bins = np.linspace(min_dc, max_dc, nvlines)
#
# min_dv, max_dv = dv_l.min(), dv_l.max()
# dv_bins = np.linspace(min_dv, max_dv, nhlines)

# nvlines, nhlines = 10, 10 #dc_bins.size, dv_bins.size
# bin_size = 2.5e-3
# dc_bins = np.arange(-nvlines*bin_size, (nvlines*bin_size), bin_size)
# print(dc_bins)
# bin_size = 2.5e-3
# dv_bins = np.arange(-nhlines*bin_size, (nhlines*bin_size), bin_size)
# nvlines, nhlines = dc_bins.size, dv_bins.size
# # print(dc_bins.size, dv_bins.size)
# # # print(nvlines, nhlines = )

# dc_bins, dv_bins = [], []
# nbins_dc, nbins_dv = 15, 15
# bin_size = 2.5e-3
# j = 1
# while j < nbins_dc+1:
#     idx_dc = np.where(np.logical_and(dc_l>=-bin_size*j, dc_l<=bin_size*j))[0]
#     dc_bins.append(idx_dc)
#     j +=1
#
# j = 1
# while j < nbins_dv+1:
#     idx_dv = np.where(np.logical_and(dv_l>=-bin_size*j, dv_l<=bin_size*j))[0]
#     dv_bins.append(idx_dv)
#     j += 1
#
# binned_dc = np.array([np.mean(dc_l[dc_bins[j]]) for j in range(len(dc_bins))])
# binned_dv = np.array([np.mean(dv_l[dv_bins[j]]) for j in range(len(dv_bins))])
#
# interp2d(binned_dc, binned_dv, tau_l)
# print(dc_bins[0], dv_bins[0])
# # print(binned_dc, binned_dv)
#
# # print(idx_dc.size, idx_dv.size)
# # print(np.mean(dc_l[idx_dc]))
# # print(np.mean(dv_l[idx_dv]))

# plt.plot(x, dv_l)
# plt.plot(x, dc_l)
#
# plt.show()

# print(dc_l[:100])
# print(dv_l[:100])

# def find_nearest(a, a0):
#     """Element in nd array 'a' closest to the scalar value 'a0'."""
#     idx = np.abs(a - a0).argmin()
#     return idx, a.flat[idx]
#
# bsx =0.3e-1#0.1
# bsy = 0.3e-1#0.1#1e-2
# print(np.abs(dc_l[1]-dc_l[0]))
# dels, thes, taus, co_x1, co_x2, co_y1, co_y2 = [], [], [], [], [], [], []
# nbins = 40
#
# zero_x = 0#find_nearest(dc_l, 0)[1]
# zero_y = 0#find_nearest(dv_l, 0)[1]
#
# minima = 7 #min(taus)
# maxima = 8 #max(taus)
# norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
# # print(zero_x, zero_y)
# # co_xl, co_xr = dc_l.min() + j*bsx, dc_l.min() + (j+1)*(bsx)
# # co_yl, co_yr = dv_l.min() + j*bsy, dv_l.min() + (j+1)*(bsy)
#
# coor_x = [(-((bsx/2) + ((nbins/2)-j+1)*bsx + zero_x), -((bsx/2) + ((nbins/2)-j)*bsx + zero_x)) for j in range(nbins)]
# coor_y = [(-((bsy/2) + ((nbins/2)-j+1)*bsy + zero_y), -((bsy/2) + ((nbins/2)-j)*bsx + zero_y)) for j in range(nbins)]
#
#
# for j in range(nbins):
#     for l in range(nbins):
#         # map = 8
#         # verts = [[coor_x[j][0], coor_y[l][1]], [coor_x[j][0], coor_y[l][0]], [coor_x[j][1], coor_y[l][0]], [coor_x[j][1], coor_y[l][1]]]
#         # poly = PolyCollection([verts], facecolors='r', edgecolors='k', linewidth=1)
#         # ax.add_collection(poly)
#
#         try:
#             dc_co = np.logical_and(dc_l>=coor_x[j][0], dc_l<=coor_x[j][1])
#             dv_co = np.logical_and(dv_l>=coor_y[l][0], dv_l<=coor_y[l][1])
#             idx = np.where(np.logical_and(dc_co, dv_co))[0]
#             idx_dc = np.where(dc_co)[0]
#             idx_dv = np.where(dv_co)[0]
#
#             bin0_dc = np.mean(dc_l[idx_dc])
#             bin0_dv = np.mean(dv_l[idx_dv])
#             bin0_tau = np.mean(tau_l[idx])
#             dels.append(bin0_dc)
#             thes.append(bin0_dv)
#             taus.append(bin0_tau)
#             map = bin0_tau
#             verts = [[coor_x[j][0], coor_y[l][1]], [coor_x[j][0], coor_y[l][0]], [coor_x[j][1], coor_y[l][0]], [coor_x[j][1], coor_y[l][1]]]
#             color = mapper.to_rgba(map)
#             poly = PolyCollection([verts], facecolors=color, edgecolors='k', linewidth=1)
#             ax.add_collection(poly)
#         except Exception as e: print(e)
# try:
#     print(co_xl, co_xr)
#     print(co_yl, co_yr)
#
#     dc_co = np.logical_and(dc_l>=co_xl, dc_l<=co_xr)
#     dv_co = np.logical_and(dv_l>=co_yl, dv_l<=co_yr)
#     idx = np.where(np.logical_and(dc_co, dv_co))[0]
#     idx_dc = np.where(dc_co)[0]
#     idx_dv = np.where(dv_co)[0]
#
#     bin0_dc = np.mean(dc_l[idx_dc])
#     bin0_dv = np.mean(dv_l[idx_dv])
#     bin0_tau = np.mean(tau_l[idx])
#     dels.append(bin0_dc)
#     thes.append(bin0_dv)
#     taus.append(bin0_tau)
#
#     co_x1.append(co_xl)
#     co_y1.append(co_yl)
#     co_x2.append(co_xr)
#     co_y2.append(co_yr)
#
#
#     # ax.scatter(np.mean(dc_l[dc_co]), np.mean(dv_l[dv_co]), c='r')
# except Exception as e: print(e)
#     # pass

# # print(taus)
# minima = min(taus)
# maxima = max(taus)
# norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='viridis')

# for j in range(len(co_x1)):
#     # for l in range(len(co_y1)):
#     map = taus[j]
#     verts = [[co_x1[j], co_y2[j]], [co_x1[j], co_y1[j]], [co_x2[j], co_y1[j]], [co_x2[j], co_y2[j]]]
#     color = mapper.to_rgba(map)
#     poly = PolyCollection([verts], facecolors = color, edgecolors='k', linewidth=1)
#     ax.add_collection(poly)


# points = (dels, thes)
# x_g = np.arange(dc_l.min(), dc_l.max(), 1e-3)
# y_g = np.arange(dv_l.min(), dv_l.max(), 1e-3)
#
# grid_x, grid_y = np.meshgrid(x_g, y_g)
# tau = griddata(points, taus, (grid_x, grid_y))
#
# print(tau)
# def find_nearest(a, a0):
#     """Element in nd array 'a' closest to the scalar value 'a0'."""
#     idx = np.abs(a - a0).argmin()
#     return idx, a.flat[idx]
#
# ind_x, val_x = find_nearest(x_g, 0)
# ind_y, val_y = find_nearest(y_g, 0)
#
# print(ind_x, val_x)
# print(ind_y, val_y)

# print(tau[ind_x, ind_y])

# ind_dc, val_dc = find_nearest(dels, 0)
# ind_dv, val_dv = find_nearest(thes, 0)

# print(taus[ind_dv])
# print(np.mean(taus))
# # print(dels, thes, taus)

# # ##plotting
ax.set_xlabel(r'$\delta_{l}$', fontsize=18)
ax.set_ylabel(r'$\theta_{l}$', fontsize=18)
ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(np.round(a, 3), int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# ax.set_title(r'$a = {}, \Lambda = {}\;[2\pi h\;\mathrm{{Mpc}}^{{-1}}]$ ({})'.format(int(Lambda/(2*np.pi)), kind_txt), fontsize=16)
# minima = min(taus)
# maxima = max(taus)
# norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
# for j in range(len(dels)):
#     # ax.axvline(dc_bins[j], c='k', lw=0.5)
#     ax.axvline(co_x1[j], c='k', lw=0.5)
#     ax.axvline(co_x2[j], c='k', lw=0.5)
#
#     # map = taus
#     # color = mapper.to_rgba(map)
#     ax.axhline(co_y1[j], c='k', lw=0.5)
#     ax.axhline(co_y2[j], c='k', lw=0.5)
#
#     # for l in range(len(dels)):
#     #     verts = [[co_x1[j], co_y2[l]], [co_x1[j], co_y1[l]], [co_x2[j], co_y1[l]], [co_x2[j], co_y2[l]]]
#     #     poly = PolyCollection([verts], facecolors = color, edgecolors='k', linewidth=1)
#     #     ax.add_collection(poly)


ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)
# cbar = plt.colorbar(mapper)
# cbar.ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>$', fontsize=16)
# cbar.ax.set_ylabel(r'fit to $\left<[\tau]_{\Lambda}\right>$', fontsize=16)
# cbar.ax.set_ylabel(r'residual', fontsize=16)
# ax.plot(dels, thes, c='seagreen', lw=2, marker='o')
cm = plt.cm.get_cmap('RdYlBu')

ax.axvline(0, lw=0.5, c='k', ls='dashed')
ax.axhline(0, lw=0.5, c='k', ls='dashed')
# ax.scatter(dc_l, dv_l, c='b', s=10)
from matplotlib import ticker
# obj = ax.scatter(dc_l, dv_l, c=tau_l, s=20, cmap='rainbow', norm=colors.LogNorm(vmin=tau_l.min(), vmax=tau_l.max()))
# taus_calc -= np.mean(taus_calc)

fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
del_tau = tau_l-0.06 #fit - tau_l
print(del_tau.min(), del_tau.max())
obj = ax.scatter(dc_l, dv_l, c=del_tau, s=20, cmap='rainbow', rasterized=True)#, norm=colors.Normalize(vmin=del_tau.min(), vmax=del_tau.max()))

# ax.scatter(dc_l[ind-10:ind+10], dv_l[ind-10:ind+10])#, c=tau_l, s=20, cmap='rainbow', norm=colors.LogNorm(vmin=tau_l.min(), vmax=tau_l.max()))

# j = 5
# ind = np.argmin(tau_l**2)
# ax.scatter(dc_l[ind], dv_l[ind], c='k', s=20)
# ax.scatter(dc_l[ind+j], dv_l[ind+j], c='k', s=20)



cbar = fig.colorbar(obj, ax=ax)
cbar.ax.set_ylabel(r'$[\tau]_{\Lambda}$', fontsize=18)
# cbar.ax.set_ylabel(r'$\Delta[\tau]_{\Lambda}\; [\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=18)
cbar.ax.tick_params(labelsize=12.5)

tick_locator = ticker.MaxNLocator(nbins=10)
cbar.locator = tick_locator
cbar.update_ticks()

# ind = np.argmin(dc_l**2 + dv_l**2)
# npoints = 20
# for gap in range(npoints):
#     gap *= dc_l.size//npoints
#     ax.scatter(dc_l[ind-gap], dv_l[ind-gap], c='k', s=20)
#     # ax.scatter(dc_l[ind+gap], dv_l[ind+gap], c='k', s=20)
# dc_0, dv_0 = dc_l[np.argmax(tau_l)-2000], dv_l[np.argmax(tau_l)-2000]#
# ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)

# dtau = np.abs(tau_l-0.4)
# ind = np.argmin(dtau)
# ax.scatter(dc_l[ind], dv_l[ind], c='k', s=20)

delta = 0.01
sub_dc = np.arange(dc_l.min(), dc_l.max(), delta)
sub_dv = np.arange(dv_l.min(), dv_l.max(), delta)
print('boo1')

n_sub = np.minimum(sub_dc.size, sub_dv.size)
print('boo')
dc_subbed, dv_subbed = [], []
for j in range(n_sub):
    print(j)
    dc_0, dv_0 = sub_dc[j], sub_dv[j]
    ind = np.argmin((dc_l - dc_0)**2 + (dv_l - dv_0)**2)
    dc_subbed.append(dc_l[ind])
    dv_subbed.append(dv_l[ind])

ax.scatter(dc_subbed, dv_subbed, c='k', s=20)

# ax.scatter(dc_0, dv_0, c='b', s=20)

# for j in range(n_sub):
#     dc_0, dv_0 = np.repeat(sub[j], 2)
#     ind = np.argmin((dc_l - dc_0)**2 + (dv_l - dv_0)**2)
#     print(dc_l[ind], dv_l[ind])
#     ax.scatter(dc_l[ind], dv_l[ind], c='k', s=20)


# plt.savefig('../plots/test/new_paper_plots/tau_diff.png', bbox_inches='tight', dpi=150)
# plt.savefig('../plots/test/new_paper_plots/tau_diff_pre_sc_gauss.pdf', bbox_inches='tight', dpi=300)
# plt.savefig('../plots/test/new_paper_plots/tau_diff_post_sc_gauss.pdf', bbox_inches='tight', dpi=300)
#
#
# plt.close()
plt.show()

# ax.set_xlabel(r'$\delta_{l}$', fontsize=18)
# ax.set_ylabel(r'$\theta_{l}$', fontsize=18)
# ax.set_title('a = {}'.format(np.round(a, 3)))
# ###
#
# meds, counts, inds, yerr = [], [], [], []
# taus, dels, thes, delsq, thesq, delthe = [], [], [], [], [], []
# mns = []
# count = 0
# for i in range(nhlines-1):
#     for j in range(nvlines-1):
#         count += 1
#         start_coos = (i,j)
#         m,n = start_coos
#         mns.append([m,n])
#         ##plotting
#         ax.text((dc_bins[i]+dc_bins[i+1])/2, (dv_bins[j]+dv_bins[j+1])/2, s=count, fontsize=10)
#         # ax.plot(dc_bins[m], dv_bins[n], marker='o', c='r')
#         # ax.plot(dc_bins[m], dv_bins[n+1], marker='o', c='r')
#         # ax.plot(dc_bins[m+1], dv_bins[n], marker='o', c='r')
#         # ax.plot(dc_bins[m+1], dv_bins[n+1], marker='o', c='r')
#         ###
#
#
#         indices = []
#         for l in range(x.size):
#             if dc_bins[m] <= dc_l[l] <= dc_bins[m+1] and dv_bins[n] <= dv_l[l] <= dv_bins[n+1]:
#                 indices.append(l)
#         indices = np.array(indices)
#         try:
#             # # print(indices.size)
#             # left = indices[0]
#             # right = indices.size // 2
#             # inds_ = list(np.arange(left, left+right+1, 1))
#             inds_ = np.sort(indices)
#             tau_mean = sum(tau_l[inds_]) / len(inds_)
#             delta_mean = sum(dc_l[inds_]) / len(inds_)
#             theta_mean = sum(dv_l[inds_]) / len(inds_)
#             del_sq_mean = sum((dc_l**2)[inds_]) / len(inds_)
#             the_sq_mean = sum((dv_l**2)[inds_]) / len(inds_)
#             del_the_mean = sum((dc_l*dv_l)[inds_]) / len(inds_)
#
#
#             taus.append(tau_mean)
#             dels.append(delta_mean)
#             thes.append(theta_mean)
#             delsq.append(del_sq_mean)
#             thesq.append(the_sq_mean)
#             delthe.append(del_the_mean)
#
#             yerr.append(np.sqrt(sum((tau_l[inds_] - tau_mean)**2) / (len(inds_)-1)))
#
#             medians = np.median(inds_)
#             meds.append(medians)
#             counts.append(count)
#         except:
#             left = None
#
# # meds, counts = (list(t) for t in zip(*sorted(zip(meds, counts))))
# # meds, taus = (list(t) for t in zip(*sorted(zip(meds, taus))))
# # meds, dels = (list(t) for t in zip(*sorted(zip(meds, dels))))
# # meds, thes = (list(t) for t in zip(*sorted(zip(meds, thes))))
# # meds, delsq = (list(t) for t in zip(*sorted(zip(meds, delsq))))
# # print(counts)
# #
# # plt.savefig('../plots/test/new_paper_plots/tests.png', bbox_inches='tight', dpi=150)
#
#
# # def fitting_function(X, a0, a1, a2):
# #     x1, x2 = X
# #     return a0 + a1*x1 + a2*x2
# #
# # guesses = 1, 1, 1
# # C, cov = curve_fit(fitting_function, (dels, thes), taus, guesses, sigma=yerr, method='lm', absolute_sigma=True)
# # errs = np.sqrt(np.diag(cov))
# # C0, C1, C2 = C
# # err0, err1, err2 = errs
# # best_fit = fitting_function((np.array(dels), np.array(thes)), C0, C1, C2)
#
# def fitting_function(X, a0, a1, a2, a3, a4, a5):
#     x1, x2, x3, x4, x5 = X
#     return a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5
#
# guesses = 1, 1, 1, 1, 1, 1
# C, cov = curve_fit(fitting_function, (dels, thes, delsq, thesq, delthe), taus, guesses, sigma=yerr, method='lm', absolute_sigma=True)
# C0, C1, C2, C3, C4, C5 = C
# best_fit = fitting_function((np.array(dels), np.array(thes), np.array(delsq), np.array(thesq), np.array(delthe)), C0, C1, C2, C3, C4, C5)
#
# # print(C)
# #
# # best_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), C0, C1, C2, C3, C4, C5)
# # C0_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), C0, 0, 0, 0, 0, 0)
# # C1_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), 0, C1, 0, 0, 0, 0)
# # C2_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), 0, 0, C2, 0, 0, 0)
# # C3_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), 0, 0, 0, C3, 0, 0)
# # C4_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), 0, 0, 0, 0, C4, 0)
# # C5_fit = fitting_function((dc_l, dv_l, dc_l**2, dv_l**2, dc_l*dv_l), 0, 0, 0, 0, 0, C5)
# #
# # plots = [best_fit, C0_fit, C1_fit, C2_fit, C3_fit, C4_fit, C5_fit]
# # labels = ['best_fit', 'C0_fit', 'C1_fit', 'C2_fit', 'C3_fit', 'C4_fit', 'C5_fit']
# #
# # linestyles = ['solid', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 10, 1, 10)), 'dashdot', 'dashed', 'dotted']
# # colors = ['brown', 'darkcyan', 'dimgray', 'violet', 'orange', 'cyan', 'b', 'r', 'k']
# #
# # first_order = C1_fit + C2_fit
# # second_order = C3_fit + C4_fit + C5_fit
# #
# #
# # fig, ax = plt.subplots()
# # ax.set_title('a = {}'.format(np.round(a, 3)))
# # # for j in range(7):
# # #     ax.plot(x, plots[j], c=colors[j], ls='solid', label=labels[j])
# # # # plt.plot(x, best_fit, c='k', ls='dashed')
# #
# # ax.plot(x, C0_fit, c=colors[0], ls='solid', label=labels[1])
# # ax.plot(x, first_order, c='k', ls='solid', label='first_order')
# # ax.plot(x, second_order, c='b', ls='solid', label='second_order')
# #
# #
# # # sum_fit = sum([plots[j] for j in range(1, len(plots))])
# # # plt.plot(x, sum_fit, c='b')
# # #
# # # plt.plot(x, tau_l, c='k', ls='dashed')
# # # plt.legend()
# # # plt.show()
#
#
# # def AIC(k, chisq, n=1):
# #     """Calculates the Akaike Information from the number of parameters k
# #     and the chi-squared of the fit chisq. If n > 1, it modifies the formula to
# #     account for a small sample size (specified by n).
# #     """
# #     aic = 2*k + chisq
# #     if n > 1:
# #         aic += ((2*(k**2) + 2*k) / (n - k - 1))
# #     else:
# #         pass
# #     return aic
# #
# # def BIC(k, n, chisq):
# #     """Calculates the Bayesian Information from the number of parameters k,
# #     the sample size n, and the chi-squared of the fit chisq.
# #     """
# #     bic = k*np.log(n) + chisq
# #     if n > 1:
# #         aic += ((2*(k**2) + 2*k) / (n - k - 1))
# #     else:
# #         pass
# #     return aic
# #
#
# new_dict = dict(zip(counts, best_fit))
#
# minima = min(x)#min(taus)
# maxima = max(x)#max(taus)
# norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
# cbar = plt.colorbar(mapper)
# cbar.ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>$', fontsize=16)
# cbar.ax.set_ylabel(r'fit to $\left<[\tau]_{\Lambda}\right>$', fontsize=16)
# # cbar.ax.set_ylabel(r'residual', fontsize=16)
#
# count = 0
# new_count = 0
# # mns = [[8,0]]
# for m, n in mns:
#     count += 1
#     indices = []
#     for l in range(x.size):
#         if dc_bins[m] <= dc_l[l] <= dc_bins[m+1] and dv_bins[n] <= dv_l[l] <= dv_bins[n+1]:
#             indices.append(l)
#
#     indices = np.array(indices)
#     if indices.size != 0:
#         # print(x[indices].mean())
#         print(indices)
#         left = indices[0]
#         right = indices.size #// 2
#         print(right)
#         inds_ = indices #list(np.arange(left, left+right+1, 1))
#         tau_mean = sum(tau_l[inds_]) / len(inds_)
#         fit = new_dict[count] #best_fit[new_count]
#         resid = fit - tau_mean
#         verts = [[dc_bins[m+1], dv_bins[n]], [dc_bins[m], dv_bins[n]], [dc_bins[m], dv_bins[n+1]], [dc_bins[m+1], dv_bins[n+1]]]
#
#         mapping = [tau_mean, fit, resid, np.mean(x[inds_])]
#         map = mapping[-1]
#
#         color = mapper.to_rgba(map)
#
#
#         new_count += 1
#
#         poly = PolyCollection([verts], facecolors = color, edgecolors='k', linewidth=1)
#         # ax.text((3*dc_bins[m]+dc_bins[m+1])/4, (dv_bins[n]+dv_bins[n+1])/2, s=(np.round(map, 5)), color='white', fontsize=6)
#         ax.add_collection(poly)
#     else:
#         pass
#
# # ax.scatter(dc_l, dv_l, c='seagreen', s=0.5)
# # ax.scatter(dels, thes, c='r', s=20)
#
# plt.show()
# # plt.savefig('../plots/test/new_paper_plots/grid_plots/tau_all.png', bbox_inches='tight', dpi=150)
# # # plt.savefig('../plots/test/new_paper_plots/grid_plots/fit_all.png', bbox_inches='tight', dpi=150)
# # #
# # plt.close()
#
#
#
# # print(C)
# #
# # fit = fitting_function((dc_l, dv_l), C0, C1, C2)
# # plt.rcParams.update({"text.usetex": True})
# #
# # fig, ax = plt.subplots()
# # ax.set_title('a = {}'.format(np.round(a, 3)))
# # ax.plot(x, tau_l, c='b', label=r'measured $\left<[\tau]_{\Lambda}\right>$')
# # ax.plot(x, fit, c='k', ls='dashed', label=r'fit to $\left<[\tau]_{\Lambda}\right>$')
# # ax.set_ylabel(r'$\left<[\tau]_{\Lambda}\right>\;[\mathrm{M}_{10}h^{2}\frac{\mathrm{km}^{2}}{\mathrm{Mpc}^{3}s^{2}}]$', fontsize=16)
# # ax.set_xlabel(r'$x\;[h^{-1}\;\mathrm{Mpc}]$', fontsize=18)
# # ax.minorticks_on()
# # ax.tick_params(axis='both', which='both', direction='in', labelsize=13.5)
# # plt.legend(fontsize=12)
# # plt.show()
#
# # plt.savefig('../plots/test/new_paper_plots/after_sc.png', bbox_inches='tight', dpi=150)
# # plt.close()
#
#
# # # Creating figure
# # fig = plt.figure(figsize =(14, 9))
# # ax = plt.axes(projection ='3d')
# #
# # # Creating color map
# # cm = plt.get_cmap('viridis')
# # # Creating plot
# # ax.view_init(0, 0)
# # print(zz)
# # ax.plot_surface(xx, yy, zz, cmap=cm)
# #
# # # plt.show()
# # plt.savefig('../plots/test/new_paper_plots/new_fitting.png', dpi=300)
# # plt.close()
