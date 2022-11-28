#from cosmo_sim_moments.py on 15.02.22; 16:37
n = 0
dx_g = L / x_nbody.size
x_g = np.arange(-L/2, L/2, dx_g)
M0_nbody = kde_gaussian_moments(n, x_g, x_nbody, v_nbody, sm, L)
M0_zel = kde_gaussian_moments(n, x_g, x_zel, v_zel, sm, L)

def is_between(x, x1, x2):
   """returns a subset of x that lies between x1 and x2"""
   indices = []
   values = []
   for j in range(x.size):
      if x1 <= x[j] <= x2:
         indices.append(j)
         values.append(x[j])
   values = np.array(values)
   indices = np.array(indices)
   return indices, values

x_grid = np.arange(-0.5, 0.5, 1e-3)
print(v_nbody.size)
v_mean = np.zeros(x_grid.size)
for j in range(x_grid.size - 1):
   s = is_between(x_nbody, x_grid[j], x_grid[j+1])
   vels = v_nbody[s[0]]
   v_mean[j] = sum(vels) / len(vels)

v_sch = MH_1 / MH_0
C2_nbody = (sum(v_nbody**2) / v_nbody.size) - v_mean**2
# rho = kde_gaussian_moments(0, x_grid, x_nbody, v_nbody, sm, L)
# rho_u = kde_gaussian_moments(1, x_grid, x_nbody, v_nbody, sm, L)
# u = rho_u #/ rho

C2_sch = (MH_2 - (MH_1**2 / smoothing(MW_0, W_k_an)))

#from cosmo_sim_an.py on Mar 7, 2022; 16:21
k_grid = np.fft.fftfreq(x_grid.size, dx_grid) * 2.0 * np.pi

dk_nbody = np.zeros(x_grid.size, dtype=complex)
for j in range(x_nbody.size):
   print(j)
   dk_nbody += np.exp(1j * k_grid * x_nbody[j])
M0 = np.real(np.fft.ifft(dk_nbody))

#from cosmo_sim_an.py on Mar 7, 2022; 18:23
def W_CIC(x_nbody, x_grid):
   W = np.zeros(x_grid.size)
   for x in x_nbody:
      dx_grid = x_grid[1] - x_grid[0]
      diff = np.abs(x_grid - x)
      index = np.where(diff == np.min(diff))[0]
      W[index] += 1 - diff[index] / dx_grid
   return W


#from adaptive_test.py on 17.02.22; 11:22
# p_max = 1
# dv = 2 * p_max / (2 * Nx)
# p = np.arange(-p_max, p_max, dv) #np.sort(k * h)
#
# # print(sigma_x, sigma_p)
# # print(dx, dv)
#
# from functions import husimi
# [X, P] = np.meshgrid(x, p)
# F = husimi(psi, X, P, sigma_x, h, L)
# mom = MW_1 / MW_0
# vel = mom / (m * a0)
#
# v_pec = MH_1 / MH_0 / (m * a0)
# #
# # fig, ax = plt.subplots()
# # ax.set_xlabel(r'x$\,$[$h^{-1}$ Mpc]', fontsize=20)
# # ax.set_ylabel(r'$v\,$[km s$^{-1}$]', fontsize=20)
# # title = ax.text(0.05, 0.9, 'a = {}'.format(str(np.round(a0, 3))),  bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
# # plot2d_2 = ax.pcolormesh(x, p, F, shading='auto', cmap='inferno')
# #
# # ax.grid(linewidth=0.15, color='gray', linestyle='dashed')
# # c = fig.colorbar(plot2d_2, fraction=0.15)
# # c.set_label(r'$f_{H}$', fontsize=20)
# # ax.plot(x, v_zel * (m * a0), c='k', label='v_zel')
# # ax.plot(x, p_pec, c='r', ls='dashed', label='v_pec')
# #
# # # ax.set_ylim(-20, 20)
# # ax.legend(loc='upper right')
# # legend = ax.legend(frameon = 1, loc='upper right', fontsize=12)
# # frame = legend.get_frame()
# # plt.tight_layout()
# # frame.set_edgecolor('white')
# # frame.set_facecolor('black')
# # for text in legend.get_texts():
# #     plt.setp(text, color = 'w')
#
# rho_bar = np.mean(np.sum(F, axis=0))
# plt.plot(x, np.sum(F, axis=0))
# plt.savefig('/vol/aibn31/data1/mandar/plots/f_dp.png')
# plt.close()
#
# F /= rho_bar
# print(np.mean(np.sum(F, axis=0)))
# print(np.sum(np.sum(F, axis=0)))
#
# #F is now the correctly normalised distribution
# M0 = np.sum(F, axis=0) #* (m * (a0**3))
# M1 = np.sum(F * P, axis=0) #this produces a 'velocity density' as opposed to a momentum density
# M2 = np.sum(F * (P ** 2), axis=0) #this is the second moment, i.e., the Schrodinger stress
#
# # M0 /= np.mean(M0)
# # M1 /= np.mean(M1)
# # M2 /= np.mean(M0)
# # #
# # v_f = M1 / M0 / (m * a0)
# # # # plt.plot(x, nd, c='k')
# # # plt.plot(x, v_f, c='k')
# # # plt.plot(x, v_pec, c='brown', ls='dashdot')
# # # plt.plot(x, v_zel, c='b', ls='dashed')
# # # plt.plot(x, vel, c='r', ls='dotted')
# # # plt.scatter(k, W_k_an, c='b', s=50)
# # # plt.scatter(k, W_k_an_1, c='k', s=20)
# #
# # # C2 = M2 - (M1**2 / M0)
# # # CH_2 = MH_2 - (MH_1**2 / MH_0)
# # #
# # # plt.plot(x, CH_2, c='k')
# # # plt.plot(x, C2, c='brown', ls='dashdot')
# # #
# # # plt.savefig('/vol/aibn31/data1/mandar/plots/ps.png')
# # # plt.close()
# #
# # # C1 = M1 / M0 #* (a0**4)
# # # C2 = (M2 - (M1**2 / M0))
# # # CH_1 = MH_1 / MH_0
# # # CH_2 = (MH_2 - (MH_1**2 / MH_0)) * (np.mean(MH_0))
# #
# # #
###From EFT_nbody_solver.py on March 28, 2022; 11:53
###code for calculating spatial correlations; it is too slow (spectral method is faster and agrees)
def correlation(f1, f2, x):
   P_12 = np.zeros(x.size)
   sep = np.zeros(x.size)
   for j in range(0, x.size):
      print(j)
      P_12[j] = sum(np.roll(f1, j) * f2) / x.size
      sep[j] = x[j] - x[0]
   return sep, P_12

sep, P_ad = correlation(A_l, dc_l, x)

# P_dt = correlation(dc_l, Theta_l, x)
# P_at = correlation(A_l, Theta_l, x)
# P_dd = correlation(dc_l, dc_l, x)
# P_tt = correlation(Theta_l, Theta_l, x)
#
# den = ((spectral_calc(P_dt, k, 1.0, o=2, d=1)**2) - (spectral_calc(P_dd, k, 1.0, o=2, d=1) * (spectral_calc(P_tt, k, 1.0, o=2, d=1))))
# d2_P_dt = spectral_calc(P_dt, k, 1.0, o=2, d=0)
# cs2 = a**2 * ((P_at * d2_P_dt) - P_ad * spectral_calc(P_tt, k, 1.0, o=2, d=0)) / den
# cv2 = a**2 * ((P_ad * d2_P_dt) - P_at * spectral_calc(P_dd, k, 1.0, o=2, d=0)) / den

###From EFT_nbody_solver.py on March 29, 2022; 19:33
# A_l = spectral_calc(tau_l, k, 1.0, o=2, d=0) / (rho_b * a**2)
# Theta_l = -dv_l / (H0 / np.sqrt(a))

# fig, ax = plt.subplots()
# ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
# ax.scatter(k, cs2, c='r', s=20, label=r'$P_{\Theta\Theta}$')
# ax.set_xlim(0, 15)
# ax.set_xlabel('k', fontsize=14)
# ax.set_ylabel(r'$P(k)$', fontsize=14)
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
# plt.savefig('../plots/EFT_nbody/s.png', bbox_inches='tight', dpi=120)
# plt.close()


# print(cs2)
# num1 = Power(A_l, Theta_l) * spectral_calc(Power(dc_l, Theta_l), k, 1.0, o=2, d=0)
# num2 = Power(A_l, dc_l) * spectral_calc(Power(Theta_l, Theta_l), k, 1.0, o=2, d=0)
# den1 = (spectral_calc(Power(dc_l, Theta_l), k, 1.0, o=2, d=0))**2
# den2 = spectral_calc(Power(dc_l, dc_l), k, 1.0, o=2, d=0) * spectral_calc(Power(Theta_l, Theta_l), k, 1.0, o=2, d=0)
# cs2_k = (a**2 * (num1 - num2) / (den1 - den2))
# # cs2_k = (num1 - num2) / (den1 - den2) / (num1.size**2)
#
# num1 = Power(A_l, dc_l) * spectral_calc(Power(dc_l, Theta_l), k, 1.0, o=2, d=0)
# num2 = Power(A_l, Theta_l) * spectral_calc(Power(dc_l, dc_l), k, 1.0, o=2, d=0)
# # cv2_k = den1 - den2 #a**2 * (num1 - num2) / (den1 - den2) / num2.size)
# cv2_k = (a**2 * (num1 - num2) / (den1 - den2))
# # cs2 = np.real(np.fft.fft(cs2_k) / cs2_k.size)[1]
# # cv2 = np.real(np.fft.fft(cv2_k) / cv2_k.size)[1]
#
# cs2 = np.mean(cs2_k)#[0]
# cv2 = np.mean(cv2_k)#[0]
#
# ctot2 = cs2 + cv2 #a**2 * (num1 - num2) / (den1 - den2) #(cs2 + cv2)

# fig, ax = plt.subplots()
# ax.set_title(r'$a = {}, \Lambda = {}$'.format(a, Lambda))
# # ax.plot(x, fit, c='b', lw=2, label='fit')
# # ax.plot(x, tau_l, c='k', lw=2, ls='dashed', label=r'$\tau_{l}$')
# # ax.plot(x, lap_tau_l, c='r', lw=2, ls='dotted', label=r'$\nabla\tau_{l}$')
# ax.plot(sep, P_ad, c='b', lw=2, label=r'$P_{A\delta}$')
# ax.plot(sep, P_ad_spec, c='r', ls='dashed', lw=2, label=r'$P_{A\delta}:$ spectral')
#
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel(r'$\tau_{l}$', fontsize=14)
# ax.legend()
# plt.savefig('../plots/EFT_nbody/s.png', bbox_inches='tight', dpi=120)
# plt.close()


###Code to test spectral_calc post generalisation of L
##the function choice of cosine is just one (simplest) example. could take anything else
# L = 2 * np.pi
# N = 1000
# dx = L / N
# x = np.arange(0, L, dx)
# A = 0.5 * np.sin(2 * np.pi * x / L)
# n = 5
# k = np.fft.fftfreq(x.size, dx) * L
# dA = spectral_calc(A, k, L, o=n, d=0)
# signs = [+1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1]
# if n%2 == 0:
#    A_ = A
# else:
#    A_ = 0.5 * np.cos(2 * np.pi * x / L)
# dA_an = signs[n-1] * ((2*np.pi)**n / L**n) * A_ #np.cos(2 * np.pi * x / L)
#
# fig, ax = plt.subplots()
# ax.set_title('order = {}'.format(n))
# ax.plot(x, dA_an, c='k', lw=2, label='An')
# ax.plot(x, dA, c='b', lw=2, ls='dashed', label='Num')
# plt.savefig('../plots/cosmo_sim/spec_test.png', bbox_inches='tight', dpi=120)

##from functions one Apr 13, 2022; 11:30
# def husimi(wfc, x, p, sigma_x, h, L, ini=0):
#     hu = np.empty(shape=(len(p), len(x)), dtype=complex)
#     dx = x[1] - x[0]
#     dv = p[1] - p[0]
#     # A = ((2 * np.pi * h) ** (-1/2))# * np.sqrt(np.pi / sigma_x)#((2 * np.pi * sigma_x **2) ** (-1/4))
#     A = ((2 * np.pi * h) ** (-1/2)) * ((2 * np.pi * sigma_x **2) ** (-1/4)) #the normalisation of Husimi
#     for m in range(0, len(p)):
#         dist = x - 0
#         dist[dist < 0] += L
#         dist[dist > L/2] = - L + dist[dist > L/2]
#         g = np.exp(-(sigma_x * (dist)**2) + 1j*p[m]*dist/h)
#         hu[m] = A * np.exp(-1j*p[m]*x/(2*h))*(np.fft.ifft(np.fft.fft(g)*np.fft.fft(wfc)))
#     f = np.abs(hu)**2
#     return f


### from EFT_nbody.py on April 22, 2022; 11:21
# from scipy.optimize import curve_fit
# def fitting_function(X, c, n):
#    P_spt, P11, a, mode = X
#    return P_spt + ((c * (a**n)) * (mode[0]**2) * P11)

# guesses = 1, 1
# FF = curve_fit(fitting_function, (P_1l_tr, P_lin, a_list, k[mode]*np.ones(a_list.size)), P_nb, guesses, sigma=1e-5*np.ones(a_list.size), method='lm')
# c, n = FF[0]
# cov = FF[1]
# err_c, err_n = np.sqrt(np.diag(cov))
# fit = fitting_function((P_1l_tr, P_lin, a_list, k[mode]*np.ones(a_list.size)), c, n)

# alpha_c_fit = (fit - P_1l_tr) / (2 * P_lin * k[mode]**2)

# fig, ax = plt.subplots()
# ax.set_title(r'$k = {}, \Lambda = {}$'.format(mode, Lambda))
# ax.set_ylabel(r'$\frac{\Delta P}{P_{\mathrm{NL}}}$', fontsize=18)
# ax.set_xlabel(r'$a$', fontsize=14)
#
# err_lin = (P_lin - P_nb) / P_nb
# err_1l_tr = (P_1l_tr - P_nb) / P_nb
# err_eft_tr = (P_eft_tr - P_nb) / P_nb
# err_eft2_tr = (P_eft2_tr - P_nb) / P_nb
# err_eft3_tr = (P_eft3_tr - P_nb) / P_nb
#
# err_fit = (fit - P_nb) / P_nb
#
# ax.axhline(0, label=r'$P^{\mathrm{N-body}}_{\mathrm{NL}}$', lw=2.5, c='b')
# ax.plot(a_list, err_lin, label=r'$P^{\mathrm{SPT}}_{\mathrm{lin}}$', ls='dashed', lw=2.5, c='r')
# ax.plot(a_list, err_1l_tr, label=r'$P^{\mathrm{SPT}}_{\mathrm{1-loop}}$', ls='dashed', lw=2.5, c='brown')
# ax.plot(a_list, err_eft_tr, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ fit to $\tau_{l}$', ls='dashdot', lw=2.5, c='k')
# ax.plot(a_list, err_eft2_tr, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from M&W estimator', ls='dashdot', lw=2.5, c='cyan')
# ax.plot(a_list, err_eft3_tr, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from from correlations (Baumann)', ls='dashdot', lw=2.5, c='orange')
# ax.plot(a_list, err_fit, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from fit to $P^{\mathrm{N-body}}_{\mathrm{NL}}$', ls='dashdot', lw=2.5, c='green')
#
#
# ax.minorticks_on()
#
# ax.tick_params(axis='both', which='both', direction='in')
# ax.ticklabel_format(scilimits=(-2, 3))
# ax.grid(lw=0.2, ls='dashed', color='grey')
# ax.yaxis.set_ticks_position('both')
# ax.legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/EFT_nbody/results/EFT_nbody_k{}_L{}.png'.format(mode, Lambda), bbox_inches='tight', dpi=150)
# plt.close()

##from EFT_nbody.py on May 5, 2022; 15:27

# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [3, 1]})
# ax[0].set_title(r'$k = {}, \Lambda = {} \;[2\pi h^{{-1}}\;\mathrm{{Mpc}}]$'.format(mode, int(Lambda/(2*np.pi))))
# # ax[0].set_title(r'$k = {}$'.format(mode))
# ax[0].set_ylabel(r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$|\tilde{\delta}(k=1, a)|^{2} / a^{2}$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$P(k=1, a)$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$c_{\mathrm{tot}}^{2}\;[\mathrm{km}^{2}\mathrm{s}^{-2}]$', fontsize=14) # / a^{2}$')
#
# ax[1].set_xlabel(r'$a$', fontsize=14)
#
# # # # # # ax[0].plot(a_list, alpha_c_naive, c='b', lw=2)
# # # # # # ax[0].plot(a_list, alpha_c_fit, label='fit', ls='dashed', c='k', lw=2)
# # # # #
# # # # # ax[0].plot(a_list, cs2_list, c='b', ls='dashed', lw=2, label=r'$c^{2}_{\mathrm{s}}$')
# # # # # ax[0].plot(a_list, cv2_list, c='r', ls='dotted', lw=2, label=r'$c^{2}_{\mathrm{v}}$')
# # ax[0].plot(a_list, ctot2_list, c='k', lw=2.5, label=r'from fit to $\tau_{l}$')
# # ax[0].plot(a_list, ctot2_list2, c='cyan', ls='dashdot', lw=2.5, label=r'M&W')
# # ax[0].plot(a_list, ctot2_list3, c='orange', ls='dashed', lw=2.5, label=r'Baumann')
#
# ax[0].plot(a_list, alpha_c_true, c='blue', lw=2.5, label=r'from matching $P^{\mathrm{N-body}}_{\mathrm{NL}}$')
# # ax[0].plot(a_list, alpha_c_fit, c='green', ls='dashdot', lw=2.5, label=r'from fit to $P^{\mathrm{N-body}}_{\mathrm{NL}}$')
# ax[0].plot(a_list, alpha_c_naive, c='k', ls='dashdot', lw=2.5, label=r'from fit to $\tau_{l}$')
# ax[0].plot(a_list, alpha_c_naive2, c='cyan', ls='dashdot', lw=2.5, label=r'M&W')
# ax[0].plot(a_list, alpha_c_naive3, c='orange', ls='dashdot', lw=2.5, label=r'Baumann')
# #
# #
# # ax[0].plot(a_list, P_nb / a_list**2, label=r'$P^{\mathrm{N-body}}_{\mathrm{NL}}$', lw=2.5, c='b')
# # # ax[0].plot(a_list, P_lin / a_list**2, label=r'$P^{\mathrm{SPT}}_{\mathrm{lin}}$', lw=2.5, c='r')
# #
# # # truncated spectra
# # ax[0].plot(a_list, P_1l_tr / a_list**2, label=r'$P^{\mathrm{SPT}}_{\mathrm{1-loop}}$', lw=2.5, c='brown')
# # ax[0].plot(a_list, P_eft_tr / a_list**2, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from fit to $\tau_{l}$', ls='dashdot', lw=2.5, c='k')
# # ax[0].plot(a_list, P_eft2_tr / a_list**2, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ (M&W)', ls='dashdot', lw=2.5, c='cyan')
# # ax[0].plot(a_list, P_eft3_tr / a_list**2, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ (Baumann)', ls='dashdot', lw=2.5, c='orange')
# # ax[0].plot(a_list, P_eft_fit / a_list**2, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from matching to $P^{\mathrm{N-body}}_{\mathrm{NL}}$', ls='dashdot', lw=2.5, c='green')
# # # ax[0].set_yscale('log')
#
# # # smoothed spectra
# # ax[0].plot(a_list, P_1l_sm, label=r'$P^{\mathrm{SPT}}_{\mathrm{1-loop}}$: smoothed', ls='dashed', lw=2.5, c='brown')
# # ax[0].plot(a_list, P_eft_sm, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}}$ from fit', ls='dashdot', lw=3, c='k')
# # ax[0].plot(a_list, P_eft2_sm, label=r'$P^{\mathrm{EFT}}_{\mathrm{1-loop}} from correlations$', ls='dashdot', lw=3, c='cyan')
#
# # # smoothed spectra
# # ax[0].plot(a_list, P_1l_sm, label=r'SPT: $P_{\mathrm{1-loop}}$: smoothed', lw=2.5, c='magenta')
# # ax[0].plot(a_list, P_eft_sm, label=r'EFT: via $c^{2}_{\mathrm{tot}}:$ smoothed', ls='dashed', lw=3, c='cyan')
# # ax[0].plot(a_list, P_eft2_sm, label=r'EFT: directly from $\tau_{l}:$ smoothed', ls='dashed', lw=3, c='orange')
#
# # #bottom panel; errors
# ax[1].axhline(0, color='b')
#
# # #error on the linear PS
# # err_lin = (P_lin - P_nb) * 100 / P_nb
# # # ax[1].plot(a_list, err_lin, lw=2.5, c='r')
# #
# # #errors on the truncated spectra
# # err_1l_tr = (P_1l_tr - P_nb) * 100 / P_nb
# # err_eft_tr = (P_eft_tr - P_nb) * 100 / P_nb
# # err_eft2_tr = (P_eft2_tr - P_nb) * 100 / P_nb
# # err_eft3_tr = (P_eft3_tr - P_nb) * 100 / P_nb
# #
# # err_fit = (P_eft_fit - P_nb) * 100 / P_nb
# #
# # ax[1].plot(a_list, err_1l_tr, lw=2.5, c='brown')
# # ax[1].plot(a_list, err_eft_tr, ls='dashdot', lw=2.5, c='k')
# # ax[1].plot(a_list, err_eft2_tr, ls='dashdot', lw=2.5, c='cyan')
# # ax[1].plot(a_list, err_eft3_tr, ls='dashdot', lw=2.5, c='orange')
# #
# # ax[1].plot(a_list, err_fit, ls='dashdot', lw=2.5, c='green')
#
# # err_ctot2_2 = (ctot2_list2 - ctot2_list) * 100 / ctot2_list
# # err_ctot2_3 = (ctot2_list3 - ctot2_list) * 100 / ctot2_list
# #
# # ax[1].axhline(0, c='k')
# # ax[1].plot(a_list, err_ctot2_2, ls='dashdot', lw=2.5, c='cyan')
# # ax[1].plot(a_list, err_ctot2_3, ls='dashdot', lw=2.5, c='orange')
#
#
# err_alpha = (alpha_c_naive - alpha_c_true) * 100 / P_nb
# err_alpha_2 = (alpha_c_naive2 - alpha_c_true) * 100 / P_nb
# err_alpha_3 = (alpha_c_naive3 - alpha_c_true) * 100 / P_nb
# # err_alpha_fit = (alpha_c_fit - alpha_c_true) * 100 / P_nb
#
# ax[1].axhline(0, c='blue')
# ax[1].plot(a_list, err_alpha, ls='dashdot', lw=2.5, c='k')
# ax[1].plot(a_list, err_alpha_2, ls='dashdot', lw=2.5, c='cyan')
# ax[1].plot(a_list, err_alpha_3, ls='dashdot', lw=2.5, c='orange')
# # ax[1].plot(a_list, err_alpha_fit, ls='dashdot', lw=2.5, c='green')
#
# # ax[1].set_ylim(-5, 5)
#
# # #errors on the smoothed spectra
# # err_1l_sm = (P_1l_sm - P_nb) * 100 / P_nb
# # err_eft_sm = (P_eft_sm - P_nb) * 100 / P_nb
# # err_eft2_sm = (P_eft2_sm - P_nb) * 100 / P_nb
# # ax[1].plot(a_list, err_1l_sm, lw=2.5, ls='dashed', c='brown')
# # ax[1].plot(a_list, err_eft_sm, ls='dashdot', lw=3, c='k')
# # ax[1].plot(a_list, err_eft2_sm, ls='dashdot', lw=3, c='cyan')
#
# # #errors on the smoothed spectra
# # err_1l_sm = (P_1l_sm - P_nb) * 100 / P_nb
# # err_eft_sm = (P_eft_sm - P_nb) * 100 / P_nb
# # err_eft2_sm = (P_eft2_sm - P_nb) * 100 / P_nb
# # ax[1].plot(a_list, err_1l_sm, lw=2.5, c='magenta')
# # ax[1].plot(a_list, err_eft_sm, ls='dashed', lw=3, c='cyan')
# # ax[1].plot(a_list, err_eft2_sm, ls='dashed', lw=3, c='orange')
#
# # ax[1].plot(a_list, fit, ls='dotted', lw=3, c='green')
#
# ax[1].set_ylabel('% err', fontsize=14)
# # ax[1].set_ylabel('difference', fontsize=14)
#
# ax[1].minorticks_on()
#
# for i in range(2):
#     ax[i].tick_params(axis='both', which='both', direction='in')
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].grid(lw=0.2, ls='dashed', color='grey')
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/{}/PS_l{}.png'.format(plots_folder, int(Lambda/(2*np.pi))), bbox_inches='tight', dpi=150)
# plt.close()
#
# # print('c = ', c)
# # print('n = ', n)
#
# fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'width_ratios': [1], 'height_ratios': [4, 1]})
# ax[0].set_title(r'$k = {}, \Lambda = {} \;[2\pi h^{{-1}}\;\mathrm{{Mpc}}]$'.format(mode, int(Lambda/(2*np.pi))))
# # ax[0].set_title(r'$k = {}$'.format(mode))
# # ax[0].set_ylabel(r'$|\tilde{\delta}(k)|^{2}$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$\alpha_c\;[h^{-2}\mathrm{Mpc}^{2}]$', fontsize=14) # / a^{2}$')
# # ax[0].set_ylabel(r'$c_{{\mathrm{{tot}}}}^{{2}}\;[\mathrm{{km}}^{{2}}\mathrm{{s}}^{{-2}}]$', fontsize=14) # / a^{2}$')
# ax[0].set_ylabel(r'$|\tau_{l}(k)|^{2}$', fontsize=14)
#
# ax[1].set_xlabel(r'$a$', fontsize=16)
#
# # ax[0].plot(a_list, alpha_c_fit, label='fit', c='k', lw=2)
# # ax[0].plot(a_list, alpha_c_naive, label='measured', ls='dashed', c='r', lw=2)
#
# # ax[0].plot(a_list, ctot2_list, c='k', lw=2)
#
# ax[0].plot(a_list, np.log(tau_list), label='N-body', lw=2.5, c='b')
# ax[0].plot(a_list, np.log(fit_list), label='fit', ls='dashed', lw=2.5, c='k')
# # ax[0].plot(a_list, fit_k, label='fit in k-space', ls='dashed', lw=2.5, c='r')
#
# err_fit = (fit_list - tau_list) * 100 / tau_list
# # err_k_fit = (fit_k - tau_list) * 100 / tau_list
#
# #bottom panel; errors
# ax[1].axhline(0, color='b')
#
# ax[1].plot(a_list, err_fit, ls='dashed', lw=2.5, c='k')
# # ax[1].plot(a_list, err_k_fit, ls='dashed', lw=2.5, c='r')
#
# ax[1].set_ylabel('% err', fontsize=16)
# # ax[1].set_ylim(-100, 500)
# ax[1].minorticks_on()
#
# for i in range(2):
#     ax[i].tick_params(axis='both', which='both', direction='in')
#     ax[i].ticklabel_format(scilimits=(-2, 3))
#     ax[i].grid(lw=0.2, ls='dashed', color='grey')
#     ax[i].yaxis.set_ticks_position('both')
#
# ax[0].legend(fontsize=11, loc=2, bbox_to_anchor=(1,1))
#
# plt.savefig('../plots/{}/tau_fit_k{}_l{}.png'.format(plots_folder, mode, int(Lambda/(2*np.pi))), bbox_inches='tight', dpi=150)
# # print(err_fit)
# # plt.savefig('../plots/sch_hfix_run19/eft_k{}_l{}_fit.png'.format(mode, Lambda), bbox_inches='tight', dpi=120)
# # plt.savefig('../plots/sch_hfix_run19/alpha_c_l{}.png'.format(Lambda), bbox_inches='tight', dpi=120)
# # plt.savefig('../plots/sch_hfix_run19/ctot2_l{}_test.png'.format(Lambda), bbox_inches='tight', dpi=120)
#
# plt.close()

##from ensembles.py on Sep 5, 2022; 11:20
# def fitting_function(X, a0, a1):
#   x1 = X
#   return a0 + a1*x1
# guesses = 1, 1
# FF = curve_fit(fitting_function, (dc_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
# C0, C1 = FF[0]
# cov = FF[1]
# err0, err1 = np.sqrt(np.diag(cov))
# err2 = 0
# C1 += (err1)
#
# # print(C0, C1)
# fit = fitting_function((dc_l), C0, C1)
# fit_sp = fit[0::n_ev]
# C = [C0, C1]
#
# cs2 = np.real(C1 / rho_b)
# cv2 = 0
# ctot2 = cs2
#
# cov = np.array(cov)
# corr = np.zeros(cov.shape)
#
# for i in range(2):
#     for j in range(2):
#         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#
# print(C0, C1)
# print(cov)
# print(corr)
#
# solving the weighted linear least-squares of the form Y = X\beta + resid, with weight matrix W
# X = np.array([n_ev, sum(dc_l_sp), sum(dv_l_sp), sum(dc_l_sp), sum(dc_l_sp**2), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp**2)])
# Y = np.array([sum(tau_l_sp), sum(tau_l_sp*dc_l_sp), sum(tau_l_sp*dv_l_sp)])
#
#
# X = (np.array([n_use, sum(dc_l_sp), sum(dv_l_sp), sum(dc_l_sp), sum(dc_l_sp**2), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp), sum(dv_l_sp*dc_l_sp), sum(dv_l_sp**2)])).reshape((3,3))
# Y = np.array([sum(tau_l_sp), sum(tau_l_sp*dc_l_sp), sum(tau_l_sp*dv_l_sp)])
# cov = np.linalg.inv(X.T.dot(X))
# # for element in cov:
# #     element = (element - np.mean(cov))**2
#
#
# dc_l_sp = np.array(dc_l_sp)
# dv_l_sp = np.array(dv_l_sp)
# tau_l_sp = np.array(tau_l_sp)
#
# C0, C1, C2 = (np.linalg.inv(X)).dot(Y)
#
# err0 = np.sqrt((sum(np.ones(dc_l_sp.size)**2) - (dc_l_sp.size * np.mean(np.ones(dc_l_sp.size))**2)) / dc_l_sp.size)
# err1 = np.sqrt((sum(dc_l_sp**2) - (dc_l_sp.size * np.mean(dc_l_sp)**2)) / dc_l_sp.size)
# err2 = np.sqrt((sum(dv_l_sp**2) - (dv_l_sp.size * np.mean(dv_l_sp)**2)) / dv_l_sp.size)
#
# err12 = np.sqrt((sum(np.abs(dc_l_sp*dv_l_sp)) - (dv_l_sp.size * np.mean(dc_l_sp) * np.abs(np.mean(dv_l_sp)))) / dv_l_sp.size)
#
# print(A, B)
#
# C0, C1, C2 = np.linalg.lstsq(X, Y)[0]
#
# print(C0_, C1_, C2_)

###from llsq_fits.py on Oct 20, 2022 (16:28). Contains the code that went into the function weighted_ls_fit().
#
# w = 1/yerr_sp
# Xw = (np.array([sum(w), sum(w*dc_l_sp), sum(w*dv_l_sp), sum(w*dc_l_sp), sum(w*dc_l_sp**2), sum(w*dv_l_sp*dc_l_sp), sum(w*dv_l_sp), sum(w*dv_l_sp*dc_l_sp), sum(w*dv_l_sp**2)])).reshape((3,3))
# yw = np.array([sum(w*tau_l_sp), sum(w*tau_l_sp*dc_l_sp), sum(w*tau_l_sp*dv_l_sp)])
# x = (np.ones(w.size), dc_l_sp, dv_l_sp)
# W = np.diag([sum(w), sum(w), sum(w)])
# cov__ = (np.linalg.inv(Xw.T.dot(W.dot(Xw)))).reshape((3,3))
# corr__ = np.zeros(cov.shape)
#
# for i in range(3):
#     for j in range(3):
#         corr__[i,j] = cov__[i,j] / np.sqrt(cov__[i,i]*cov__[j,j])
#
# err1__ = cov__[1,1]
# err2__ = cov__[2,2]
#
# terr__ = np.sqrt(err1__**2 + err2__**2 + corr__[1,2]*err1__*err2__ + corr__[2,1]*err2__*err1__)
#
# C0__, C1__, C2__ = (np.linalg.inv(Xw)).dot(yw)
# C__ = [C0__, C1__, C2__]

##from functions.py on Oct 20, 2022 (19:18). Contains the code for fitting using curve_fit()
    # guesses = 1, 1, 1
    # FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    # C0, C1, C2 = FF[0]
    # cov = FF[1]
    # err0, err1, err2 = np.sqrt(np.diag(cov))
    #
    # fit = fitting_function((dc_l, dv_l), C0, C1, C2)
    # fit_sp = fit[0::n_ev]
    # C = [C0, C1, C2]
    # # print(C)
    #
    # cs2 = np.real(C1 / rho_b)
    # cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
    # ctot2 = (cs2 + cv2)
    #
    # C1_fd = np.zeros(tau_l.size)
    # C2_fd = np.zeros(tau_l.size)
    #
    # # d_dcl_list = []
    # # for i in range(1, tau_l.size):
    # #     d_dcl = (dc_l[i] - dc_l[i-1])
    # #     d_dvl = (dv_l[i] - dv_l[i-1])
    # #     d_dcl_list.append(d_dc_l)
    # #     d_tau = (tau_l[i] - tau_l[i-1])
    # #     C1_fd[i] =  d_tau / d_dcl
    # #     C2_fd[i] =  d_tau / d_dvl
    # #
    # #     # print(d_tau, d_dcl)
    # # d_dcl_list = np.array(d_dcl_list)
    # # print(d_dcl_list[x.size-5:x.size+5])
    #
    # # plt.plot(x, C1_fd, c='b')
    # # plt.axhline(np.mean(C1_fd), c='k', ls='dotted')
    # # plt.axhline(C1, c='r', ls='dashed')
    #
    # # tau_new = np.mean(tau_l) + C1_fd*dc_l + C2_fd*dv_l
    # # plt.plot(x, tau_l, c='k')
    # # plt.plot(x, tau_new, c='b', ls='dashed')
    # # plt.plot(x, fit, c='r', ls='dotted')
    # #
    # # print(np.mean(tau_l))
    # # # plt.plot(x, C2_fd, c='b')
    # # # plt.axhline(np.mean(C2_fd), c='k', ls='dotted')
    # # # plt.axhline(C2, c='r', ls='dashed')
    # #
    # # plt.show()
    #
    #
    # # def fitting_function(X, a0, a1, a2, a3):
    # #     x1, x2, x3 = X
    # #     return a0 + a1*x1 + a2*x2 + a3*(x3)
    # # # v_l = spectral_calc(dv_l, 1.0, o=1, d=1)
    # #
    # # field3 = dc_l**2
    # # field3_sp = field3[0::n_ev]
    # # guesses = 1, 1, 1, 1
    # # FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp, field3_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
    # # C0, C1, C2, C3 = FF[0]
    # # cov = FF[1]
    # # err0, err1, err2, err3 = np.sqrt(np.diag(cov))
    # #
    # # fit = fitting_function((dc_l, dv_l, field3), C0, C1, C2, C3)
    # # fit_sp = fit[0::n_ev]
    # # C = [C0, C1, C2, C3]
    # #
    # # cs2 = np.real(C1 / rho_b)
    # # cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
    # # c1 = np.real(C3 / rho_b)
    # # print(cs2, cv2, c1)
    # # ctot2 = (cs2 + cv2 + c1)
    # cov = np.array(cov)
    # corr = np.zeros(cov.shape)
    #
    # for i in range(3):
    #     for j in range(3):
    #         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])

## code for data covariance from functions.py on Oct 25, 2022; 12:27

    # cov_mat = np.empty(shape=(n_use, n_use))
    # corr_mat = np.empty(shape=(n_use, n_use))
    # # cov_mat[i][j] = (taus[0][i] - np.mean([taus[l][i] for l in range(n_runs)]))*(taus[0][j] - np.mean([taus[l][j] for l in range(n_runs)]))
    #
    #
    # for i in range(n_use):
    #     for j in range(n_use):
    #         cov_mean = []
    #         for k in range(n_runs):
    #             tau_l_0_sp = taus[k][0::n_ev]
    #             cov_0 = (tau_l_0_sp[i] - np.mean(tau_l_sp)) * (tau_l_0_sp[j] - np.mean(tau_l_sp))
    #             cov_mean.append(cov_0)
    #             # cov_mat[i][j] = (taus[0][i] - np.mean(taus[0])) * (taus[0][j] - np.mean(taus[0]))
    #         cov_mat[i][j] = np.mean(cov_mean)
    #
    # for i in range(n_use):
    #     for j in range(n_use):
    #         # print(cov_mat[i,j], cov_mat[i,i], cov_mat[j,j])
    #         corr_mat[i,j] = cov_mat[i,j] / np.sqrt(cov_mat[i,i] * cov_mat[j,j])
    #
    # import seaborn as sns
    # plt.figure(figsize=(10,10))
    # hm = sns.heatmap(corr_mat,
    #              cbar=True,
    #              annot=False,
    #              square=True,
    #              fmt='.3f',
    #              annot_kws={'size': 8})
    # plt.title('Correlation matrix')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('../plots/test/new_paper_plots/data_corr_{}.png'.format(n_use), bbox_inches='tight', dpi=150)

## code from functions.py (old version of param_calc_ens) taken on Oct 28, 2022; 16:07
# def param_calc_ens(j: int, Lambda: float, path: str, A: list, mode: int, kind: str, n_runs: int, n_use: int) --> tuple:
#     a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
#     taus = []
#     taus.append(tau_l_0)
#
#     if modes == True:
#         for run in range(2, n_runs+1):
#             if run > 10:
#                 ind = -3
#             else:
#                 ind = -2
#             path = path[:ind] + '{}/'.format(run)
#             taus.append(read_sim_data(path, Lambda, kind, j, folder_name)[1])
#     else:
#         for j in range(2, n_runs+1):
#             taus.append(tau_l_0)
#
#     taus = np.array(taus)
#     Nt = len(taus)
#
#     tau_l = sum(np.array(taus)) / Nt
#
#     rho_0 = 27.755
#     rho_b = rho_0 / a**3
#     H0 = 100
#
#     diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
#     yerr = np.sqrt(sum(diff) / (Nt-1))
#
#
#     # # print(x.size, np.sort(sub))
#     # n_ev = int(x.size / (n_use-1))
#     # if len(sub) == 0:
#     #     dc_l_sp = dc_l[0::n_ev]
#     #     dv_l_sp = dv_l[0::n_ev]
#     #     tau_l_sp = tau_l[0::n_ev]
#     #     yerr_sp = yerr[0::n_ev]
#     #     x_sp = x[0::n_ev]
#     #     taus_sp = np.array([taus[k][0::n_ev] for k in range(n_runs)])
#
#     # else:
#     n_fits = 15
#     # print(n_fits)
#     FF = []
#     for j in range(n_fits):
#         sub = sub_find(n_use, x.size)
#         dc_l_sp = np.array([dc_l[j] for j in sub])
#         dv_l_sp = np.array([dv_l[j] for j in sub])
#         tau_l_sp = np.array([tau_l[j] for j in sub])
#         yerr_sp = np.array([yerr[j] for j in sub])
#         x_sp = np.array([x[j] for j in sub])
#         taus_sp = np.array([np.array([taus[k][j] for j in sub]) for k in range(n_runs)])
#
#         cov_mat = np.empty(shape=(n_use, n_use))
#         for i in range(n_use):
#             for j in range(n_use):
#                 tau_k_i = np.array([taus_sp[k][i] for k in range(n_runs)])
#                 tau_k_j = np.array([taus_sp[k][j] for k in range(n_runs)])
#                 cov = np.cov(tau_k_i, tau_k_j)
#                 cov_mat[i][j] = cov[0][1]
#
#         def fitting_function(X, a0, a1, a2):
#             x1, x2 = X
#             return a0 + a1*x1 + a2*x2
#
#         if fitting_method == 'WLS':
#             ### weighted_ls_fit begins here
#             X = (np.ones(yerr_sp.size), dc_l_sp, dv_l_sp)
#             C, cov, corr = weighted_ls_fit(tau_l_sp, X, len(X), cov_mat)
#
#             C0, C1, C2 = C
#             err0 = cov[0,0]
#             err1 = cov[1,1]
#             err2 = cov[2,2]
#             ### weighted_ls_fit ends here
#
#         elif fitting_method == 'curve_fit':
#             ### curve_fit begins here
#             guesses = 1, 1, 1
#             # FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
#             FF.append(curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=cov_mat, method='lm', absolute_sigma=True))
#
#             # C0, C1, C2 = FF[0][0]
#             # cov = FF[0][1]#[FF[k][1] for k in range(n_fits)]
#             # # err0, err1, err2 = np.sqrt(np.diag(cov))
#
#             # C = [C0, C1, C2]
#             #
#             # cov = np.array(cov)
#
#             # corr = np.zeros(cov.shape)
#             #
#             # for i in range(3):
#             #     for j in range(3):
#             #         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#             ### curve_fit ends here
#
#         elif fitting_method == 'lmfit':
#             d = tau_l_sp
#             X = (np.ones(dc_l_sp.size), dc_l_sp, dv_l_sp)
#             err = yerr_sp
#             C, cov, corr, red_chi = lmfit_est(d, X, err, cov_mat)
#             err0, err1, err2 = np.sqrt(np.diag(cov))
#             C0, C1, C2 = C
#
#         else:
#             raise Exception('Fitting method must be curve_fit or WLS')
#     # #
#     # cov_mat = np.empty(shape=(n_use, n_use))
#     # corr_mat = np.empty(shape=(n_use, n_use))
#     # # for i in range(n_use):
#     # #     for j in range(n_use):
#     # #         cov_mean = []
#     # #         for k in range(n_runs):
#     # #             if len(sub) == 0:
#     # #                 tau_l_0_sp = taus[k][0::n_ev]
#     # #             else:
#     # #                 tau_l_0_sp = [taus[k][p] for p in sub]
#     # #             # cov_0 = (tau_l_0_sp[i] - np.mean(tau_l_sp)) * (tau_l_0_sp[j] - np.mean(tau_l_sp))
#     # #             cov_0 = (tau_l_0_sp[i] - tau_l[i]) * (tau_l_0_sp[j] -tau_l[j])
#     # #
#     # #             cov_mean.append(cov_0)
#     # #         cov_mat[i][j] = np.mean(cov_mean) * (len(cov_mean) / (len(cov_mean)-1))
#     #
#     # # for i in range(n_use):
#     # #     for j in range(n_use):
#     # #         cov_mat[i][j] = sum([(taus[k][i] - tau_l[i]) * (taus[k][j] - tau_l[j]) for k in range(n_runs)]) / (n_runs-1)
#     #
#     # for i in range(n_use):
#     #     for j in range(n_use):
#     #         cov_mat[i][j] = [(taus[k][i] - (sum(np.array([taus[k][i] for k in range(n_runs)])) / n_runs)) * (taus[k][j] - (sum(np.array([taus[k][j] for k in range(n_runs)])) / n_runs)) for k in range(n_runs)] / (n_runs-1)
#     #
#     #
#     # for i in range(n_use):
#     #     for j in range(n_use):
#     #         # print(cov_mat[i,j], cov_mat[i,i], cov_mat[j,j])
#     #         corr_mat[i,j] = cov_mat[i,j] / np.sqrt(cov_mat[i,i] * cov_mat[j,j])
#     #
#     # import seaborn as sns
#     # plt.figure(figsize=(10,10))
#     # hm = sns.heatmap(corr_mat,
#     #              cbar=True,
#     #              annot=True,
#     #              square=True,
#     #              fmt='.3f',
#     #              annot_kws={'size': 10})
#     # plt.title('Correlation matrix')
#     # plt.xlabel(r'$\tau(x_{j})$', fontsize=16)
#     # plt.ylabel(r'$\tau(x_{i})$', fontsize=16)
#     #
#     # plt.tight_layout()
#     # plt.show()
#     # # plt.savefig('../plots/test/new_paper_plots/data_corr_{}.png'.format(n_use), bbox_inches='tight', dpi=150)
#
#     C0 = np.mean([FF[k][0][0] for k in range(n_fits)])
#     C1 = np.mean([FF[k][0][1] for k in range(n_fits)])
#     C2 = np.mean([FF[k][0][2] for k in range(n_fits)])
#     C = C0, C1, C2
#
#     # # cov = np.empty(shape=(n_use, n_use))
#     # # cov_k = [FF[k][1] for k in range(n_fits)]
#     # cov_k = (FF[0][1])**2
#     # for k in range(1, n_fits):
#     #     cov_k += (FF[k][1])**2
#     # cov = np.sqrt(cov_k) / np.sqrt(n_fits)
#     cov = np.sqrt((sum([(FF[k][1])**2 for k in range(n_fits)])) / n_fits)
#
#     fit = fitting_function((dc_l, dv_l), C0, C1, C2)
#     if len(sub) == 0:
#         fit_sp = fit[0::n_ev]
#     else:
#         fit_sp = np.array([fit[j] for j in sub])
#
#     if fitting_method != 'lmfit':
#         resid = fit_sp - tau_l_sp
#         chisq = sum((resid / yerr_sp)**2)
#         red_chi = chisq / (n_use - 3)
#
#     cs2 = np.real(C1 / rho_b)
#     cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
#     ctot2 = (cs2 + cv2)
#
#     f1 = (1 / rho_b)
#     f2 = (-H0 / (rho_b * np.sqrt(a)))
#
#     cov[0,1] *= f1
#     cov[1,0] *= f1
#     cov[0,2] *= f2
#     cov[2,0] *= f2
#     cov[1,1] *= f1**2
#     cov[2,2] *= f2**2
#     cov[2,1] *= f2*f1
#     cov[1,2] *= f1*f2
#
#     corr = np.zeros(cov.shape)
#
#     for i in range(3):
#         for j in range(3):
#             corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#
#     err0, err1, err2 = np.sqrt(np.diag(cov))
#
#     ctot2 = (cs2 + cv2)
#     terr = (err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)**(0.5)
#
#     # print(ctot2, terr)
#
#     # M&W Estimator
#     Lambda_int = int(Lambda / (2*np.pi))
#     tau_l_k = np.fft.fft(tau_l) / x.size
#     num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
#     denom = ((d1k * np.conj(d1k)) * (a**2))
#     ntrunc = int(num.size-Lambda_int)
#     num[Lambda_int+1:ntrunc] = 0
#     denom[Lambda_int+1:ntrunc] = 0
#
#     ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b
#
#     # Baumann estimator
#     def Power(f1_k, f2_k, Lambda_int):
#       corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
#       ntrunc = corr.size - Lambda_int
#       corr[Lambda_int+1:ntrunc] = 0
#       return corr
#
#     A = np.fft.fft(tau_l) / rho_b / tau_l.size
#     T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
#     d = np.fft.fft(dc_l) / dc_l.size
#     Ad = Power(A, dc_l, Lambda_int)[mode]
#     AT = Power(A, T, Lambda_int)[mode]
#     P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
#     P_TT = Power(T, T, Lambda_int)[mode]
#     P_dT = Power(dc_l, T, Lambda_int)[mode]
#
#     cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
#     cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
#
#     ctot2_3 = np.real(cs2_3 + cv2_3)
#     return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb, P_1l, d1k


###
# def param_calc_ens(path: str, j: int, Lambda: float, kind: str, mode: int, n_runs: int, n_use: int, n_fits=1) -> tuple:
#     """This function calculates the EFT parameters for a given simulation snapshot using user-defined criteria.
#     Three estimators are used: 1) from fit to the stress, 2) B^{+12}, 3) M\&W.
#
#     Arguments
#     ----------
#     path: directory of simulation files.
#     j: index of the snapshot.
#     Lambda: Fourier space smoothing scale.
#     kind: kind of smoothing, can be sharp or smooth. In the smooth case, a Gaussian smoothing is used.
#     mode: Fourier mode for which we solve, only relevant for B^{+12} and M\&W estimators.
#     n_runs: number of realisations of a simulation with given initial overdensity, but different phases.
#     n_use: number of points used for the fitting. This must be smaller than n_runs.
#     n_fits: number of fits. n_fits = 1 means that a single fit will be used. n_fits > 1 averages over multiple fits.
#
#     Returns
#     ----------
#     a: float
#         scalefactor of the snapshot.
#     x: array
#         comoving position.
#     tau_l: array
#         coarse-grained stress.
#     fit: array
#         fit to tau_l, possibly from an average over many fits.
#     P_nb: array
#         N-body power spectrum for the given mode at each scalefactor (all spectra defined in this way).
#     P_SPT2: array
#         SPT power spectrum to second order in the overdensity. Equivalent to the linear power spectrum.
#     P_SPT4: array
#         SPT power spectrum to fourth order in the overdensity. For a Gaussian overdensity, equivalent to the 1-loop spectrum.
#     ctots: list
#         contains arrays of the three estimates of ctot2, each array contains the ctot2 for a given scalefactor.
#     terr: array
#         contains the total error on the ctot2 array measured from fitting.
#     cs2: array
#         cs2 measured from the fit at each scalefactor.
#     cv2: array
#         cv2 measured from the fit at each scalefactor.
#     red_chi: array
#         the reduced chi-square for the fit.
#     """
#
#     # Read the simulation data, define useful quantities
#     a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_SPT4 = read_sim_data(path, Lambda, kind, j)
#     P_SPT2 = np.real(d1k * np.conj(d1k)) * a**2
#     rho_b = 27.755 / a**3 # Background density
#     H0 = 100 # Hubble constant in units of h
#
#     # Create the array of all stress scalars
#     taus = []
#     taus.append(tau_l_0)
#     for run in range(2, n_runs+1):
#         if run > 10:
#             ind = -3
#         else:
#             ind = -2
#         path = path[:ind] + '{}/'.format(run)
#         taus.append(read_sim_data(path, Lambda, kind, j)[1])
#
#     Nt = len(taus)
#     taus = np.array(taus)
#     tau_l = sum(np.array(taus)) / Nt #this is the mean tau to be used for the fit
#
#
#     # Define the fitting function for three parameters and two input fields x1, x2
#     def fitting_function(X: tuple, C: tuple):
#         """Evaluates the fitting function for a given set of parameters C and a tuple of input fields X."""
#         x1, x2 = X
#         a0, a1, a2 = C
#         return a0 + a1*x1 + a2*x2
#
#     # Evaluate the parameters
#
#     # Here, we average over many fits with different points to get the estimated parameters
#     if n_fits > 1:
#         FF = []
#         for j in range(n_fits):
#             sub = sub_find(n_use, x.size) # Calculates a random, uniformly spaced set of indices to pick which points to fit
#
#             # Subsample the arrays to the points used for the fit
#             dc_l_sp = np.array([dc_l[j] for j in sub])
#             dv_l_sp = np.array([dv_l[j] for j in sub])
#             tau_l_sp = np.array([tau_l[j] for j in sub])
#             x_sp = np.array([x[j] for j in sub])
#             taus_sp = np.array([np.array([taus[k][j] for j in sub]) for k in range(n_runs)])
#
#             # Construct the covariance matrix of the errors on tau_l
#             cov_mat = np.empty(shape=(n_use, n_use))
#             for i in range(n_use):
#                 for j in range(n_use):
#                     tau_k_i = np.array([taus_sp[k][i] for k in range(n_runs)])
#                     tau_k_j = np.array([taus_sp[k][j] for k in range(n_runs)])
#                     cov = np.cov(tau_k_i, tau_k_j)
#                     cov_mat[i][j] = cov[0][1]
#
#
#             guesses = 1, 1, 1 # Initial guess for the fitting routine
#
#             # Fitting using curve_fit. We provide the covariance matrix of the errors on tau_l as input.
#             # Each call to curve_fit returns the estimated parameters and their covariance matrix
#             # We append these to a list for each fit
#             FF.append(curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=cov_mat, method='lm', absolute_sigma=True))
#
#         # Estimate the parameters as the mean of each individual fit, and calculate the combined covariance matrix
#         C0 = np.median([FF[k][0][0] for k in range(n_fits)])
#         C1 = np.median([FF[k][0][1] for k in range(n_fits)])
#         C2 = np.median([FF[k][0][2] for k in range(n_fits)])
#         C = [C0, C1, C2]
#         cov = np.empty(shape=(n_use, n_use))
#         for i in range(n_use):
#             for j in range(n_use):
#                 cov[i][j] = np.median(FF[i][j]) #np.sqrt((sum([(FF[k][1])**2 for k in range(n_fits)])) / n_fits)
#
#     else:
#         sub = sub_find(n_use, x.size) # Calculates a random, uniformly spaced set of indices to pick which points to fit
#
#         # Subsample the arrays to the points used for the fit
#         dc_l_sp = np.array([dc_l[j] for j in sub])
#         dv_l_sp = np.array([dv_l[j] for j in sub])
#         tau_l_sp = np.array([tau_l[j] for j in sub])
#         x_sp = np.array([x[j] for j in sub])
#         taus_sp = np.array([np.array([taus[k][j] for j in sub]) for k in range(n_runs)])
#
#         cov_mat = np.empty(shape=(n_use, n_use))
#         for i in range(n_use):
#             for j in range(n_use):
#                 tau_k_i = np.array([taus_sp[k][i] for k in range(n_runs)])
#                 tau_k_j = np.array([taus_sp[k][j] for k in range(n_runs)])
#                 cov = np.cov(tau_k_i, tau_k_j)
#                 cov_mat[i][j] = cov[0][1]
#
#         guesses = 1, 1, 1 # Initial guess for the fitting routine
#
#         # Fitting using curve_fit. We provide the covariance matrix of the errors on tau_l as input.
#         # Each call to curve_fit returns the estimated parameters and their covariance matrix
#         FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=cov_mat, method='lm', absolute_sigma=True)
#         C0, C1, C2 = FF[0]
#         C = [C0, C1, C2]
#         cov = FF[1]
#
#     # Evaluate the fit using the estiamted parameters
#     fit = fitting_function((dc_l, dv_l), C0, C1, C2)
#     fit_sp = np.array([fit[j] for j in sub])
#
#     # Calculate the residual and the chi-square
#     resid = fit_sp - tau_l_sp
#     chisq = np.dot(resid.T, np.dot(np.linalg.inv(cov_mat), resid))
#     red_chi = chisq / (n_use - 3)
#
#     #
#     cs2 = np.real(C1 / rho_b)
#     cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
#     ctot2 = (cs2 + cv2)
#
#     f1 = (1 / rho_b)
#     f2 = (-H0 / (rho_b * np.sqrt(a)))
#
#     cov[0,1] *= f1
#     cov[1,0] *= f1
#     cov[0,2] *= f2
#     cov[2,0] *= f2
#     cov[1,1] *= f1**2
#     cov[2,2] *= f2**2
#     cov[2,1] *= f2*f1
#     cov[1,2] *= f1*f2
#
#     corr = np.zeros(cov.shape)
#
#     for i in range(3):
#         for j in range(3):
#             corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#
#     err0, err1, err2 = np.sqrt(np.diag(cov))
#
#     ctot2 = (cs2 + cv2)
#     terr = (err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)**(0.5)
#
#
#     # M&W Estimator
#     Lambda_int = int(Lambda / (2*np.pi))
#     tau_l_k = np.fft.fft(tau_l) / x.size
#     num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
#     denom = ((d1k * np.conj(d1k)) * (a**2))
#     ntrunc = int(num.size-Lambda_int)
#     num[Lambda_int+1:ntrunc] = 0
#     denom[Lambda_int+1:ntrunc] = 0
#
#     ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b
#
#     # Baumann estimator
#     def Power(f1_k, f2_k, Lambda_int):
#       corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
#       ntrunc = corr.size - Lambda_int
#       corr[Lambda_int+1:ntrunc] = 0
#       return corr
#
#     A = np.fft.fft(tau_l) / rho_b / tau_l.size
#     T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
#     d = np.fft.fft(dc_l) / dc_l.size
#     Ad = Power(A, dc_l, Lambda_int)[mode]
#     AT = Power(A, T, Lambda_int)[mode]
#     P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
#     P_TT = Power(T, T, Lambda_int)[mode]
#     P_dT = Power(dc_l, T, Lambda_int)[mode]
#
#     cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
#     cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
#
#     ctot2_3 = np.real(cs2_3 + cv2_3)
#
#     ctots = [ctot2, ctot2_2, ctot2_3]
#     return a, x, tau_l, fit, P_nb, P_lin, P_1l, ctots, terr, cs2, cv2, red_chi


# def param_calc_ens(j, Lambda, path, A, mode, kind, n_runs, n_use, folder_name, fitting_method='curve_fit'):
#     a, x, d1k, dc_l, dv_l, tau_l_0, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, folder_name)
#     taus = []
#     taus.append(tau_l_0)
#     modes = True
#     if modes == True:
#         for run in range(2, n_runs+1):
#             if run > 10:
#                 ind = -3
#             else:
#                 ind = -2
#             path = path[:ind] + '{}/'.format(run)
#             taus.append(read_sim_data(path, Lambda, kind, j, folder_name)[1])
#     else:
#         for j in range(2, n_runs+1):
#             taus.append(tau_l_0)
#
#     taus = np.array(taus)
#     Nt = len(taus)
#
#     tau_l = sum(np.array(taus)) / Nt
#
#     rho_0 = 27.755
#     rho_b = rho_0 / a**3
#     H0 = 100
#
#     diff = np.array([(taus[i] - tau_l)**2 for i in range(1, Nt)])
#     yerr = np.sqrt(sum(diff) / (Nt-1))
#
#
#     # # print(x.size, np.sort(sub))
#     # n_ev = int(x.size / (n_use-1))
#     # if len(sub) == 0:
#     #     dc_l_sp = dc_l[0::n_ev]
#     #     dv_l_sp = dv_l[0::n_ev]
#     #     tau_l_sp = tau_l[0::n_ev]
#     #     yerr_sp = yerr[0::n_ev]
#     #     x_sp = x[0::n_ev]
#     #     taus_sp = np.array([taus[k][0::n_ev] for k in range(n_runs)])
#
#     # else:
#     n_fits = 100
#     # print(n_fits)
#     FF = []
#     for j in range(n_fits):
#         sub = sub_find(n_use, x.size)
#         dc_l_sp = np.array([dc_l[j] for j in sub])
#         dv_l_sp = np.array([dv_l[j] for j in sub])
#         tau_l_sp = np.array([tau_l[j] for j in sub])
#         yerr_sp = np.array([yerr[j] for j in sub])
#         x_sp = np.array([x[j] for j in sub])
#         taus_sp = np.array([np.array([taus[k][j] for j in sub]) for k in range(n_runs)])
#
#         cov_mat = np.empty(shape=(n_use, n_use))
#         for i in range(n_use):
#             for j in range(n_use):
#                 tau_k_i = np.array([taus_sp[k][i] for k in range(n_runs)])
#                 tau_k_j = np.array([taus_sp[k][j] for k in range(n_runs)])
#                 cov = np.cov(tau_k_i, tau_k_j)
#                 cov_mat[i][j] = cov[0][1]
#
#         def fitting_function(X, a0, a1, a2):
#             x1, x2 = X
#             return a0 + a1*x1 + a2*x2
#
#         if fitting_method == 'WLS':
#             ### weighted_ls_fit begins here
#             X = (np.ones(yerr_sp.size), dc_l_sp, dv_l_sp)
#             C, cov, corr = weighted_ls_fit(tau_l_sp, X, len(X), cov_mat)
#
#             C0, C1, C2 = C
#             err0 = cov[0,0]
#             err1 = cov[1,1]
#             err2 = cov[2,2]
#             ### weighted_ls_fit ends here
#
#         elif fitting_method == 'curve_fit':
#             ### curve_fit begins here
#             guesses = 1, 1, 1
#             # FF = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=yerr_sp, method='lm', absolute_sigma=True)
#             sol = curve_fit(fitting_function, (dc_l_sp, dv_l_sp), tau_l_sp, guesses, sigma=cov_mat, method='lm', absolute_sigma=True)
#             FF.append(sol)
#
#             # C0, C1, C2 = FF[0][0]
#             # cov = FF[0][1]#[FF[k][1] for k in range(n_fits)]
#             # # err0, err1, err2 = np.sqrt(np.diag(cov))
#
#             # C = [C0, C1, C2]
#             #
#             # cov = np.array(cov)
#
#             # corr = np.zeros(cov.shape)
#             #
#             # for i in range(3):
#             #     for j in range(3):
#             #         corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#             ### curve_fit ends here
#
#         elif fitting_method == 'lmfit':
#             d = tau_l_sp
#             X = (np.ones(dc_l_sp.size), dc_l_sp, dv_l_sp)
#             err = yerr_sp
#             C, cov, corr, red_chi = lmfit_est(d, X, err, cov_mat)
#             err0, err1, err2 = np.sqrt(np.diag(cov))
#             C0, C1, C2 = C
#
#         else:
#             raise Exception('Fitting method must be curve_fit or WLS')
#     # #
#     # cov_mat = np.empty(shape=(n_use, n_use))
#     # corr_mat = np.empty(shape=(n_use, n_use))
#     # # for i in range(n_use):
#     # #     for j in range(n_use):
#     # #         cov_mean = []
#     # #         for k in range(n_runs):
#     # #             if len(sub) == 0:
#     # #                 tau_l_0_sp = taus[k][0::n_ev]
#     # #             else:
#     # #                 tau_l_0_sp = [taus[k][p] for p in sub]
#     # #             # cov_0 = (tau_l_0_sp[i] - np.mean(tau_l_sp)) * (tau_l_0_sp[j] - np.mean(tau_l_sp))
#     # #             cov_0 = (tau_l_0_sp[i] - tau_l[i]) * (tau_l_0_sp[j] -tau_l[j])
#     # #
#     # #             cov_mean.append(cov_0)
#     # #         cov_mat[i][j] = np.mean(cov_mean) * (len(cov_mean) / (len(cov_mean)-1))
#     #
#     # # for i in range(n_use):
#     # #     for j in range(n_use):
#     # #         cov_mat[i][j] = sum([(taus[k][i] - tau_l[i]) * (taus[k][j] - tau_l[j]) for k in range(n_runs)]) / (n_runs-1)
#     #
#     # for i in range(n_use):
#     #     for j in range(n_use):
#     #         cov_mat[i][j] = [(taus[k][i] - (sum(np.array([taus[k][i] for k in range(n_runs)])) / n_runs)) * (taus[k][j] - (sum(np.array([taus[k][j] for k in range(n_runs)])) / n_runs)) for k in range(n_runs)] / (n_runs-1)
#     #
#     #
#     # for i in range(n_use):
#     #     for j in range(n_use):
#     #         # print(cov_mat[i,j], cov_mat[i,i], cov_mat[j,j])
#     #         corr_mat[i,j] = cov_mat[i,j] / np.sqrt(cov_mat[i,i] * cov_mat[j,j])
#     #
#     # import seaborn as sns
#     # plt.figure(figsize=(10,10))
#     # hm = sns.heatmap(corr_mat,
#     #              cbar=True,
#     #              annot=True,
#     #              square=True,
#     #              fmt='.3f',
#     #              annot_kws={'size': 10})
#     # plt.title('Correlation matrix')
#     # plt.xlabel(r'$\tau(x_{j})$', fontsize=16)
#     # plt.ylabel(r'$\tau(x_{i})$', fontsize=16)
#     #
#     # plt.tight_layout()
#     # plt.show()
#     # # plt.savefig('../plots/test/new_paper_plots/data_corr_{}.png'.format(n_use), bbox_inches='tight', dpi=150)
#
#     C0 = np.median([FF[k][0][0] for k in range(n_fits)])
#     C1 = np.median([FF[k][0][1] for k in range(n_fits)])
#     C2 = np.median([FF[k][0][2] for k in range(n_fits)])
#     C = [C0, C1, C2]
#     # print(FF[0][1])
#     # print(FF[0][1][0][0])
#     # cov = np.empty(shape=(n_use, n_use))
#     # cov = np.zeros(shape=(n_use, n_use))
#     cov = np.sqrt((sum([(FF[k][1])**2 for k in range(n_fits)])) / n_fits)
#     for i in range(3):
#         for j in range(3):
#             # print([FF[k][1][i][j] for k in range(2)])
#             cov[i][j] = np.median([FF[k][1][i][j] for k in range(n_fits)]) #np.sqrt((sum([(FF[k][1])**2 for k in range(n_fits)])) / n_fits)
#
#
#     fit = fitting_function((dc_l, dv_l), C0, C1, C2)
#     if len(sub) == 0:
#         fit_sp = fit[0::n_ev]
#     else:
#         fit_sp = np.array([fit[j] for j in sub])
#
#     if fitting_method != 'lmfit':
#         resid = fit_sp - tau_l_sp
#         chisq = sum((resid / yerr_sp)**2)
#         red_chi = chisq / (n_use - 3)
#
#     cs2 = np.real(C1 / rho_b)
#     cv2 = np.real(-C2 * H0 / (rho_b * np.sqrt(a)))
#     ctot2 = (cs2 + cv2)
#
#     f1 = (1 / rho_b)
#     f2 = (-H0 / (rho_b * np.sqrt(a)))
#
#     cov[0,1] *= f1
#     cov[1,0] *= f1
#     cov[0,2] *= f2
#     cov[2,0] *= f2
#     cov[1,1] *= f1**2
#     cov[2,2] *= f2**2
#     cov[2,1] *= f2*f1
#     cov[1,2] *= f1*f2
#
#     corr = np.zeros(cov.shape)
#
#     for i in range(3):
#         for j in range(3):
#             corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
#
#     err0, err1, err2 = np.sqrt(np.diag(cov))
#
#     ctot2 = (cs2 + cv2)
#     try:
#         terr = (err1**2 + err2**2 + corr[1,2]*err1*err2 + corr[2,1]*err2*err1)**(0.5)
#     except RuntimeWarning:
#         terr = 0
#     # print(ctot2, terr)
#
#     # M&W Estimator
#     Lambda_int = int(Lambda / (2*np.pi))
#     tau_l_k = np.fft.fft(tau_l) / x.size
#     num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
#     denom = ((d1k * np.conj(d1k)) * (a**2))
#     ntrunc = int(num.size-Lambda_int)
#     num[Lambda_int+1:ntrunc] = 0
#     denom[Lambda_int+1:ntrunc] = 0
#
#     ctot2_2 = np.real(sum(num) / sum(denom)) / rho_b
#
#     # Baumann estimator
#     def Power(f1_k, f2_k, Lambda_int):
#       corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
#       ntrunc = corr.size - Lambda_int
#       corr[Lambda_int+1:ntrunc] = 0
#       return corr
#
#     A = np.fft.fft(tau_l) / rho_b / tau_l.size
#     T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
#     d = np.fft.fft(dc_l) / dc_l.size
#     Ad = Power(A, dc_l, Lambda_int)[mode]
#     AT = Power(A, T, Lambda_int)[mode]
#     P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
#     P_TT = Power(T, T, Lambda_int)[mode]
#     P_dT = Power(dc_l, T, Lambda_int)[mode]
#
#     cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
#     cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
#
#     ctot2_3 = np.real(cs2_3 + cv2_3)
#     return a, x, ctot2, ctot2_2, ctot2_3, err0, err1, err2, cs2, cv2, red_chi, yerr, tau_l, fit, terr, P_nb, P_1l, d1k
