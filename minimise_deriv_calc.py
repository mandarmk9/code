#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from functions import read_sim_data, param_calc_ens
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2 * np.pi)
kind = 'sharp'
kind_txt = 'sharp cutoff'
# kind = 'gaussian'
# kind_txt = 'Gaussian smoothing'

j = 39
a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j)
print('a = {}'.format(np.round(a, 3)))
dv_l *= -np.sqrt(a) / 100
tau_l -= np.mean(tau_l)

def deriv_param_calc(dc_l, dv_l, tau_l):
    def new_param_calc(dc_l, dv_l, tau_l, dist, ind):
        def dir_der_o1(X, tau_l, ind):
            """Calculates the first-order directional derivative of tau_l along the vector X."""
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind+1], X[1][ind+1]])
            v = (x2 - x1)
            D_v_tau = (tau_l[ind+1] - tau_l[ind]) / v[0]
            return v, D_v_tau

        def dir_der_o2(X, tau_l, ind):
            """Calculates the second-order directional derivative of tau_l along the vector X."""
            v0, D_v_tau0 = dir_der_o1(X, tau_l, ind-2)
            v1, D_v_tau1 = dir_der_o1(X, tau_l, ind)
            v2, D_v_tau2 = dir_der_o1(X, tau_l, ind+2)
            x0 = np.array([X[0][ind-2], X[1][ind-2]])
            x1 = np.array([X[0][ind], X[1][ind]])
            x2 = np.array([X[0][ind+2], X[1][ind+2]])
            v = (x2 - x1)
            D2_v_tau = (D_v_tau2 - D_v_tau1) / v[0]
            return v, D2_v_tau

        X = np.array([dc_l, dv_l])
        params_list = []
        for j in range(-dist//2, dist//2 + 1):
            v1, dtau1 = dir_der_o1(X, tau_l, ind+j)
            v1_o2, dtau1_o2 = dir_der_o2(X, tau_l, ind+j)
            dc_0, dv_0 = dc_l[ind], dv_l[ind]
            C_ = [((tau_l[ind])-(dtau1*dc_0)+((dtau1_o2*dc_0**2)/2)), dtau1-(dtau1_o2*dc_0), dtau1_o2/2]
            # C_ = [(tau_l[ind]), dtau1, dtau1_o2/2]
            params_list.append(C_)


        params_list = np.array(params_list)
        C0_ = np.mean(np.array([params_list[j][0] for j in range(dist)]))
        C1_ = np.mean(np.array([params_list[j][1] for j in range(dist)]))
        C2_ = np.mean(np.array([params_list[j][2] for j in range(dist)]))

        C_ = [C0_, C1_, C2_]
        return C_

    def minimise_deriv(params, dc_l, dv_l, tau_l):
        start, thresh = params
        N = dc_l.size
        if start < 0:
            start = N + start
        dist = 10
        n_sub = 50
        C_list = []
        # start = 8000
        sub = np.linspace(start, N-start+1, n_sub, dtype=int)
        print(start, N-start+1, sub[-1])
        del_ind = np.argmax(tau_l)
        # thresh = 8000
        for point in sub:
            if del_ind-thresh < point < del_ind+thresh:
                sub = np.delete(sub, np.where(sub==point)[0][0])
            else:
                pass
        n_sub = sub.size
        print(sub[-1])
        for j in range(n_sub):
            tau_val = tau_l[sub[j]]
            tau_diff = np.abs(tau_l - tau_val)
            ind_tau = np.argmin(tau_diff)
            dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
            ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
            C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
            # print(j, C_[0])
            C_list.append(C_)

        try:
            C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
            C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
            C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
        except:
            C0_ = 0
            C1_ = 10
            C2_ = 0

        C_ = [C0_, C1_, C2_]
        fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

        resid = sum((tau_l - fit)**2)
        return resid

    x0 = (8000, 8000)
    meth = 'Powell'
    sol = minimize(minimise_deriv, x0, args=(dc_l, dv_l, tau_l), bounds=[(0, 20000), (0, 20000)], method=meth)

    sol_params = sol.x
    # print(sol_params, sol.fun)
    def calc_fit(params, dc_l, dv_l, tau_l):
        start, thresh = params
        dist = 10
        n_sub = 50
        C_list = []
        N = dc_l.size
        sub = np.linspace(start, N-start+1, n_sub, dtype=int)
        del_ind = np.argmax(tau_l)
        for point in sub:
            if del_ind-thresh < point < del_ind+thresh:
                sub = np.delete(sub, np.where(sub==point)[0][0])
            else:
                pass
        n_sub = sub.size

        for j in range(n_sub):
            tau_val = tau_l[sub[j]]
            tau_diff = np.abs(tau_l - tau_val)
            ind_tau = np.argmin(tau_diff)
            dc_0, dv_0 = dc_l[ind_tau], dv_l[ind_tau]
            ind = np.argmin((dc_l-dc_0)**2 + (dv_l-dv_0)**2)
            C_ = new_param_calc(dc_l, dv_l, tau_l, dist, ind)
            C_list.append(C_)

        C0_ = np.mean([C_list[l][0] for l in range(len(C_list))])
        C1_ = np.mean([C_list[l][1] for l in range(len(C_list))])
        C2_ = np.mean([C_list[l][2] for l in range(len(C_list))])
        err0_ = sum([(C_list[l][0] - C0_)**2 for l in range(len(C_list))])
        err1_ = sum([(C_list[l][0] - C2_)**2 for l in range(len(C_list))])
        err2_ = sum([(C_list[l][0] - C2_)**2 for l in range(len(C_list))])

        C_ = [C0_, C1_, C2_]
        err_ = [err0_, err1_, err2_]
        fit = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2

        return C_, fit

    C_, fit2 = calc_fit(sol_params, dc_l, dv_l, tau_l)
    return C_, err_


guesses = 1, 1, 1

def fitting_function(X, a0, a1, a2):
    x1 = X
    return a0 + a1*x1 + a2*(x1**2)
C, cov = curve_fit(fitting_function, (dc_l), tau_l, guesses, sigma=np.ones(dc_l.size), method='lm', absolute_sigma=True)
fit = C[0] + C[1]*dc_l + C[2]*dc_l**2


C_ = deriv_param_calc(dc_l, dv_l, tau_l)
fit2 = C_[0] + C_[1]*dc_l + C_[2]*dc_l**2
print(C_)
fig, ax = plt.subplots()
ax.plot(x, tau_l, c='b')
ax.plot(x, fit, c='k', ls='dashdot')
ax.plot(x, fit2, c='r', ls='dashed')
plt.savefig('../plots/test/new_paper_plots/test.png', bbox_inches='tight', dpi=150)
plt.close()
