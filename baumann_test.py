#!/usr/bin/env python3
import os
import numpy as np
from functions import read_sim_data
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

path = 'cosmo_sim_1d/sim_k_1_11/run1/'
Lambda = 3 * (2*np.pi)

kind = 'sharp'
kind_txt = 'sharp cutoff'

kind = 'gaussian'
kind_txt = 'Gaussian smoothing'
mode = 1
j = 0

for j in range(51):
    a, x, d1k, dc_l, dv_l, tau_l, P_nb, P_1l = read_sim_data(path, Lambda, kind, j, '')

    rho_0 = 27.755
    H0 = 100
    rho_b = rho_0 / a**3

    # M&W Estimator
    Lambda_int = int(Lambda / (2*np.pi))
    num = (np.conj(a * d1k) * ((np.fft.fft(tau_l)) / x.size))
    denom = ((d1k * np.conj(d1k)) * (a**2))
    ntrunc = int(num.size-Lambda_int)
    ctot2_2 = np.real(sum(num[:Lambda_int+1]) / sum(denom[:Lambda_int+1])) / rho_b

    # Baumann estimator
    def Power(f1_k, f2_k, Lambda_int):
      corr = (f1_k * np.conj(f2_k) + np.conj(f1_k) * f2_k) / 2
      ntrunc = corr.size - Lambda_int
      corr[Lambda_int+1:ntrunc] = 0
      return corr

    A = np.fft.fft(tau_l) / rho_b / tau_l.size
    T = np.fft.fft(dv_l) / (H0 / (a**(1/2))) / dv_l.size
    d = np.fft.fft(dc_l) / dc_l.size
    # A = tau_l / tho_b
    # T = dv_l / (H0 / (a**(1/2)))

    # Ad = Power(A, dc_l, Lambda_int)[mode]
    # AT = Power(A, T, Lambda_int)[mode]
    # P_dd = Power(dc_l, dc_l, Lambda_int)[mode]
    # P_TT = Power(T, T, Lambda_int)[mode]
    # P_dT = Power(dc_l, T, Lambda_int)[mode]


    Ad = Power(A, d, Lambda_int)[mode]
    AT = Power(A, T, Lambda_int)[mode]
    P_dd = Power(d, d, Lambda_int)[mode]
    P_TT = Power(T, T, Lambda_int)[mode]
    P_dT = Power(d, T, Lambda_int)[mode]

    print('\na = ', a)
    print(np.real(dc_l - d))
    # try:
    #     cs2_3 = ((P_TT * Ad) - (P_dT * AT)) / (P_dd * P_TT - (P_dT)**2)
    #     cv2_3 = ((P_dT * Ad) - (P_dd * AT)) / (P_dd * P_TT - (P_dT)**2)
    #
    #     N1 = ((P_TT * Ad) - (P_dT * AT))
    #     N2 = ((P_dT * Ad) - (P_dd * AT))
    #     D = (P_dd * P_TT - (P_dT)**2)
    #
    #     print(np.real(N1+N2), np.real(D))
    #     ctot2_3 = np.real(cs2_3 + cv2_3)
    #     print(ctot2_2, ctot2_3)
    #
    # except:
    #     print(ctot2_2, 0)
    #     pass
