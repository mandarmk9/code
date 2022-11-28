#!/usr/bin/env python3
"""
This script writes out the initial conditions using the ZA (1LPT).
Starting from the specified initial density d0 and Lagrangian coordinates (q),
we compute the Zeldovich displacement field \Psi, and the Eulerian positions (x).
Finally, the Eulerian density contrast at a desired time (a) is written out.

date created: May 12, 2021
version: 0.0
author: mandarmk9
"""

# import modules
import numpy as np
# import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from functions import spectral_calc

def initial_density(q, A, L):
    N_waves = len(A) // 2
    den = 0
    for j in range(0, N_waves):
        den += A[2*j] * np.cos(2 * np.pi * q * A[2*j+1] / L)
    return den

# def initial_density(q, A, L):
#     return (A[0] * np.cos(2 * np.pi * q * A[1] / L)) + (A[2] * np.cos(2 * np.pi * q * A[3] / L))

def eulerian_sampling(q, a, A, L):

    def initial_density(q, A, L):
        N_waves = len(A) // 2
        den = 0
        for j in range(0, N_waves):
            den += A[2*j] * np.cos(2 * np.pi * q * A[2*j+1] / L)
        return den

    def nabla_Psi(q, A, a, L):
        return - a * initial_density(q, A, L)

    def Psi(q, A, a, L):
        N = q.size
        del_Psi = nabla_Psi(q, A, a, L)
        k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
        return spectral_calc(del_Psi, k, o=1, d=1)

    def eul_pos(q, A, a, L):
        disp = Psi(q, A, a, L)
        return q + disp

    guess = 0
    def f(point):
        N_waves = len(A) // 2
        term = 0
        for j in range(0, N_waves):
            term += (A[2*j] * L / (2 * np.pi * A[2*j+1])) * np.sin(2 * np.pi * point * A[2*j+1] / L)
        return point - a * term - c

    # def f(point):
    #     return point - a * (((A[0] * L / (2 * np.pi * A[1])) * np.sin(2 * np.pi * point * A[1] / L)) + ((A[2] * L / (2 * np.pi * A[3])) * np.sin(2 * np.pi * point * A[3] / L))) - c

    q_traj = np.empty(q.size)
    for i in range(q.size):
        c = q[i]
        q_traj[i] = fsolve(f, guess)

    delta = (np.abs(1 + nabla_Psi(q_traj, A, a, L)) ** (-1)) - 1
    return q_traj, delta

# def eulerian_sampling(q, a, A, L):
#     def initial_density(q, A, L):
#         return (A[0] * np.cos(2 * np.pi * q * A[1] / L)) + (A[2] * np.cos(2 * np.pi * q * A[3] / L))
#
#     def nabla_Psi(q, A, a, L):
#         return - a * initial_density(q, A, L)
#
#     def Psi(q, A, a, L):
#         N = q.size
#         del_Psi = nabla_Psi(q, A, a, L)
#         k = np.fft.ifftshift(2.0 * np.pi / L * np.arange(-N/2, N/2))
#         return spectral_calc(del_Psi, k, o=1, d=1)
#
#     def eul_pos(q, A, a, L):
#         disp = Psi(q, A, a, L)
#         return q + disp
#
#     guess = 0
#     def f(point):
#         return point - a * (((A[0] * L / (2 * np.pi * A[1])) * np.sin(2 * np.pi * point * A[1] / L)) + ((A[2] * L / (2 * np.pi * A[3])) * np.sin(2 * np.pi * point * A[3] / L))) - c
#
#     q_traj = np.empty(q.size)
#     for i in range(q.size):
#         c = q[i]
#         q_traj[i] = fsolve(f, guess)
#         # print(c, q_traj[i])
#
#     delta = (np.abs(1 + nabla_Psi(q_traj, A, a, L)) ** (-1)) - 1
#     return q_traj, delta

# delta = eulerian_sampling(q, a, A)[1]
# print('mean eulerian overdensity at a = {} is {}'.format(np.round(a, 3), np.mean(delta)))

# # some plotting
# fig, ax = plt.subplots()
# ax.plot(q, delta, color='k')
# # ax.plot(q, delta_x2, color='b')
# ax.set_ylabel(r'$\delta(x)$')
# ax.set_xlabel(r'$x\;[h^{-1}\mathrm{Mpc}]$')
# plt.show()
#
# # print('mean overdensity initially is {}'.format(np.mean(initial_density(q, A))))
# # mean_eul_den = np.trapz(delta_x, dx=x[1]-x[0])
# # mean_lag_den = np.trapz(delta_q, dx=q[1]-q[0])
