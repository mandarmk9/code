#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solving for the cosmological SPT Kernels in 1D, following 309PT notation using sympy

author : @mandar
date created : 09/09/2020
"""
from sympy import *
import sympy.core.numbers as num
import numpy as np
import itertools
import dill
import h5py
import time

def kernels(n, s=0, f=True, g=False, write=False):
    def alpha(k1, k2):
        return (k1 + k2) * k1 / (k1**2)
    def beta(k1, k2):
        return ((k1 + k2)**2) * (k1 * k2) / (2 * (k1**2) * (k2**2))
    def F(n, modes):
        if len(modes) != n:
            raise ValueError
        if n == 1:
            return 1
        else:
            J = 0
            for m in range(n-1):
                q1m = modes[0:m+1]
                qmn = modes[m+1:n+1]
                k1 = sum(q1m)
                k2 = sum(qmn)
                kernel = G(m + 1, q1m) * ((2 * n + 1) * alpha(k1, k2) * F(n - m - 1, qmn) + 2 * beta(k1, k2) * G(n - m - 1, qmn)) / ((2 * n + 3 ) * (n - 1))
                J += kernel
                if write == True and f == True:
                    M = expand(simplify(kernel))
                    dill.dump(M, open('./spt_kernels/F{}.txt'.format(m+2), mode='wb'))
                    # pprint(M)
            return J
    def G(n, modes):
        if len(modes) != n:
            raise ValueError
        if n == 1:
            return 1
        else:
            J = 0
            for m in range(n-1):
                q1m = modes[0:m+1]
                qmn = modes[m+1:n+1]
                k1 = sum(q1m)
                k2 = sum(qmn)
                kernel = G(m + 1, q1m) * (3 * alpha(k1, k2) * F(n - m - 1, qmn) + 2 * n * beta(k1, k2) * G(n - m - 1, qmn)) / ((2 * n + 3 ) * (n - 1))
                J += kernel
                if write == True and g == True:
                    dill.dump(J, open('./spt_kernels/G{}.txt'.format(m+2), mode='wb'))
            return J
    def sym(n, modes, f=True, g=False):
        perm = list(itertools.permutations(modes))
        Fn = 0
        Gn = 0
        for j in range(factorial(n)):
            modes = perm[j]
            if f == True:
                Fn += F(n, modes)
            if g == True:
                Gn += G(n, modes)
        Fn = Fn / factorial(n)
        Gn = Gn / factorial(n)
        return [Fn, Gn]
    modes = symbols('q{}:{}'.format(1, n+1))
    if f == True:
        if s == 0:
            Fn = expand(simplify(F(n, modes)))
            # Fn = F(n, modes)
        elif s == 1:
            Fn = expand(simplify(sym(n, modes, f=f, g=g)[0]))
    if g == True:
        if s == 0:
            Gn = expand(simplify(G(n, modes)))
            # Gn = F(n, modes)
        elif s == 1:
            Gn = expand(simplify(sym(n, modes, f=f, g=g)[1]))
    # Fn = F(n, modes)
    # Gn = G(n, modes)
    # Fn = expand(simplify(Fn))
    # Gn = expand(simplify(Gn))
    if g == False:
        return [Fn, 0]
    elif f == False:
        return [0, Gn]
    else:
        return [Fn, Gn]

def sim(kern, n, fun, s=0):
    if n == 1:
        return 1
    else:
        x = symbols('x')
        modes = symbols('q{}:{}'.format(1, n+1))
        def sim1(l, kern, n):
            m = [[] for _ in range(n)]
            terms = list(kern.args[l].args)
            g = kern.args[l].func
            A = []
            B = []
            for i in range(n):
                for j in range(len(terms)):
                    if type(terms[j]) == num.Rational or  type(terms[j]) == num.Half or type(terms[j]) == num.One or type(terms[j]) == num.Integer or type(terms[j]) == num.NegativeOne:
                        B.append(terms[j])
                    if str(modes[i]) in str(terms[j]):
                        m[i].append(terms[j])
                A.append(prod(m[i]))

            A.extend(list(set(B)))
            return A

    C = []
    for l in range(len(kern.args)):
        C.append(sim1(l, kern, n))

    def stream(C, n, fun):
        dict = {}
        for i in range(len(C)):
            for j in range(n):
                for k in range(len(C[i])):
                    for o in range(n):
                        args = np.repeat(x, o)
                        if C[i][k] == fun(modes[j]) * (modes[j] ** o):
                            C[i][k] = diff(fun(x), x, o)
                        elif C[i][k] == fun(modes[j]) / (modes[j] ** o):
                            C[i][k] = integrate(fun(x), *args)
                dict[str(modes[j])] = x
            C[i] = prod(C[i])
        return sum(C).subs(dict)

    if s == 1:
        return C
    elif s == 0:
        return stream(C, n, fun)

#Don't touch anything above this line

def SPT_solve(n, K, solved=0):
    """using the SPT module and an analytical initial density,
    this function returns a numerical density function of order n
    """
    x = symbols('x')
    A = [-0.05, 1, -0.5, 7, 0]
    L = 1.0
    dc_in = (A[0] * cos(2 * np.pi * x * A[1] / L)) + (A[2] * cos(2 * np.pi * x * A[3] / L))

    H = Function('H')
    modes = symbols('q{}:{}'.format(1, n+1))
    for mode in modes:
        K *= H(mode)
    K = expand(K)
    # ans = sim(K, n, H)
    if solved == 0:
        if n == 1:
            return dc_in
        else:
            ans = simplify(sim(K, n, H))
            return ans
    elif solved == 1:
        if n == 1:
            return lambdify(x, dc_in, 'numpy')
        else:
            ans = simplify(sim(K, n, H).subs([(H(x), dc_in)]))
            f = lambdify(x, ans, 'numpy')
            return f

def SPT_agg(n, x, s=0):
    F = np.empty(shape=(n, len(x)))
    if s == 0:
        K0 = kernels(1, s=0, f=1, g=0, write=0)[0]
    elif s == 1:
        K0 = kernel_sym(1)
    F[0] = SPT_solve(1, K0, solved=1)(x)
    print('F1 computed, solving F2.')
    for j in range(1, n):
        t0 = time.time()
        if s == 0:
            K = kernels(j + 1, s=0, f=1, g=0, write=0)[0]
        elif s == 1:
            K = kernel_sym(j + 1)
        F[j] = SPT_solve(j + 1, K, solved=1)(x)
        t1 = time.time()
        print('F{} computed. Time for last computation : {} seconds'.format(j + 1, np.round(t1 - t0, 3)))
    return F

def sim2(kern, n):
    modes = symbols('q{}:{}'.format(1, n+1))
    def sim1(l, kern, n):
        m = [[] for _ in range(n)]
        terms = list(kern.args[l].args)
        g = kern.args[l].func
        A = []
        B = []
        for i in range(n):
            for j in range(len(terms)):
                if type(terms[j]) == Rational or type(terms[j]) == S.Half:
                    B.append(terms[j])
                if str(modes[i]) in str(terms[j]):
                    m[i].append(terms[j])
            A.append(prod(m[i]))
        A.extend(list(set(B)))
        return A

    C = []
    for l in range(len(kern.args)):
        C.append(sim1(l, kern, n))

    return C

def kernel_sym(n, s=0):
    modes = symbols('q{}:{}'.format(1, n+1))
    H = Function('H')
    Fn = (sum(modes) ** n) / (factorial(n) * prod(modes))
    if s == 0:
        return Fn
    for mode in modes:
        Fn *= H(mode)
    Fn = expand(Fn)
    if s == 1:
        return Fn
    elif s == 2:
        Fn = lambdify([modes, H], Fn)
        return Fn

def spt_write_to_hdf5(filename, F):
    # import time
    with h5py.File(filename, mode='w') as hdf:
        for j in range(1, len(F) + 1):
            # t0 = time.time()
            hdf.create_dataset('F{}'.format(j), data=F[j - 1])
            # t1 = time.time()
            # print('F{} written, took {} seconds'.format(j, np.round(t1 - t0, 3)))
        print('writing done!')

def spt_read_from_hdf5(filename):
    with h5py.File(filename, mode='r') as hdf:
        F = np.empty(shape=(len(hdf.keys()), len(hdf['F1'])))
        for j in range(len(hdf.keys())):
            F[j] = hdf['F{}'.format(j + 1)]
        # print('reading done!')
    return F

def SPT_final(F, a):
    for j in range(len(F)):
        F[j] *= (a ** (j + 1))
    for i in range(1, len(F)):
        F[i] += F[i - 1]
    return F
