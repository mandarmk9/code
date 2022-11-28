#!/usr/bin/env python3
import time
import numpy as np
from numpy import sqrt as np_sqrt
from math import sqrt

#define the array, number of iterations for testing
N = 1000000000
N_iter = 1
# A = np.random.randint(1, 1e6, 1)
A = np.random.uniform(0, 1, N)


#function
def math_sqrt(A, method='numpy'):
    t0 = time.time()
    if method == 'numpy':
        sq = np_sqrt(A)
    elif method == 'pow':
        sq = A**(.5)
    elif method == 'math':
        sq = sqrt(A)

    t1 = time.time()
    del_t = t1-t0
    return del_t

#timekeeping
methods = ['numpy', 'pow']#, 'math']
for method in methods:
    t = np.zeros(N_iter)
    for j in range(N_iter):
        t[j] = math_sqrt(A, method)
    print('{} takes {}s for {} iterations'.format(method, (np.sum(t)), N_iter))
