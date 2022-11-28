#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dt = 0.001
t = np.arange(0, 1, dt)
p = 1
q = 0

def eq(r, t, p, q):
    y, z = r
    return [z, -((p*z) + q)]

z0 = [0, 1]
sol = odeint(eq, z0, t, args=(p, q))
sol_num_y = sol[:, 0]
sol_num_z = sol[:, 1]

sol_an_y = 1 - np.exp(-t)
sol_an_z = np.exp(-t)


fig, ax = plt.subplots()
# ax.plot(t, sol_an_y, lw=2, c='k', label='y(t): an')
# ax.plot(t, sol_num_y, lw=2, ls='dashed', c='b', label='y(t): num')

ax.plot(t, sol_an_z, lw=2, c='k', label='z(t): an')
ax.plot(t, sol_num_z, lw=2, ls='dashed', c='b', label='z(t): num')

ax.set_xlabel('t')

plt.legend()
plt.show()
