#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import simpson

x = np.arange(0, 10.00001, 1)
y = x**3


x_new = np.arange(0, 10, 0.001)

lin_func = interp1d(x, y, kind='linear')
lin_interp = lin_func(x_new)

plt.scatter(x, y, s=10, c='k', label='data')
plt.plot(x_new, lin_interp, c='b', lw=2, label='linear interpolation')


plt.legend()
plt.savefig('../plots/sch_hfix_run19/num_int.png', bbox_inches='tight', dpi=120)
