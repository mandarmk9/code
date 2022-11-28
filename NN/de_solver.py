#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff
from neurodiffeq.ode import Solver1D
from neurodiffeq.conditions import IVP
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.networks import FCNN
from neurodiffeq.networks import SinActv

# specify the ODE system and its parameters
alpha, beta, delta, gamma = 1, 1, 1, 1
lotka_volterra = lambda u, v, t : [ diff(u, t) - (alpha*u  - beta*u*v),
                                    diff(v, t) - (delta*u*v - gamma*v), ]

# specify the initial conditions
init_vals_lv = [
    IVP(t_0=0.0, u_0=1.5),  # 1.5 is the value of u at t_0 = 0.0
    IVP(t_0=0.0, u_0=1.0),  # 1.0 is the value of v at t_0 = 0.0
]

# specify the network to be used to approximate each dependent variable
# the input units and output units default to 1 for FCNN
nets_lv = [
    FCNN(n_input_units=1, n_output_units=1, hidden_units=(64, 64), actv=SinActv),
    FCNN(n_input_units=1, n_output_units=1, hidden_units=(64, 64), actv=SinActv)
]

# Let's create a monitor first
monitor = Monitor1D(t_min=0.0, t_max=12.0, check_every=100)
# ... and turn it into a Callback instance
monitor_callback = monitor.to_callback()

# Instantiate a solver instance
solver = Solver1D(
    ode_system=lotka_volterra,
    conditions=init_vals_lv,
    t_min=0.1,
    t_max=12.0,
    nets=nets_lv,
)

# Fit the solver (i.e., train the neural networks)
solver.fit(max_epochs=1000, callbacks=[monitor_callback])

# Get the solution
solution_lv = solver.get_solution()

ts = np.linspace(0, 12, 100)

# ANN-based solution
prey_net, pred_net = solution_lv(ts, to_numpy=True)

# numerical solution
from scipy.integrate import odeint
def dPdt(P, t):
    return [P[0]*alpha - beta*P[0]*P[1], delta*P[0]*P[1] - gamma*P[1]]
P0 = [1.5, 1.0]
Ps = odeint(dPdt, P0, ts)
prey_num = Ps[:,0]
pred_num = Ps[:,1]

fig = plt.figure(figsize=(12, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts, prey_net, label='ANN-based solution of prey')
ax1.plot(ts, prey_num, '.', label='numerical solution of prey')
ax1.plot(ts, pred_net, label='ANN-based solution of predator')
ax1.plot(ts, pred_num, '.', label='numerical solution of predator')
ax1.set_ylabel('population')
ax1.set_xlabel('t')
ax1.set_title('Comparing solutions')
ax1.legend()

ax2.set_title('Error of ANN solution from numerical solution')
ax2.plot(ts, prey_net-prey_num, label='error in prey number')
ax2.plot(ts, pred_net-pred_num, label='error in predator number')
ax2.set_ylabel('populator')
ax2.set_xlabel('t')
ax2.legend()
plt.show()
