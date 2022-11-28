#!/usr/bin/env python3

from sympy import *
import sympy.core.numbers as num
import numpy as np

H0, x, y, a = symbols('H0 x y a')
# G = Function('G')(x, y)

G = (2 / (5 * H0**2)) * ((y**(5/2) - x**(5/2)) / (x**(3/2) * y))
I0 = integrate(simplify(y * G.diff(x)), (y, 0, x))
# I0 = simplify((3 * H0**2 * G) + (2 * H0**2 * x**2 * G.diff(x)))
# I1 = simplify(I0 * 5 * H0**2 * (y**2))
# I2 = simplify(integrate(I1 / y, (y, 0, x)))
# G1 = G.subs([(x, a), (y, x)])
# I3 = simplify(I2 * G1 / x)
# D3 = simplify(integrate(I3, (x, 0, a)))

# D0 = G1 * x**2 * H0**2
# D = 4 * integrate(D0, (x, 0, a))
# D4 = D3 - D
pprint(I0)
