#!/usr/bin/env python3
"""Independent lab-frame ODE for the ideal-field alpha magnet.

Used by test_alpha_magnet.py as a second check of the closed-form map.
The midplane orbit is x'' from dphi/ds = -k x, and the vertical variational
equation is y'' = k cos(phi) y, integrated with scipy.

Author: Eremey Valetov
"""
import math

import numpy as np
from scipy.integrate import solve_ivp

MEC2 = 0.51099895        # MeV
C_MM = 299.792458        # MeV/c per T*m
THETA_ALPHA = math.radians(40.70991)


def ode_arm(E_MeV, g):
    """(s_alpha, 6x6-relevant blocks, gamma, k) from a lab-frame integration."""
    gamma = 1.0 + E_MeV / MEC2
    bg = math.sqrt(gamma * gamma - 1.0)
    k = g / (bg * MEC2 / C_MM)
    phi_in = math.pi / 2.0 + THETA_ALPHA

    def rhs(s, u):
        x, _z, phi, y1, p1, y2, p2 = u
        c = math.cos(phi)
        return [math.sin(phi), c, -k * x, p1, k * c * y1, p2, k * c * y2]

    def edge(s, u):
        return u[0]
    edge.terminal, edge.direction = True, -1

    scale = 0.19165 * math.sqrt(bg / g)
    sol = solve_ivp(rhs, (1e-12, 6 * scale), [0, 0, phi_in, 1, 0, 0, 1],
                    events=edge, rtol=1e-12, atol=1e-14,
                    max_step=scale / 5000.0, dense_output=True)
    s_a = sol.t_events[0][0]
    u = sol.sol(s_a)
    My = np.array([[u[3], u[5]], [u[4], u[6]]])
    M = np.eye(6)
    M[0, 0], M[0, 1], M[1, 1] = -1.0, -0.5 * s_a, -1.0
    M[2:4, 2:4] = My
    M[4, 5] = s_a * (1.0 / gamma ** 2 - 0.5) * gamma ** 2 / (gamma + 1.0) ** 2
    return s_a, M, gamma, k
