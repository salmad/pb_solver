# import scikits.bvp1lg as bvp
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp


@dataclass
class Mesh:
    d: float = 0.005
    dr: float = 0.001  # in sigma
    R1: float = 1.0
    R2: float = 3.0
    nx: int = 32000


@dataclass
class Solution:
    # solution properties

    # Surface properties
    # dimensionless surface charge
    rho1: float = -0.35
    rho2: float = -0.35

    # dipole
    c_p: float = 1.0  # normalised by 2c0
    p: float = 0.1


@dataclass
class System(Mesh, Solution):

    @property
    def Lx(self):
        return self.R2 + 20

    def dy(self, r, Y):
        return fsub(r, Y, system=self)

    def y_dy_guess(self, r):
        return guess(r, system=self)

    def bc(self, Y_a, Y_b):
        """The boundary conditions."""
        return gsub(Y_a, Y_b)

    def print_params(self):
        print(" *** Printing system parameters ***")
        print(f"  Microgel dimensioless radius  kR1 = {self.R1}")
        print("  Microgel dimensioless radius  kR2 = ", self.R2)
        print("  Microgel dimensioless charge rho1 = ", self.rho1)
        print("  Microgel dimensioless charge rho2 = ", self.rho2)
        print("  Mesh size                     dr  = ", self.dr)
        print("  Avoiding singularity at r = 0, d  = ", self.d)
        print(" ***  ***  *** *** *** *** *** ***    ")
        return 0


# homogeneous
def delta(x_mesh, start, end):
    """

    :param x_mesh:
    :param start:
    :param end:
    :return:
    """

    d = end - start
    x0 = start
    r = x_mesh

    return 0.5 * (np.sign(r - x0) + 1) - 0.5 * (np.sign(-d + r - x0) + 1.0)


def phis_DH(rho, R):
    return -rho / R * np.exp(-R) * (np.sinh(R) - R * np.cosh(R))


def phi0_DH(rho, R):
    return -rho * (1 + R) * np.exp(-R) + rho


def pot_DH(r, rho, R):
    return np.piecewise(r, [np.abs(r) < R, np.abs(r) >= R], [lambda r: np.sinh(r) / r * (phi0_DH(rho, R) - rho) + rho,
                                                             lambda r: R * phis_DH(rho, R) * np.exp(R - r) / r])


def dpot_DH(r, rho, R):
    return np.piecewise(r, [np.abs(r) < R, np.abs(r) >= R],
                        [lambda r: -rho * (1 + R) * np.exp(-R) * (r * np.cosh(r) - np.sinh(r)) / r ** 2,
                         lambda r: -rho * (1 + r) * np.exp(-r) * (R * np.cosh(R) - np.sinh(R)) / r ** 2])


def charge_density(r, y, system: System):
    d = system.d
    rho1 = system.rho1
    rho2 = system.rho2
    R1 = system.R1
    R2 = system.R2
    return - (np.sinh(y) - rho1 * delta(r, start=d, end=R1 + d) - rho2 * delta(r, end=R2 + d, start=R1 + d))


def langevin(x):
    return 1. / np.tanh(x) - 1. / (x)


def G(x):
    return np.sinh(x) * langevin(x) / x ** 2


# E is grad phi
# epsilon as a function of electric field
def epsilon(E, system: System):
    p = system.p
    c_p = system.c_p
    return 1 + c_p * p ** 2 * G(p * E)


# proportionality between field and epsilon

def depsilon(E, system: System):
    p = system.p
    c_p = system.c_p
    return c_p * np.sinh(p * E) * (-3 * langevin(p * E) + p * E) / E ** 3


def fsub(r, Y, system: System):
    """The equations, in the form: Y' = f(x, Y)."""
    y, dy = Y
    d_y = dy
    d_dy = - charge_density(r, y, system=system)
    d_dy /= epsilon(dy, system=system) + depsilon(dy, system=system) * dy
    return np.array([d_y, d_dy])


def gsub(Y_a, Y_b):
    """The boundary conditions."""
    y_a, dy_a = Y_a
    y_b, dy_b = Y_b
    return np.array([dy_a , y_b])


def guess(r, system: System):
    rho1 = system.rho1
    rho2 = system.rho2
    R1 = system.R1
    R2 = system.R2

    y = pot_DH(r, rho2, R2)
    dy = dpot_DH(r, rho2, R2)
    # ~ v = phi1*np.exp(-x)
    # ~ dv = phi1*np.exp(-x+H)

    Y = np.array([y, dy])
    d_Y = fsub(r, Y, system=system)
    return Y, d_Y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    system = System(R1=5, R2=10, nx=1000, d=0.01)
    print(system.print_params())

    r = np.linspace(system.d, system.Lx, system.nx)
    Y, d_Y = guess(r, system=system)

    a = solve_bvp(fun=system.dy, bc=system.bc, x=r, y=Y, max_nodes=10000, bc_tol=1e-8,tol=1e-8)
    a.message
    a.success
    a.x
    plt.plot(r, d_Y[0])
    plt.plot(a.x, a.y[1])
    print(a.y[1][1], a.y[1][-1])
    print(a.y[0][0], a.y[0][-1])
    print(a.rms_residuals.max())
    df = pd.DataFrame()
    df['x'] = a.x
    df['y'] = a.y[0]
    df['dy'] = a.y[1]
    df['rms'] = np.insert(a.rms_residuals,0,-1)
    df.sort_values('rms', ascending=False)
