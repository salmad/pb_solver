from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


@dataclass
class Mesh:
    d: float = 0.05
    dr: float = 0.01  # in sigma
    R1: float = 5.0
    R2: float = 10.0
    nx: int = 1000


@dataclass
class Solution:
    # solution properties

    # Surface properties
    # dimensionless surface charge
    rho1: float  # = -0.35
    rho2: float  # = -0.35

    # dipole
    c_p: float  # = 1.0  # normalised by 2c0
    p: float  # = 0.1


@dataclass
class System(Mesh, Solution):
    nodes: int = 1000

    @property
    def Lx(self):
        return self.R2 + 10

    def calc_dy(self, r, Y):
        return fsub(r, Y, system=self)

    @property
    def y_guess(self):
        rho1 = self.rho1
        rho2 = self.rho2
        R1 = self.R1
        R2 = self.R2

        y = pot_DH(self.mesh_r, rho2, R2)
        dy = dpot_DH(self.mesh_r, rho2, R2)
        # ~ v = phi1*np.exp(-x)
        # ~ dv = phi1*np.exp(-x+H)

        Y = np.array([y, dy])

        return Y

    @property
    def dy_guess(self):
        d_Y = self.calc_dy(self.mesh_r, self.y_guess)
        return d_Y

    def set_bc(self, Y_a, Y_b):
        """The boundary conditions."""
        return gsub(Y_a, Y_b)

    @property
    def mesh_r(self):
        return np.linspace(self.d, self.Lx, self.nx)

    # @property
    # TODO : decide if needed to compute
    # def phis(self):
    #     y[(np.abs(R2 - r)).argmin()],

    @property
    def pb_sln(self):
        a = solve_bvp(fun=self.calc_dy, bc=self.set_bc, x=self.mesh_r,
                      y=self.y_guess, max_nodes=self.nodes, bc_tol=1e-3,
                      tol=1e-3)

        return a

    def sln_df(self, a, verbose=False):
        df = pd.DataFrame()
        df['x'] = a.x
        df['phi'] = a.y[0]
        df['dphi'] = a.y[1]
        df['c_cat'] = np.exp(-a.y[0])
        df['c_an'] = np.exp(a.y[0])
        df['phis'] = a.y[0][(np.abs(self.R2 - self.mesh_r)).argmin()]
        df['dphis'] = a.y[1][(np.abs(self.R2 - self.mesh_r)).argmin()]
        df['rms'] = np.insert(a.rms_residuals, 0, np.NaN)
        if verbose:
            df['yp_0'] = a.yp[0]
            df['yp_1'] = a.yp[1]
        return df

    def __post_init__(self):
        self.checker()

    def checker(self):
        assert self.rho1 == self.rho2
        assert self.R2 >= self.R1
        assert self.R2 <= self.Lx
        a = self.pb_sln
        print(a.rms_residuals.max(), a.rms_residuals.sum())
        assert a.rms_residuals.max() < 0.25
        assert a.rms_residuals.sum() < 1
        assert np.abs(a.y[1][0]) < 0.01
        assert np.abs(a.y[1][-1]) < 0.01
        assert np.abs(a.y[0][-1]) < 0.01
        return True

    def print_params(self):
        print(" *** Printing system parameters ***")
        self.checker()
        print(f"  Microgel dimensioless radius  kR1 =  {self.R1}")
        print("  Microgel dimensioless radius  kR2 = ", self.R2)
        print("  Microgel dimensioless charge rho1 = ", self.rho1)
        print("  Microgel dimensioless charge rho2 = ", self.rho2)
        print("  Mesh size                     dr  = ", self.dr)
        print("  Avoiding singularity at r = 0, d  = ", self.d)
        print(" ***  ***  *** *** *** *** *** ***    ")
        return 0

    @classmethod
    def __setitem__(cls, key, value):
        cls.checker()
        setattr(cls, key, value)


def sigmoid(x, d=0.01):
    return 1 - 1 / (1 + np.exp(-x / d))


# homogeneous
def delta(x_mesh, start, end, use_sigmoid=True, width=0.01):
    """

    :param x_mesh:
    :param start:
    :param end:
    :return:
    """

    d = end - start
    x0 = start
    r = x_mesh

    if use_sigmoid:
        return sigmoid(x_mesh - end, width) - sigmoid(x_mesh - start, width)
    else:
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
    cd = - (np.sinh(y) - rho1 * delta(r, start=0, end=R1) - rho2 * delta(r, end=R2, start=R1))
    # cd = interp1d(r, cd, kind='cubic')
    return cd


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
    # d_y = np.where(np.abs(dy)<1e-6,0 ,dy)
    d_y = dy
    d_dy = - charge_density(r, y, system=system)
    d_dy /= epsilon(dy, system=system) + depsilon(dy, system=system) * dy
    # d_dy = np.where(np.abs(d_dy) < 1e-6, 0, d_dy)
    return np.array([d_y, d_dy])


def gsub(Y_a, Y_b):
    """The boundary conditions."""
    y_a, dy_a = Y_a
    y_b, dy_b = Y_b
    return np.array([dy_a, y_b])


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
    gel = System(rho1=-0.2, rho2=-0.2, c_p=1, p=.1)
    print(gel.print_params())

    a = gel.pb_sln

    df = gel.sln_df(a)
    print(a.rms_residuals.max(), a.rms_residuals.sum())

    cd = charge_density(gel.mesh_r, a.y[0] * 0, gel)
    plt.plot(a.x, cd)
    plt.plot(gel.mesh_r, gel.y_guess[0])
    plt.plot(a.x, a.y[0])
    plt.show()

    # plt.plot(df['x'], df['phi'])
    #
    # plt.plot(df['x'], df['dphi'])
    # plt.plot(df['x'], df['c_an'])
    # plt.plot(df['x'], df['c_cat'])
