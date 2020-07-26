# import scikits.bvp1lg as bvp
from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    d: float = 0.005
    dr: float = 0.001  # in sigma
    R1: float = 1.0
    R2: float = 3.0
    nx: float = 32000


@dataclass
class Solution:
    ######
    ###### solution properties

    ###### Surface properties
    # dimensionless surface charge
    rho1: float = -0.35
    rho2: float = -0.35

    # dipole
    c_p: float = 1.0  # normalised by 2c0
    p: float = 0.1


# homogeneous
def delta(r, d, x0):
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


def Charge_density(r:float, y, mesh:Mesh, solution:Solution):
    global d, rho1, rho2, R1, R2
	d = Mesh.d
	rho1 =
    return - (np.sinh(y) - rho1 * delta(r, R1, d) - rho2 * delta(r, R2 - R1, R1))


def Langevin(x:float):
    return 1. / np.tanh(x) - 1. / x


def G(x):
    return np.sinh(x) * Langevin(x) / x ** 2


# E is grad phi
# epsilon as a function of electric field
def epsilon(E):
    global p, c_p
    return 1 + c_p * p ** 2 * G(p * E)


# proportionality between field and epsilon

def depsilon(E):
    global p, c_p
    return c_p * np.sinh(p * E) * (-3 * Langevin(p * E) + p * E) / E ** 3


def potential(R1=1.0, R2=3.0):
    # ~ ,b1,b2,q1,q2,mu1,mu2

    global dr
    global d, rho1, rho2
    global nx

    def fsub(r, Y):
        """The equations, in the form: Y' = f(x, Y)."""
        y, dy = Y
        d_y = dy;
        # ~ d_v=dv
        d_dy = - Charge_density(r, y) / (epsilon(dy) + depsilon(dy) * dy)
        return np.array([d_y, d_dy])

    def gsub(Y):
        """The boundary conditions."""
        y, dy = Y
        return np.array([dy[0], dy[1]])

    def guess(r):
        y = pot_DH(r, rho2, R2)
        dy = dpot_DH(r, rho2, R2)
        # ~ v = phi1*np.exp(-x)
        # ~ dv = phi1*np.exp(-x+H)

        Y = np.array([y, dy])
        dm = fsub(r, Y)
        return Y, dm

    L = R2 + 10.0  # channel width

    dr = L / nx
    # 2 times zero as we have two bcs at x=0 :
    boundary_points = [d, L]
    tol = 1e-5 * np.ones_like(boundary_points)
    # print tol
    degrees = [1, 1]
    solution = bvp.colnew.solve(boundary_points, degrees, fsub, gsub, initial_guess=guess, tolerances=tol,
                                vectorized=True, maximum_mesh_size=256000)

    r = np.linspace(d, L, nx)
    y, dy = solution(r).transpose()

    return r, y, y[0], y[(np.abs(R2 - r)).argmin()], dy


def print_params():
    print(" *** Printing system parameters ***")
    print(f"  Microgel dimensioless radius  kR1 ={R1}"
    print("  Microgel dimensioless radius  kR2 = ", R2)
    print("  Microgel dimensioless charge rho1 = ", rho1)
    print("  Microgel dimensioless charge rho2 = ", rho2)
    print("  Mesh size                     dr  = ", dr)
    print("  Avoiding singularity at r = 0, d  = ", d)
    print(" ***  ***  *** *** *** *** *** ***    ")
    return 0
