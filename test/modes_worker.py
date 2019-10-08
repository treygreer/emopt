import emopt
import numpy as np
import matplotlib.pyplot as plt
from emopt.misc import NOT_PARALLEL, run_on_master


class Args(object):
    pass
args = Args()
args.symmetric = False

dxyz = 0.01
dx = dxyz # grid spacing along x
dy = dxyz # grid spacing along y
dz = dxyz # grid spacing along z
port_offset = 1.0 # distance in x from edge to i/o port
X = 5.0   # simulation size along x
Y = 2.0   # simulation size along y
if args.symmetric: Y=Y/2
Z = 1.5   # simulation size along z
w_wg = 0.5
h_si = 0.2
wavelength = 1.55

def make_eps():
    # background
    background_poly = emopt.grid_cuda.PolyMat([-dx, X+dx, X+dx, -dx],
                                              [-dy, -dy,  Y+dy, Y+dy],
                                              material_value=1.444**2)

    # mmi
    if args.symmetric:
        waveguide_poly = emopt.grid_cuda.PolyMat([-dx,  X+dx, X+dx, -dx],
                                                 [-dy, -dy,  w_wg/2, w_wg/2],
                                                 material_value = 3.45**2)
    else:
        waveguide_poly = emopt.grid_cuda.PolyMat([-dx,  X+dx, X+dx, -dx],
                                                 [Y/2-w_wg/2, Y/2-w_wg/2,  Y/2+w_wg/2, Y/2+w_wg/2],
                                                 material_value = 3.45**2)


    eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                               [dx, dy, dz],
                                               [[background_poly, -dz, Z+dz],
                                                [waveguide_poly, Z/2-h_si/2, Z/2+h_si/2]])
    return eps

def make_mu():
    return emopt.grid_cuda.ConstantMaterial3D(1.0)

#####################################################################################
# Setup the sources
#####################################################################################
# We excite the system by injecting the fundamental mode of the input waveguide
slc = emopt.misc.DomainCoordinates(port_offset, port_offset, # x
                                   0, Y,  # y
                                   0, Z,  # z
                                   dx, dy, dz)

eps = make_eps()
mu = make_mu()
mode = emopt.modes.ModeFullVector(wavelength, eps, mu, slc, n0=3.45,
                                  neigs=4)

# The mode boundary conditions should match the simulation boundary conditions.
# Mode is in the y-z plane, so the boundary conditions are H0
mode.bc = 'HH' if args.symmetric else 'HH'
mode.build()
mode.solve()

# get the Yee grid fields, work in non-permuted space for now
MODE=0
Ex = mode.get_field(MODE, 'Ex', permute=False, squeeze=True)
Ey = mode.get_field(MODE, 'Ey', permute=False, squeeze=True)
Ez = mode.get_field(MODE, 'Ez', permute=False, squeeze=True)
Hx = mode.get_field(MODE, 'Hx', permute=False, squeeze=True)
Hy = mode.get_field(MODE, 'Hy', permute=False, squeeze=True)
Hz = mode.get_field(MODE, 'Hz', permute=False, squeeze=True)
neff = mode.neff[MODE] * mode._dir
