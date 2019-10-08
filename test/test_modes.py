import argparse
import emopt
import numpy as np
import matplotlib.pyplot as plt
from emopt.misc import NOT_PARALLEL, run_on_master
import ipyparallel as ipp

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--symmetric', action='store_true')
args=parser.parse_args()

normalize = 'maxHz'

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

#####################################################################################
# Generate the mode on the mpi farm
#####################################################################################
client = ipp.Client(profile='mpi')
view_all = client[:] # run on all nodes
view_all.block = True
view_all.run('modes_worker.py')

#####################################################################################
# Get the fields from node 0
#####################################################################################
view_0 = client[0]
view_0.block = True
Ex = view_0['Ex']
Ey = view_0['Ey']
Ez = view_0['Ez']
Hx = view_0['Hx']
Hy = view_0['Hy']
Hz = view_0['Hz']

eps = make_eps()
mu = make_mu()

if normalize == 'power':
    # normalize power to ~1.0
    S = 0.5*dx*dy*np.sum(Ex*np.conj(Hy)-Ey*np.conj(Hx))
    if args.symmetric:
            S = S*2.0
    Ex = Ex/np.sqrt(S); Ey = Ey/np.sqrt(S); Ez = Ez/np.sqrt(S)
    Hx = Hx/np.sqrt(S); Hy = Hy/np.sqrt(S); Hz = Hz/np.sqrt(S)

elif normalize == 'maxHz':
    # normlize maximum abs(Hz) to 1.0
    maxHz = np.max(np.abs(Hz))
    Ex = Ex/maxHz; Ey = Ey/maxHz; Ez = Ez/maxHz;
    Hx = Hx/maxHz; Hy = Hy/maxHz; Hz = Hz/maxHz;


# get the material values at the Yee grid points  TODO:  verify these offsets for ndir in {x,y,z}
eps_x = eps.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.5, ioff=0.0, arr=None)[:,:,0]
eps_y = eps.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.0, ioff=0.5, arr=None)[:,:,0]
eps_z = eps.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.0, ioff=0.0, arr=None)[:,:,0]
mu_x  =  mu.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.0, ioff=0.5, arr=None)[:,:,0]
mu_y  =  mu.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.5, ioff=0.0, arr=None)[:,:,0]
mu_z  =  mu.get_values(slc.k1, slc.k2, slc.j1, slc.j2, slc.i1, slc.i2, koff=0.0, joff=0.5, ioff=0.5, arr=None)[:,:,0]

assert(eps_x.shape == Ex.shape)

odx = view_0['mode.R/mode.dx'] # non-dimensionalize
ody = view_0['mode.R/mode.dy'] # non-dimensionalize
neff = view_0['mode.neff[0]']

atol = 1e-7
    
lhs=0
rhs=0

def test_Ex():
    global lhs, rhs
    dHzdy = np.diff(Hz, axis=0, prepend=Hz[[0],:]) * ody
    lhs = dHzdy + 1j*eps_x*Ex
    rhs = 1j*neff*Hy
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

def test_Ey():
    global lhs, rhs
    dHzdx = np.diff(Hz, axis=1, prepend=Hz[:,[0]]) * odx
    lhs = -dHzdx + 1j*eps_y*Ey
    rhs = -1j*neff*Hx
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

def test_Ez():
    global lhs, rhs
    dHydx = np.diff(Hy, axis=1, prepend=Hy[:,[0]]) * odx
    dHxdy = np.diff(Hx, axis=0, prepend=Hx[[0],:]) * ody
    lhs = dHydx - dHxdy + 1j*eps_z*Ez
    rhs = np.zeros_like(lhs)
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

def test_Hx():
    global lhs, rhs
    dEzdy = np.diff(Ez, axis=0, append=Ez[[-1],:]) * ody
    lhs = dEzdy - 1j*mu_x*Hx
    rhs = 1j*neff*Ey
    lhs = lhs[:-1,:]; rhs = rhs[:-1,:]  # skip last row
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

def test_Hy():
    global lhs, rhs
    dEzdx = np.diff(Ez, axis=1, append=Ez[:,[-1]]) * odx
    lhs = -dEzdx - 1j*mu_y*Hy
    rhs = -1j*neff*Ex
    lhs = lhs[:,:-1]; rhs = rhs[:,:-1]   # skip last column
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

def test_Hz():
    global lhs, rhs
    dEydx = np.diff(Ey, axis=1, append=Ey[:,[-1]]) * odx
    dExdy = np.diff(Ex, axis=0, append=Ex[[-1],:]) * ody
    lhs = dEydx - dExdy - 1j*mu_z*Hz
    rhs = np.zeros_like(lhs)
    lhs = lhs[:-1,:-1]; rhs = rhs[:-1,:-1]  # skip last row and column
    np.testing.assert_allclose(lhs, rhs, rtol=0,atol=atol)

