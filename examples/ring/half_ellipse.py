#!/bin/python3
import argparse
import emopt
from emopt.misc import DomainCoordinates, NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('-cg', '--coupler_gap', type=float, default=0.3, help='coupler gap [microns]')
parser.add_argument('-bgw', '--bus_guide_width', type=float, default=0.5, help='bus guide width [microns]')
parser.add_argument('-rgw', '--ring_guide_width', type=float, default=0.5, help='ring guide width [microns]')
parser.add_argument('-rh', '--radius_hor', type=float, default=6, help='horizontal radius [microns]')
parser.add_argument('-rv', '--radius_ver', type=float, default=3, help='vertical radius [microns]')

parser.add_argument('-g', '--grid', type=float, default=0.04, help='grid [microns]')
args=parser.parse_args()

####################################################################################
# Simulation parameters
####################################################################################
coupler_gap = args.coupler_gap
bus_guide_width = args.bus_guide_width
ring_rih = args.radius_hor - args.ring_guide_width/2
ring_riv = args.radius_ver - args.ring_guide_width/2
ring_roh = args.radius_hor + args.ring_guide_width/2
ring_rov = args.radius_ver + args.ring_guide_width/2
grid = args.grid
wavelength = 1.55
h_si = 0.22

fig_title = f'gap={args.coupler_gap} bus width={args.bus_guide_width} rih={ring_rih} riv={ring_riv} roh={ring_roh} rov={ring_rov} grid={grid}'
print(fig_title)

eps_oxide = 1.444**2
eps_si = 3.45**2

dx = grid
dy = grid
dz = grid
w_pml = grid * 15

# vertical distance from center of ring to centerline of bus
coupler_radius = ring_rov + coupler_gap + bus_guide_width/2

# center coordinates of ring
ring_x = ring_rih
ring_y = w_pml + 4*bus_guide_width + coupler_radius
bus_y = ring_y - coupler_radius   # centerline of bus

# simulation domain
X = ring_x + ring_roh + 4*bus_guide_width + w_pml
Y = ring_y + 2*dy + w_pml
Z = 2.5

class Ellipse(object):
    def __init__(self, x, y, rh, rv, N=200):
        theta = np.linspace(0, 2*pi, N, endpoint=False)
        self.x = x + rh * np.cos(theta)
        self.y = y + rv * np.sin(theta)

#####################################################################################
# Setup simulation
#####################################################################################
sim = emopt.fdtd_cuda.FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5, nconv=10000)
sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]

X = sim.X
Y = sim.Y
Z = sim.Z

#####################################################################################
# Define the geometry/materials
#####################################################################################
background_pm = emopt.grid_cuda.PolyMat([-X/2, 3*X/2, 3*X/2, -X/2],
                                        [-Y/2, -Y/2,  3*Y/2, 3*Y/2],
                                        material_value=eps_oxide)

slab_pm = emopt.grid_cuda.PolyMat([-X/2, 3*X/2, 3*X/2, -X/2],
                                        [-Y/2, -Y/2,  3*Y/2, 3*Y/2],
                                        material_value=eps_si)

ring_outer = Ellipse(ring_x, ring_y, ring_roh, ring_rov)
ring_inner = Ellipse(ring_x, ring_y, ring_rih, ring_riv)
ring_outer_pm = emopt.grid_cuda.PolyMat(ring_outer.x, ring_outer.y, material_value=eps_si)
ring_inner_pm = emopt.grid_cuda.PolyMat(ring_inner.x, ring_inner.y, material_value=eps_oxide)

bus_poly_x = [-X, 2*X, 2*X, -X]
bus_poly_y = [bus_y-bus_guide_width/2, bus_y-bus_guide_width/2, bus_y+bus_guide_width/2, bus_y+bus_guide_width/2]
bus_pm = emopt.grid_cuda.PolyMat(bus_poly_x, bus_poly_y, material_value=eps_si)

eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                           [dx, dy, dz],
                                           [[background_pm, -Z, 2*Z],
                                            [bus_pm, Z/2-h_si/2, Z/2+h_si/2],
                                            [ring_outer_pm, Z/2-h_si/2, Z/2+h_si/2],
                                            [ring_inner_pm, -Z, 2*Z]])

mu = emopt.grid_cuda.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the source
#####################################################################################
mode_x = w_pml + dx
mode_slice = emopt.misc.DomainCoordinates(mode_x, mode_x,
                                          w_pml + dy, bus_y + 4*bus_guide_width,
                                          w_pml, Z-w_pml,
                                          dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, mode_slice, n0=3.45,
                                   neigs=4)
mode.build()
mode.solve()
sim.set_sources(mode, mode_slice)


#####################################################################################
# Simulate and view results
#####################################################################################
sim.solve_forward()


zmin = w_pml + dz
zmax = Z - w_pml - dz
# calculate power transmitted through left bus
xmin = mode_x + 2*dx
ymin = w_pml + dy
ymax = bus_y + bus_guide_width*4
left_boundary = DomainCoordinates(xmin, xmin, ymin, ymax, zmin, zmax, dx, dy, dz)
Ey = sim.get_field_interp('Ey', left_boundary)
Ez = sim.get_field_interp('Ez', left_boundary)
Hy = sim.get_field_interp('Hy', left_boundary)
Hz = sim.get_field_interp('Hz', left_boundary)

if NOT_PARALLEL:
    Pleft = 0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
del Ey; del Ez; del Hy; del Hz

# calculate power transmitted through right bus
xmax = X - w_pml - dx
ymin = w_pml + dy
ymax = bus_y + bus_guide_width*4
right_boundary = DomainCoordinates(xmax, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
Ey = sim.get_field_interp('Ey', right_boundary)
Ez = sim.get_field_interp('Ez', right_boundary)
Hy = sim.get_field_interp('Hy', right_boundary)
Hz = sim.get_field_interp('Hz', right_boundary)

if NOT_PARALLEL:
    Pright = 0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
del Ey; del Ez; del Hy; del Hz

# calculate power transmitter through top boundary
pxmin = ring_x + ring_rih - bus_guide_width*4
xmax = X - w_pml - dx
top_boundary = DomainCoordinates(xmin, xmax, ring_y, ring_y, zmin, zmax, dx, dy, dz)
Ex = sim.get_field_interp('Ex', top_boundary)
Ez = sim.get_field_interp('Ez', top_boundary)
Hx = sim.get_field_interp('Hx', top_boundary)
Hz = sim.get_field_interp('Hz', top_boundary)

if(NOT_PARALLEL):
    Ptop = 0.5*dx*dz*np.sum(np.real(Ez*np.conj(Hx)-Ex*np.conj(Hz)))
del Ex; del Ez; del Hx; del Hz

Ptotal = sim.get_source_power()
print(f"coupler power fraction = {Ptop/Pleft*100:.3f}%")

field_monitor1 = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2+.005, Z/2+.005,
                                  dx, dy, dz)
Hz1 = sim.get_field_interp('Hz', domain=field_monitor1, squeeze=True)

field_monitor2 = emopt.misc.DomainCoordinates(ring_x, ring_x, 0, Y, 0, Z,
                                  dx, dy, dz)
Hz2 = sim.get_field_interp('Hz', domain=field_monitor2, squeeze=True)

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    eps1 = eps.get_values_in(field_monitor1, squeeze=True)
    extent1 = [0,X,0,Y]
    eps2 = eps.get_values_in(field_monitor2, squeeze=True)
    extent2 = [0,Y,0,Z]

    f = plt.figure(1)
    plt.clf()
    f.suptitle(fig_title)

    ax = f.add_subplot(221)
    vmax = np.max(abs(np.real(Hz1)))
    ax.imshow(np.real(Hz1), extent=extent1, origin='lower',
              vmin=-vmax, vmax=vmax, cmap='seismic')

    ax.contour(np.real(eps1), extent=extent1, levels=2,
                      colors='#666666', linewidths=0.1)
    ax.set(title='Hz, z = SOI middle')
    ax.set(xlabel='x [microns]')
    ax.set(ylabel='y [microns]')

    ax = f.add_subplot(222)
    ax.imshow(np.real(eps1), extent=extent1, origin='lower')
    ax.set(title='eps, z = SOI middle')
    ax.set(xlabel='x [microns]')
    ax.set(ylabel='y [microns]')

    ax = f.add_subplot(223)
    vmax = np.max(abs(np.real(Hz2)))
    ax.imshow(np.real(Hz2), extent=extent2, origin='lower',
              vmin=-vmax, vmax=vmax, cmap='seismic')
    ax.set(title='Hz, x = mid ring')
    ax.set(xlabel='y [microns]')
    ax.set(ylabel='z [microns]')

    ax.contour(np.real(eps2), extent=extent2, levels=2,
                      colors='#666666', linewidths=0.1)

    ax = f.add_subplot(224)
    ax.imshow(np.real(eps2), extent=extent2, origin='lower')
    ax.set(title='eps, x = mid ring')
    ax.set(xlabel='y [microns]')
    ax.set(ylabel='z [microns]')

    plt.show()

