#!/bin/python3
import argparse
import emopt
from emopt.misc import DomainCoordinates, NOT_PARALLEL
from euler_ring import EulerRing

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('-cg', '--coupler_gap', type=float, default=0.3, help='coupler gap [microns]')
parser.add_argument('-gw', '--guide_width', type=float, default=0.5, help='guide width [microns]')
parser.add_argument('-c', '--circumference', type=float, default=31.4, help='ring circumference [microns]')

parser.add_argument('-g', '--grid', type=float, default=0.04, help='grid [microns]')
args=parser.parse_args()

####################################################################################
# Simulation parameters
####################################################################################
coupler_gap = args.coupler_gap
guide_width = args.guide_width
ring_circumference = args.circumference
grid = args.grid
wavelength = 1.55
h_si = 0.22

fig_title = f'gap={coupler_gap} guide width={guide_width} circumference={ring_circumference} grid={grid}'
print(fig_title)

eps_oxide = 1.444**2
eps_si = 3.45**2

dx = grid
dy = grid
dz = grid
w_pml = grid * 15

ring = EulerRing(x=0, y=0, circumference=ring_circumference, num=400)
ring_rh = np.max(ring.x)
ring_rv = np.max(ring.y)
del ring
ring_scale_outer = (ring_rv + guide_width/2) / ring_rv
ring_scale_inner = (ring_rv - guide_width/2) / ring_rv
outer_ring = EulerRing(x=0, y=0, circumference = ring_scale_outer * ring_circumference, num=400)
inner_ring = EulerRing(x=0, y=0, circumference = ring_scale_inner * ring_circumference, num=400)

# vertical distance from center of ring to centerline of bus
coupler_radius = np.max(outer_ring.y) + coupler_gap + guide_width/2

# determine center coordinates of ring
ring_x = max(outer_ring.x)
ring_y = w_pml + 4*guide_width + coupler_radius
bus_y = ring_y - coupler_radius   # centerline of bus
# simulation domain
X = ring_x + max(outer_ring.x) + 4*guide_width + w_pml
Y = ring_y + 2*dy + w_pml
Z = 2.5

# move rings
outer_ring.x += ring_x
outer_ring.y += ring_y
inner_ring.x += ring_x
inner_ring.y += ring_y

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

outer_ring_pm = emopt.grid_cuda.PolyMat(outer_ring.x, outer_ring.y, material_value=eps_si)
inner_ring_pm = emopt.grid_cuda.PolyMat(inner_ring.x, inner_ring.y, material_value=eps_oxide)

bus_poly_x = [-X, 2*X, 2*X, -X]
bus_poly_y = [bus_y-guide_width/2, bus_y-guide_width/2, bus_y+guide_width/2, bus_y+guide_width/2]
bus_pm = emopt.grid_cuda.PolyMat(bus_poly_x, bus_poly_y, material_value=eps_si)

eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                           [dx, dy, dz],
                                           [[background_pm, -Z, 2*Z],
                                            [bus_pm, Z/2-h_si/2, Z/2+h_si/2],
                                            [outer_ring_pm, Z/2-h_si/2, Z/2+h_si/2],
                                            [inner_ring_pm, -Z, 2*Z]])

mu = emopt.grid_cuda.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the source
#####################################################################################
mode_x = w_pml + dx
mode_slice = emopt.misc.DomainCoordinates(mode_x, mode_x,
                                          w_pml + dy, bus_y + 4*guide_width,
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
ymax = bus_y + guide_width*4
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
ymax = bus_y + guide_width*4
right_boundary = DomainCoordinates(xmax, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
Ey = sim.get_field_interp('Ey', right_boundary)
Ez = sim.get_field_interp('Ez', right_boundary)
Hy = sim.get_field_interp('Hy', right_boundary)
Hz = sim.get_field_interp('Hz', right_boundary)

if NOT_PARALLEL:
    Pright = 0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
del Ey; del Ez; del Hy; del Hz

# calculate power transmitter through top boundary
pxmin = np.max(inner_ring.x) - guide_width*4
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

