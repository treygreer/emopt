#!/bin/python3
import emopt
from emopt.misc import DomainCoordinates, NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

class Arc(object):
    def __init__(self, x_start, y_start, radius, width, initial_angle, final_angle, num_per_pi=100):
        N = num_per_pi * abs(final_angle-initial_angle) / pi
        theta = np.linspace(initial_angle, final_angle, N)
        x0 = radius * np.cos(initial_angle)
        y0 = radius * np.sin(initial_angle)
        x1 = radius * np.cos(final_angle)
        y1 = radius * np.sin(final_angle)
        x_outer = (radius+width/2) * np.cos(theta)
        y_outer = (radius+width/2) * np.sin(theta)
        x_inner = (radius-width/2) * np.cos(theta[::-1])
        y_inner = (radius-width/2) * np.sin(theta[::-1])

        self.x = np.append(x_outer, x_inner) - x0 + x_start
        self.y = np.append(y_outer, y_inner) - y0 + y_start

        self.x_start = x_start
        self.y_start = y_start

        self.x_end = x_start - x0 + x1
        self.y_end = y_start - y0 + y1

####################################################################################
# Simulation parameters
####################################################################################
ring_diameter = 10
guide_width = 0.5
gap_width = 0.2
wavelength = 1.55
h_si = 0.22

dxyz = 0.04
dx = dxyz
dy = dxyz
dz = dxyz
w_pml = dxyz * 15

ring_x = ring_diameter/2
ring_y = w_pml + 4*guide_width + gap_width + guide_width/2 + ring_diameter/2
X = ring_x + ring_diameter/2 + 4*guide_width + w_pml
Y = ring_y + 2*dy + w_pml
Z = 2.5

bus_y = ring_y - ring_diameter/2 - guide_width - gap_width

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
                                        material_value=1.444**2)
bus_pm = emopt.grid_cuda.PolyMat(
    [-X/2,              3*X/2,             3*X/2,             -X/2],
    [bus_y-guide_width/2, bus_y-guide_width/2, bus_y+guide_width/2, bus_y+guide_width/2],
    material_value=3.45**2)

ring = Arc(ring_x-ring_diameter/2, ring_y,
           radius=ring_diameter/2, width=guide_width,
           initial_angle=-pi, final_angle=pi/2)
ring_pm = emopt.grid_cuda.PolyMat(ring.x, ring.y, material_value=3.45**2)

eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                           [dx, dy, dz],
                                           [[background_pm, -Z, 2*Z],
                                            [bus_pm, Z/2-h_si/2, Z/2+h_si/2],
                                            [ring_pm, Z/2-h_si/2, Z/2+h_si/2]])

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
xmin = ring_x + ring_diameter/2 - guide_width*4
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

field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2, Z/2,
                                  dx, dy, dz)
Ey = sim.get_field_interp('Ey', domain=field_monitor, squeeze=True)
Hz = sim.get_field_interp('Hz', domain=field_monitor, squeeze=True)

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    eps_arr = eps.get_values_in(field_monitor, squeeze=True)

    f = plt.figure(1)

    ax = f.add_subplot(221)
    vmax = np.max(np.real(Ey))
    ax.imshow(np.real(Ey), extent=[0,X,0,Y], origin='lower',
              vmin=-vmax, vmax=vmax, cmap='seismic')

    ax = f.add_subplot(222)
    vmax = np.max(np.real(Hz))
    ax.imshow(np.real(Hz), extent=[0,X,0,Y], origin='lower',
              vmin=-vmax, vmax=vmax, cmap='seismic')


    ax = f.add_subplot(223)
    ax.imshow(np.real(eps_arr), extent=[0,X,0,Y], origin='lower')

    plt.show()

