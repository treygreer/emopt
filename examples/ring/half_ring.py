#!/bin/python3
import argparse
import emopt
from emopt.misc import DomainCoordinates, NOT_PARALLEL

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('-cg', '--coupler_gap', type=float, default=0.4, help='coupler gap [microns]')
parser.add_argument('-hs', '--h_slab', type=float, default=0.110, help='slab height [microns]')
parser.add_argument('-ca', '--coupler_angle', type=float, default=0.0, help='coupler angle [degrees]')
parser.add_argument('-rgw', '--ring_guide_width', type=float, default=0.5, help='ring guide width [microns]')
parser.add_argument('-bgw', '--bus_guide_width', type=float, default=0.5, help='bus guide width [microns]')
parser.add_argument('-g', '--grid', type=float, default=0.02, help='grid [microns]')
args=parser.parse_args()

####################################################################################
# Simulation parameters
####################################################################################
print(args.coupler_angle)
coupler_angle = args.coupler_angle * pi/180
h_slab = args.h_slab
coupler_gap = args.coupler_gap
bus_guide_width = args.bus_guide_width
ring_guide_width = args.ring_guide_width
grid = args.grid
ring_diameter = 10
entry_radius = 5
wavelength = 1.55
h_si = 0.22
fig_title = f'gap={coupler_gap} angle={round(coupler_angle*180/pi)} h_slab={h_slab} bus width={bus_guide_width} ring width={ring_guide_width} grid={grid}'
print(fig_title)

eps_oxide = 1.444**2
eps_si = 3.45**2

dx = grid
dy = grid
dz = grid
w_pml = grid * 15

# coupler radius from ring diameter and gap (radius measured to center of bus)
coupler_radius = ring_diameter/2 + ring_guide_width/2 + coupler_gap + bus_guide_width/2

# center coordinates of ring
#ring_x = w_pml + 2*bus_guide_width + (entry_radius + coupler_radius)*np.sin(coupler_angle/2)
ring_x = ring_diameter/2
ring_y = w_pml + 4*bus_guide_width + coupler_radius

# simulation domain
X = ring_x + ring_diameter/2 + 4*bus_guide_width + w_pml
Y = ring_y + 2*dy + w_pml
Z = 2.5

class Arc(object):
    def __init__(self, x_start, y_start, radius, width, initial_angle, final_angle, num_per_pi=100):
        N = max(10, round(num_per_pi * abs(final_angle-initial_angle) / pi))
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

class Segment(object):
    def __init__(self, x_start, y_start, x_end, y_end, width):
        perp_x = y_end - y_start
        perp_y = x_start - x_end
        norm = 1/np.sqrt(perp_x*perp_x + perp_y*perp_y)
        perp_x = perp_x * norm
        perp_y = perp_y * norm
        self.x = np.array([x_start + perp_x*width/2,
                           x_end   + perp_x*width/2,
                           x_end   - perp_x*width/2,
                           x_start - perp_x*width/2])
        self.y = np.array([y_start + perp_y*width/2,
                           y_end   + perp_y*width/2,
                           y_end   - perp_y*width/2,
                           y_start - perp_y*width/2])
        
        self.x_start = x_start
        self.y_start = y_start

        self.x_end = x_end
        self.y_end = y_end

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

ring = Arc(ring_x-ring_diameter/2, ring_y,
           radius=ring_diameter/2, width=ring_guide_width,
           initial_angle=-pi, final_angle=pi/2)
ring_pm = emopt.grid_cuda.PolyMat(ring.x, ring.y, material_value=eps_si)

if coupler_angle:
    bus0 = Arc(ring_x, ring_y-coupler_radius,
               radius=coupler_radius, width=bus_guide_width,
               initial_angle=-pi/2, final_angle=-pi/2-coupler_angle/2)
    bus1 = Arc(bus0.x_end, bus0.y_end,
               radius=entry_radius, width=bus_guide_width,
               initial_angle=pi/2-coupler_angle/2, final_angle=pi/2)
    bus2 = Segment(bus1.x_end, bus1.y_end,
                   bus1.x_end-10, bus1.y_end,
                   width=bus_guide_width)
    bus3 = Arc(ring_x, ring_y-coupler_radius,
               radius=coupler_radius, width=bus_guide_width,
               initial_angle=-pi/2, final_angle=-pi/2+coupler_angle/2)
    bus4 = Arc(bus3.x_end, bus3.y_end,
               radius=entry_radius, width=bus_guide_width,
               initial_angle=pi/2+coupler_angle/2, final_angle=pi/2)
    bus5 = Segment(bus4.x_end, bus4.y_end,
                   bus4.x_end+10, bus4.y_end,
                   width=bus_guide_width)
    bus_pms = [emopt.grid_cuda.PolyMat(bus0.x, bus0.y, material_value=eps_si),
               emopt.grid_cuda.PolyMat(bus1.x, bus1.y, material_value=eps_si),
               emopt.grid_cuda.PolyMat(bus2.x, bus2.y, material_value=eps_si),
               emopt.grid_cuda.PolyMat(bus3.x, bus3.y, material_value=eps_si),
               emopt.grid_cuda.PolyMat(bus4.x, bus4.y, material_value=eps_si),
               emopt.grid_cuda.PolyMat(bus5.x, bus5.y, material_value=eps_si)]
    mode_y = bus3.y_end
else:
    bus =  Segment(-X,  ring_y-coupler_radius,
                   2*X, ring_y-coupler_radius,
                   width=bus_guide_width)
    bus_pms = [emopt.grid_cuda.PolyMat(bus.x, bus.y, material_value=eps_si)]
    mode_y = bus.y_end


eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                           [dx, dy, dz],
                                           [[background_pm, -Z, 2*Z],
                                            [slab_pm, Z/2-h_si/2, Z/2-h_si/2+h_slab],
                                            *[ [bus_pm, Z/2-h_si/2, Z/2+h_si/2] for bus_pm in bus_pms],
                                            [ring_pm, Z/2-h_si/2, Z/2+h_si/2]])

mu = emopt.grid_cuda.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

#####################################################################################
# Setup the source
#####################################################################################
mode_x = w_pml + dx
mode_slice = emopt.misc.DomainCoordinates(mode_x, mode_x,
                                          w_pml + dy, mode_y + 4*bus_guide_width,
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
ymax = mode_y + bus_guide_width*4
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
ymax = mode_y + bus_guide_width*4
right_boundary = DomainCoordinates(xmax, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
Ey = sim.get_field_interp('Ey', right_boundary)
Ez = sim.get_field_interp('Ez', right_boundary)
Hy = sim.get_field_interp('Hy', right_boundary)
Hz = sim.get_field_interp('Hz', right_boundary)

if NOT_PARALLEL:
    Pright = 0.5*dy*dz*np.sum(np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy)))
del Ey; del Ez; del Hy; del Hz

# calculate power transmitter through top boundary
xmin = ring_x + ring_diameter/2 - bus_guide_width*4
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
