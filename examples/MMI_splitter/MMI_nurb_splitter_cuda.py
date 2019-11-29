#!/bin/python3
########
######## for mpi run: OMP_NUM_THREADS=1 mpirun -n 4 python3 MMI_nurb_splitter.py
########

"""Demonstrates how optimize an MMI 1x2 splitter in 3D using the CW-FDTD
solver.

This optimization involves varying the width and height of a silicon slab in
order to fine tune multimode interference with the ultimate goal of splitting
light from a single input waveguide equally between two output waveguides:

        --------------------
        |                  -------------
        |                  -------------
---------        MMI       |
---------      Splitter    |
        |                  -------------
        |                  -------------
        --------------------

This structure should have pretty well-defined local optima. For a given MMI
splitter height, there will be a corresponding optimal length which produces
the desired relative phase between the fundamental and higher order modes.

This optimization is setup to for the TE-like polarization. In order to design
a TM-like device, you should be able to just modify the symmetric boundary
conditions.

To run the script run:

    $ mpirun -n 16 python mmi_1x2_splitter_3D_fdtd.py

If you want to run the script on a different number of processors, change 16 to
the desired value.

The primary advantage of using the FDTD solver is that it enables us to tackle
much larger/higher resolution problems. In this example, we use a finer grid
spacing than the FDFD example.
"""

import emopt
from emopt.adjoint_method import AdjointMethodPNF3D
import numpy as np
import scipy
from scipy import interpolate
import shelve
import gc
import shapely
import shapely.ops
import matplotlib.pyplot as plt
import sys
from emopt.misc import NOT_PARALLEL, run_on_master, RANK
from petsc4py import PETSc
rank = PETSc.COMM_WORLD.getRank()

interactive = False
output_gds = True
y_symmetry = True

if output_gds:
    import gdspy

####################################################################################
# Simulation parameters
####################################################################################
dxyz = 0.02
#dxyz = 0.04
dx = dxyz # grid spacing along x
dy = dxyz # grid spacing along y
dz = dxyz # grid spacing along z
w_pml = min(0.5, 15*dxyz) # set the PML width
port_offset = w_pml * 16/15  # distance in x from edge to i/o port
X = 5 + 2*port_offset   # simulation size along x
Y = 4.0   # simulation size along y
if y_symmetry: Y=Y/2
Z = 2.5   # simulation size along z
w_wg = 0.5
h_si = 0.22

def spline_even_knots(num_ctl_points, order):
    knots_len = num_ctl_points + order + 1
    return order*[0] + list(np.linspace(0, 1, knots_len-2*order)) + order*[1]

def spline_curvature(spline, t):
    d1 = spline.derivative(1)(t)
    d2 = spline.derivative(2)(t)
    num = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0]
    denom = np.power(np.sum(d1*d1, axis=1), 3/2)
    denom[denom==0.0] = 1e-6
    return (num.T/denom).T

def bode_knee(x, knee=1, width=0.1):
    from numpy import exp, log
    j = 0+1j
    e = exp(1)
    y = log((j*exp(x/width) - exp(knee/width)))*width - knee
    y[np.isinf(y)] = x[np.isinf(y)]
    y[np.isnan(y)] = x[np.isnan(y)]
    return y.real

def curvature_cost(spline, min_radius=0.1):
    N = 1000
    max_curvature = 1/min_radius
    t = np.linspace(0, 1, N)
    curv = np.abs(spline_curvature(spline, t))
    #curv_rect = 1-emopt.fomutils.rect(curv, 2*max_curvature, max_curvature/10)
    curv_knee = bode_knee(curv, max_curvature)
    cost = np.sum(curv_knee) / N
    return cost

def intersection_cost(spline1, spline2, min_width=0.2):
    N = 1000
    t = np.linspace(0, 1, N)
    points1 = spline1(t)
    points2 = spline2(t)
    dist_matrix = scipy.spatial.distance.cdist(points1, points2, 'euclidean')
    print(f'rank={rank} minimum distance = {np.min(dist_matrix)}')
    inv_dist_matrix = 1.0 / dist_matrix;
    inv_dist_bode = bode_knee(1.0/dist_matrix, 1.0/min_width)
    cost = np.sum(inv_dist_bode) / (N*N)
    return 1000 * cost

class Params(object):
    _xmin = port_offset
    _xmax = X-port_offset
    _ymin = 0
    _ymax = Y-w_pml
    _spl1_points = \
    ([_xmin,                                w_wg/2],
     [['a1x', _xmin+0.1, _xmax, 1.58],      w_wg/2],
     [['a2x', _xmin+0.1, _xmax, 1.75],      ['a2y', _ymin, _ymax, 0.4]],
     [['a3x', _xmin+0.1, _xmax, 1.93],      ['a3y', _ymin, _ymax, 0.9]],
     [['a4x', _xmin+0.1, _xmax, 4.0],       ['a4y', _ymin, _ymax, 0.9]],
     [['a5x', _xmin+0.1, _xmax, 4.25],      ['a5y', _ymin, _ymax, 0.8]],
     [['a6x', _xmin+0.1, _xmax-0.1, 4.4],   w_wg*3/2],
     [_xmax,                                w_wg*3/2])

    _spl2_points = \
    ([_xmax,                                w_wg/2],
     [['b1x',_xmin,_xmax-0.1, 4.4],         w_wg/2],
     [['b2x',_xmin,_xmax-0.1, 4.25],        ['b2y', _ymin+0.01, w_wg*3/2, 0.16]],
     [['b3x',_xmin,_xmax-0.1, 4.25],        ['b3y', _ymin+0.01, w_wg*3/2, 0.1]],
     ['b3x',                                _ymin])
    
    _params = {}
    for pt in (*_spl1_points, *_spl2_points):
        x,y = pt
        for coord in (x,y):
            if isinstance(coord, list):
                name, minimum, maximum, default = coord
                _params[name] = (minimum, maximum, default)

    _keys = [k for k in _params.keys()]
    idx = 0; _param_map = {}; init = []; bounds = [];
    for key in _keys: 
        bounds.append((_params[key][0], _params[key][1]))
        init.append(_params[key][2])
        _param_map[key] = idx
        idx += 1
    init = np.array(init)
    bounds = np.array(bounds)

    def __init__(self, param_vector):
        self._param_vector = param_vector
        self._make_splines()
        self._make_poly_verts()

    def _make_splines(self):
        spline_order = 3
        ctl_points1 = []
        for (x,y) in Params._spl1_points:
            if isinstance(x, list):
                x = self._param_vector[Params._param_map[x[0]]]
            if isinstance(x, str):
                x = self._param_vector[Params._param_map[x]]
            if isinstance(y, list):
                y = self._param_vector[Params._param_map[y[0]]]
            if isinstance(y, str):
                y = self._param_vector[Params._param_map[y]]
            ctl_points1.append((x,y))
        ctl_points1 = np.array(ctl_points1)
        knots = spline_even_knots(len(ctl_points1), spline_order)
        self.spline1 = interpolate.BSpline(knots, ctl_points1, spline_order)
            
        ctl_points2 = []
        for (x,y) in Params._spl2_points:
            if isinstance(x, list):
                x = self._param_vector[Params._param_map[x[0]]]
            if isinstance(x, str):
                x = self._param_vector[Params._param_map[x]]
            if isinstance(y, list):
                y = self._param_vector[Params._param_map[y[0]]]
            if isinstance(y, str):
                y = self._param_vector[Params._param_map[y]]
            ctl_points2.append((x,y))
        ctl_points2 = np.array(ctl_points2)

        if interactive:
            plt.figure(1)
            plt.clf()
            plt.plot(ctl_points1[:,0], ctl_points1[:,1], '-x')
            plt.plot(ctl_points2[:,0], ctl_points2[:,1], '-x')
            plt.xlim(-2*dx,X+2*dx)
            plt.ylim(-2*dy,Y+2*dy)
            plt.plot((0,0,X,X,0),(0,Y,Y,0,0), lw=0.5, color='grey')

        knots = spline_even_knots(len(ctl_points2), spline_order)
        self.spline2 = interpolate.BSpline(knots, ctl_points2, spline_order)

    def _make_poly_verts(self, N=50):

        # x coordinate where spline2 reflects around x-axis
        reflect_x = self._param_vector[self._param_map['b3x']] 

        t1 = np.linspace(0, 1, N)
        N2 = N//3
        t2 = np.linspace(0, 1, N2)
        spline1 = self.spline1(t1)
        spline2 = self.spline2(t2)

        if y_symmetry:
            self.poly_verts = np.concatenate((spline1,
                                              np.array([[X+dx,        w_wg*3/2],
                                                        [X+dx,        w_wg/2]]),
                                              spline2,
                                              np.array([[reflect_x,  -dy],
                                                        [-dx,        -dy],
                                                        [-dx,         w_wg/2],
                                                        [port_offset, w_wg/2]])))
        else:
            spline3 = self.spline2(t2[N2-1::-1]); spline3[:,1]=-spline3[:,1] # reverse and reflect
            spline4 = self.spline1(t1[N::-1]);   spline4[:,1]=-spline4[:,1]   # reverse and reflect
            self.poly_verts = np.concatenate((spline1,
                                              np.array([[X+dx,         w_wg*3/2],
                                                        [X+dx,         w_wg/2]]),
                                              spline2,
                                              spline3,
                                              np.array([[X+dx,        -w_wg/2],
                                                        [X+dx,        -w_wg*3/2]]),
                                              spline4,
                                              np.array([[-dx,         -w_wg/2],
                                                        [-dx,          w_wg/2],
                                                        spline1[0]])))
            # apply Y offset
            self.poly_verts[:,1] = self.poly_verts[:,1] + Y/2
        

        if interactive:
            plt.plot(reflect_x, -dy, '1')
            plt.plot(self.poly_verts[:,0], self.poly_verts[:,1], '-+')
            plt.ion()
            plt.show()
            plt.pause(.001)

    def make_eps(self):
        print(f"rank={rank} *********make_eps***********...");
        sys.stdout.flush()  # for MPI
        # Geometry consists of input waveguide, output waveguide, and MMI splitting
        # section. Structure is silicon clad in SiO2

        # background
        background_poly = emopt.grid_cuda.PolyMat([-X/2, 3*X/2, 3*X/2, -X/2],
                                                  [-Y/2, -Y/2,  3*Y/2, 3*Y/2],
                                                  material_value=1.444**2)

        # mmi

        mmi_poly = emopt.grid_cuda.PolyMat(self.poly_verts[:,0],
                                           self.poly_verts[:,1],
                                           material_value = 3.45**2)


        eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                                   [dx, dy, dz],
                                                   [[background_poly, -Z, 2*Z],
                                                    [mmi_poly, Z/2-h_si/2, Z/2+h_si/2]])
        print(f"rank={rank} *********make_eps***********...done");
        sys.stdout.flush()  # for MPI
        return eps

    def make_mu(self):
        print(f"rank={rank} *********make_mu***********...");
        sys.stdout.flush()  # for MPI
        mu = emopt.grid_cuda.ConstantMaterial3D(1.0)
        print(f"rank={rank} *********make_mu***********...done");
        sys.stdout.flush()  # for MPI
        return mu

class MMISplitterAdjointMethod(AdjointMethodPNF3D):
    """Define a figure of merit and its derivative for adjoint sensitivity
    analysis.

    Our goal is to optimize the the dimensions of an MMI 1x2 splitter using
    gradients computed using the adjoint method. In this problem, we choose the
    figure of merit (FOM) to be the overlap of the simulated fields with the
    super mode which has power equally split between the two waveguides. Both
    the evaluation of the FOM and its derivative with respect to E and H is
    handled by the emopt.fomutils.ModeMatch class.
    """
        


    @run_on_master
    def __init__(self, sim, fom_domain, mode_match):
        super(MMISplitterAdjointMethod, self).__init__(sim, step=1e-2)
        self.mode_match = mode_match
        self.fom_domain = fom_domain

    @run_on_master
    def update_system(self, param_vector):
        """Update the geometry of the system based on the current design
        parameters.

        The design parameter vector has the following format:

            params = [mmi_width, mmi_length]

        We use these values to modify the structure dimensions
        """
        print(f"rank={rank} update system: params=", param_vector)
        params = Params(param_vector)
        eps = params.make_eps()
        mu = params.make_mu() 
        self.sim.set_materials(eps, mu)

    @run_on_master
    def calc_f(self, sim, param_vector):
        """Calculate the figure of merit.

        The FOM is the mode overlap between the simulated fields and the
        fundamental super mode of the output waveguides.
        """
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        """Calculate the figure of merit with respect to E and H.

        Note: our function is normalized with respect to the total source
        power and our derivative needs to account for this`*`. Currently,
        AdjointMethodPNF does not handle 3D simulations. For now, instead use
        emopt.fomutils.power_norm_dFdx_3D directly.

        `*` One might think that no power should couple back into the source.
        Unfortunately, due to the finite extent of the source, discretization,
        etc, this is not the case and modifications to the geometry will modify
        the source power.
        """
        Psrc = sim.source_power

        dfdEx = -1*self.mode_match.get_dFdEx()
        dfdEy = -1*self.mode_match.get_dFdEy()
        dfdEz = -1*self.mode_match.get_dFdEz()
        dfdHx = -1*self.mode_match.get_dFdHx()
        dfdHy = -1*self.mode_match.get_dFdHy()
        dfdHz = -1*self.mode_match.get_dFdHz()

        return [(dfdEx, dfdEy, dfdEz, dfdHx, dfdHy, dfdHz)]

    @run_on_master
    def get_fom_domains(self):
        """We must return the DomainCoordinates object that corresponds to our
        figure of merit. In theory, we could have many of these.
        """
        return [self.fom_domain]

    @run_on_master
    def calc_penalty(self, sim, param_vector):
        """Calculate a penalty to the figure of merit due to curvature
        """
        params = Params(param_vector)
        crv_cost = self.curvature_penalty*(curvature_cost(params.spline1) +
                                           curvature_cost(params.spline2))
        
        print(f'   rank={rank} calc_penalty: crv_cost = {crv_cost}')
        return crv_cost

    @run_on_master
    def calc_grad_p(self, sim, params):
        """Calculate the derivative of the penalty term in the FOM with respect
        to design variables of the system.
        """
        print(f"rank={rank} calc_grad_p  params={params} ...")
        func = lambda ps: self.calc_penalty(sim, ps)
        gradient = scipy.optimize.approx_fprime(params, func, epsilon=0.001)
        print(f"rank={rank} ...calc_grad_p: gradient={gradient}")
        return gradient

@run_on_master
def plot_update(params, fom_list, sim, am):
    """Save a snapshot of the current state of the structure.

    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    print(f'rank={rank} Finished iteration {len(fom_list)+1}')
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    eps_arr = sim.eps.get_values_in(sim.field_domains[1])
    Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[1]
    print(f"rank={rank} params={params}, fom_list={fom_list}")

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(Ey.squeeze().real, eps_arr.squeeze().real, 
                            sim.X, sim.Y, foms, fname='nurb_current_result.pdf',
                            dark=False)

    data = {}
    data['Ex'] = Ex
    data['Ey'] = Ey
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['Hz'] = Hz
    data['eps'] = eps_arr
    data['params'] = params
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/MMI_nurb_splitter{:03d}'.format(i)
    emopt.io.save_results(fname, data)

    if output_gds:
        gdspy.current_library.cell_dict={}
        cell = gdspy.Cell(f'MMI_splitter_half')
        cell.add(gdspy.Polygon(Params(params).poly_verts[:-1])) # omit repeated vertex
        gdspy.write_gds(f'data/MMI_splitter_{i:03d}.gds', unit=1.0e-6, precision=1.0e-9)



wavelength = 1.55
field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2,
                                             Z/2, dx, dy, dz)

#####################################################################################
# Setup simulation
#####################################################################################
# Setup the simulation--rtol tells the iterative solver when to stop. 5e-5
# yields reasonably accurate results/gradients
sim = emopt.fdtd_cuda.FDTD(X,Y,Z,dx,dy,dz,wavelength, rtol=1e-5, min_rindex=1.44,
                           nconv=200) #, num_time_steps=1)
sim.Nmax = 1000*sim.Ncycle
sim.verbose = 100

# get actual simulation dimensions
X = sim.X
Y = sim.Y
Z = sim.Z

print(f'Nx,Ny,Nz = {sim._Nx},{sim._Ny},{sim._Nz}')
print(f'Nx*Ny*Nz = {sim._Nx*sim._Ny*sim._Nz}')

# we use symmetry boundary conditions at y=0 to speed things up. We
# need to make sure to set the PML width at the minimum y boundary is set to
# zero. Currently, FDTD cannot compute accurate gradients using symmetry in z
# :(
if y_symmetry:
    sim.w_pml = [w_pml, w_pml, 0,     w_pml, w_pml, w_pml]
    sim.bc = 'EHE'
else:
    sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
    sim.bc = 'EEE'

#####################################################################################
# Setup the sources
#####################################################################################
# We excite the system by injecting the fundamental mode of the input waveguide
input_slice = emopt.misc.DomainCoordinates(port_offset, port_offset,             # x
                                           0 if y_symmetry else w_pml, Y-w_pml,  # y
                                           w_pml, Z-w_pml,                       # z
                                           dx, dy, dz)

params = Params(Params.init)
eps = params.make_eps()
mu = params.make_mu()
mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.45,
                                  neigs=4)

# The mode boundary conditions should match the simulation boundary conditins.
# Mode is in the y-z plane, so the boundary conditions are H0
mode.bc = 'HE' if y_symmetry else 'EE'
mode.build()
mode.solve()
sources = mode.get_source(i=0, dx=dx, dy=dy, dz=dz)
if NOT_PARALLEL:
    print(f'local: sources[0].shape={sources[0].shape}')
    sim.set_sources(sources, input_slice)
    if interactive:
        plt.figure(2)
        plt.clf()
        plt.imshow(np.abs(np.abs(mode.get_field_interp(0, 'Ex')).squeeze()))
        plt.colorbar()
        input("Press [enter] to continue.")

#####################################################################################
# Mode match for optimization
#####################################################################################
# we need to calculate the field used as the reference field in our mode match
# figure of merit calculation. This is the fundamental super mode of the output
# waveguides.
fom_slice = emopt.misc.DomainCoordinates(X-port_offset, X-port_offset,        # x 
                                         0 if y_symmetry else w_pml, Y-w_pml, # y
                                         w_pml, Z-w_pml,                      # z
                                         dx, dy, dz)

fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45,
                                      neigs=4)

# Need to be consistent with boundary conditions!
fom_mode.bc = 'HE' if y_symmetry else 'EE'
fom_mode.build()
fom_mode.solve()
fom_mode_fields = {
    'Exm': fom_mode.get_field_interp(0, 'Ex'),
    'Eym': fom_mode.get_field_interp(0, 'Ey'),
    'Ezm': fom_mode.get_field_interp(0, 'Ez'),
    'Hxm': fom_mode.get_field_interp(0, 'Hx'),
    'Hym': fom_mode.get_field_interp(0, 'Hy'),
    'Hzm': fom_mode.get_field_interp(0, 'Hz')
    }
print(f'local: fom_mode_fields["Hzm"].shape={fom_mode_fields["Hzm"].shape}')

#with shelve.open('data/modes.shelve', 'r') as shelf:
#    fom_mode_fields = shelf[f'fom_mode_fields_{dxyz}']
#    print(f'shelved: fom_mode_fields["Hzm"].shape={fom_mode_fields["Hzm"].shape}')
mode_match = emopt.fomutils.ModeMatch(normal=[1,0,0], ds1=dy, ds2=dz, **fom_mode_fields)

if NOT_PARALLEL:
    #####################################################################################
    # define additional domains for field reporting (field domain 0 is for fom calculation)
    #####################################################################################
    # horizontal slice through middle of waveguides
    field_monitor_slice1 = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2, Z/2,
                                                        dx, dy, dz)
    # vertical slice through symmetry plane
    field_monitor_slice2 = emopt.misc.DomainCoordinates(0, X, 0, 0, 0, Z,
                                                        dx, dy, dz)
    field_monitor_slice3 = emopt.misc.DomainCoordinates(0, X, w_wg, w_wg, 0, Z,
                                                        dx, dy, dz)
    # vertical slice through output waveguide
    sim.field_domains = [fom_slice, field_monitor_slice1,
                         field_monitor_slice2,
                         field_monitor_slice3]

    #####################################################################################
    # Setup the AdjointMethod object needed for gradient calculations
    #####################################################################################
    am = MMISplitterAdjointMethod(sim, fom_slice, mode_match)
    am.curvature_penalty =  1.0
    am.intersection_penalty =  1.0
    
    #####################################################################################
    # Setup and run the optimization
    #####################################################################################
    # L-BFGS-B will print out the iteration number and FOM value
    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sim, am)
    max_iters = 10
    #initial_params = Params.init
    initial_params = [1.59640526, 1.71730974, 0.39033418, 1.89133451, 0.92390186, 4.01017023,
                      0.89927224, 4.27082338, 0.82687517, 4.39763377, 4.45615605, 4.29127133,
                      0.13311415, 4.11475779, 0.10247903]
    opt = emopt.optimizer.Optimizer(am, initial_params, Nmax=max_iters, opt_method='L-BFGS-B',
                                    callback_func=callback, bounds=Params.bounds)
    print("**************calling optimizer")
    sys.stdout.flush()  # for MPI
    #fom = am.fom(Params.init)
    final_fom, final_params = opt.run()


    # for valgrind
    #del sim, eps, mu, mode, mode_match, fom_mode, opt
    #gc.collect()


    if interactive:
        input("Press [enter] to continue.")

print(f"******* rank={RANK} ********* script done *********************")
