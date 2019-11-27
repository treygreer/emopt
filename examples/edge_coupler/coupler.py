########
######## for mpi run: OMP_NUM_THREADS=1 mpirun -n 4 *.py
########

import emopt
from emopt.adjoint_method import AdjointMethodPNF3D
import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from emopt.misc import NOT_PARALLEL, run_on_master, RANK, info_message
from sys import stdout
import splines
import pandas
from math import pi, sin, cos
from collections import namedtuple

Point2 = namedtuple('Point2', ('x', 'y'))

interactive = False

####################################################################################
# Simulation parameters.  All dimensions in microns.
####################################################################################
H_SI = 0.22
H_BOX = 2.0
INPUT_MFD = 2.5 # mode field diameter of impinging light at facet
W_WG = 0.45     # output waveguide width (y dimension, total width)

DXYZ = 0.04
DX = DXYZ # grid spacing along x
DY = DXYZ # grid spacing along y
DZ = DXYZ # grid spacing along z
W_PML = 15*DXYZ         # PML thickness
FACET_OFFSET = 16*DX    # distance in x from left sim boundary to die facet
L_TAPER =  10.0         # taper length from facet to beginning of output waveguide (x dimension)
L_WG    =  2.0          # length of output waveguide to measurement plane (x dimension)
OUTPUT_OFFSET = 16*DX   # distance in x from output measurement plane to right sim boundary

X = FACET_OFFSET + L_TAPER + L_WG + OUTPUT_OFFSET
Y = 2*INPUT_MFD/2 + W_PML  # simulation size along y (div 2 for symmetry)
Z = 2*INPUT_MFD + 2*W_PML  # simulation size along z

class Params(object):
    '   All angles in degrees, all coordinates in microns, parameter origin is x:facet edge, y:symmetry axis'
    _param_defs = pandas.DataFrame(
        columns = ('name', 'min', 'max', 'initial'),
        data =    (('tip_x',    .02,   .2,    .1),
                   ('tip_radius', .02,   .2,    .02),
                   ('tip_angle',  -5,    20,    0),    # tip included angle [degrees]
                   ('len1',      .01,   .2,    .01),   # length of first spline segment (pt0 to pt1)
                   ('alpha2',     0,     3/10,  1/10),
                   ('y2',         .02,   .4,    .2),
                   ('alpha3',     1/10,  4/10,  2/10),
                   ('y3',         .02,   .4,    .2),
                   ('alpha4',     2/10,  5/10,    3/10),
                   ('y4',         .02,   .4,    .2),
                   ('alpha5',     3/10,  6/10,    4/10),
                   ('y5',         .02,   .4,    .2),
                   ('alpha6',     4/10,  7/10,    5/10),
                   ('y6',         .02,   .4,    .2),
                   ('alpha7',     5/10,  8/10,    6/10),
                   ('y7',         .02,   .4,    .2),
                   ('alpha8',     6/10,  9/10,    8/10),
                   ('y8',         .02,   .4,    .2),
                   ('len9',       .01,   .2,    .01)   # length of last spline segment (pt8 to pt9)
                   # pt9  is at x=L_TAPER-len9, y=W_WG/2
                   # pt10 is at x=L_TAPER,      y=W_WG/2
               ))
    _param_defs.index = _param_defs['name']  # so we can index dataframe by parameter name

    @classmethod
    def initial_values(cls):
        return cls._param_defs['initial']

    @classmethod
    def bounds(cls):
        return cls._param_defs[['min','max']].values

    def _make_poly_verts(self, spline_N=100, tip_N=20):
        p = self._params['value']
        tip_center_x = FACET_OFFSET + p['tip_x'] + p['tip_radius']
        SPLINE_ORDER = 3
        ctl_points = []
        p0 = Point2(x = tip_center_x - p['tip_radius']*sin(p['tip_angle']/2 * pi/180),
                    y = p['tip_radius']*cos(p['tip_angle']/2 * pi/180))
        ctl_points.append(p0)
        p1 = Point2(x = p0.x + p['len1'] * cos(p['tip_angle']/2 * pi/180),
                    y = p0.y + p['len1'] * sin(p['tip_angle']/2 * pi/180))
        ctl_points.append(p1)
        
        for idx in range(2,9):
            ctl_points.append(Point2(x = FACET_OFFSET + p[f'alpha{idx}'] * L_TAPER,
                                     y = p[f'y{idx}']))

        # next to last point
        ctl_points.append(Point2(x = FACET_OFFSET + L_TAPER - p['len9'],
                                 y = W_WG/2))
        # last point
        ctl_points.append(Point2(x = FACET_OFFSET + L_TAPER,
                                 y = W_WG/2))
        
        knots = splines.spline_even_knots(len(ctl_points), SPLINE_ORDER)
        self.spline = interpolate.BSpline(knots, ctl_points, SPLINE_ORDER)

        tip_pts = [Point2(x = tip_center_x - p['tip_radius']*cos(t),
                          y = p['tip_radius']*sin(t))
                   for t in np.linspace(0, (90-p['tip_angle']/2)*pi/180,
                                        num=tip_N,
                                        endpoint=False)  # don't repeat first point of spline
               ]
        spline_pts = self.spline(np.linspace(0, 1, spline_N))

        self.poly_verts = np.concatenate((tip_pts,
                                          spline_pts,
                                          [Point2(x=X+DX,         y=W_WG/2),
                                           Point2(x=X+DX,         y=-DY),
                                           Point2(x=tip_pts[0].x, y=-DY),
                                           tip_pts[0]]))  # repeat first point of polygon

        if interactive and NOT_PARALLEL:
            plt.figure(1)
            plt.clf()
            plt.plot([p.x for p in ctl_points], [p.y for p in ctl_points], '-x')
            plt.plot((0,0,X,X,0),(0,Y,Y,0,0), lw=0.5, color='grey')
            plt.plot(self.poly_verts[:,0], self.poly_verts[:,1], '-+')
            plt.xlim(-2*DX,X+2*DX)
            #plt.ylim(-2*DY,W_WG*2/3)
            plt.gca().set_aspect('equal')
            plt.ion()
            plt.show()
            plt.pause(.001)

    def __init__(self, param_vector):
        self._params = Params._param_defs
        self._params['value'] = param_vector
        self._make_poly_verts()


    def make_eps(self):
        print(f"rank={RANK} *********make_eps***********...");
        stdout.flush()  # for MPI
        # Geometry consists of input waveguide, output waveguide, and MMI splitting
        # section. Structure is silicon clad in SiO2

        # background
        background_polymat = emopt.grid_cuda.PolyMat([-DX, X+DX, X+DX, -DX],
                                                     [-DY, -DY,  Y+DY, Y+DY],
                                                     material_value=1.444**2)

        # taper
        taper_polymat = emopt.grid_cuda.PolyMat(self.poly_verts[:,0],
                                                self.poly_verts[:,1],
                                                material_value = 3.45**2)


        eps = emopt.grid_cuda.StructuredMaterial3D([X, Y, Z],
                                                   [DX, DY, DZ],
                                                   [[background_polymat, -DZ, Z+DZ],
                                                    [taper_polymat, Z/2-H_SI/2, Z/2+H_SI/2]])
        print(f"rank={RANK} *********make_eps***********...done");
        stdout.flush()  # for MPI
        return eps

    def make_mu(self):
        print(f"rank={RANK} *********make_mu***********...");
        stdout.flush()  # for MPI
        mu = emopt.grid_cuda.ConstantMaterial3D(1.0)
        print(f"rank={RANK} *********make_mu***********...done");
        stdout.flush()  # for MPI
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
        super(MMISplitterAdjointMethod, self).__init__(sim) #, step=1e-2)
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
        print(f"rank={RANK} update system: params=", param_vector)
        params = Params(param_vector)
        eps = params.make_eps()
        mu = params.make_mu() 
        self.sim.set_materials(eps, mu)

    @run_on_master
    def calc_f(self, sim, param_vector):
        """Calculate the mode match based portion of the figure of merit.

        This portion of the  FOM is minus the mode overlap between the
        simulated fields and the fundamental super mode of the output waveguides.
        """
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom

    @run_on_master
    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the figure of merit with respect to E and H.

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
        crv_cost = 1.0 * splines.curvature_cost(params.spline,
                                                min_radius=0.5)
        
        print(f'   calc_penalty: crv_cost = {crv_cost}')
        return crv_cost

    @run_on_master
    def calc_grad_p(self, sim, params):
        """Calculate the derivative of the penalty term in the FOM with respect
        to design variables of the system.
        """
        print(f"rank={RANK} calc_grad_p  params={params} ...")
        func = lambda ps: self.calc_penalty(sim, ps)
        gradient = scipy.optimize.approx_fprime(params, func, epsilon=0.001)
        print(f"rank={RANK} ...calc_grad_p: gradient={gradient}")
        return gradient

@run_on_master
def plot_update(params, fom_list, sim, am):
    """Save a snapshot of the current state of the structure.

    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    print(f'rank={RANK} Finished iteration {len(fom_list)+1}')
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)
    eps_arr = sim.eps.get_values_in(sim.field_domains[1])
    Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[1]
    print(f"rank={RANK} params={params}, fom_list={fom_list}")

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(Ey.squeeze().real, eps_arr.squeeze().real, 
                            sim.X, sim.Y, foms, fname='nurb_current_result.pdf',
                            dark=False)

    data = {}
    data['X']  = sim.X;  data['Y']  = sim.Y;  data['Z']  = sim.Z
    data['dx'] = sim.dx; data['dy'] = sim.dy; data['dz'] = sim.dz
    data['Ex'] = Ex
    data['Ey'] = Ey
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['Hz'] = Hz
    data['eps'] = eps_arr
    data['params'] = params
    data['foms'] = fom_list

    fname = 'data/coupler{:03d}'.format(len(fom_list))
    emopt.io.save_results(fname, data)


    mode_match = am.mode_match.get_mode_match_forward()
    info_message(f"mode_match={mode_match}, source_power={sim.source_power}, \
    ratio={mode_match/sim.source_power}")


wavelength = 1.55
field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2, Z/2,
                                             DX, DY, DZ)

#####################################################################################
# Setup simulation
#####################################################################################
# Setup the simulation--rtol tells the iterative solver when to stop. 5e-5
# yields reasonably accurate results/gradients
sim = emopt.fdtd_cuda.FDTD(X,Y,Z,DX,DY,DZ,wavelength, rtol=1e-5, min_rindex=1.44,
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
sim.w_pml = [W_PML, W_PML, 0,     W_PML, W_PML, W_PML]
sim.bc = 'EHE'

#####################################################################################
# Setup the sources
#####################################################################################
# We excite the system by injecting a gaussian wave of width MFD at the die facet.
#  The facet is not actually a material interface in this sim, but is just inside
#  the PML boundary.  So basically we're just ignoring reflection loss.
input_slice = emopt.misc.DomainCoordinates(FACET_OFFSET, FACET_OFFSET,           # x
                                           0, Y-W_PML,                           # y
                                           W_PML, Z-W_PML,                       # z
                                           DX, DY, DZ)

params = Params(Params.initial_values())
eps = params.make_eps()
mu = params.make_mu()
mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.45,
                                  neigs=1)

# The mode boundary conditions should match the simulation boundary conditions.
# Mode is in the y-z plane, so the boundary conditions are HE
mode.bc = 'HE'
mode.build_gaussian_beam(center=Point2(x=0,y=Z/2),
                         mfd=INPUT_MFD)
sources = mode.get_source(i=0, dx=DX, dy=DY, dz=DZ)
if NOT_PARALLEL:
    print(f'local: sources[0].shape={sources[0].shape}')
    sim.set_sources(sources, input_slice)
    if interactive and NOT_PARALLEL:
        plt.figure(2)
        plt.clf()
        plt.imshow(np.abs(np.abs(mode.get_field_interp(0, 'Hz')).squeeze()))
        plt.colorbar()
        input("Press [enter] to continue.")

#####################################################################################
# Mode match for optimization
#####################################################################################
# We need to calculate the field used as the reference field in our mode match
# figure of merit calculation. This is the fundamental mode of the output waveguide.
fom_slice = emopt.misc.DomainCoordinates(X-OUTPUT_OFFSET, X-OUTPUT_OFFSET,    # x 
                                         0, Y-W_PML,                          # y
                                         W_PML, Z-W_PML,                      # z
                                         DX, DY, DZ)
fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45,
                                      neigs=4)

# Need to be consistent with boundary conditions!
fom_mode.bc = 'HE'
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

mode_match = emopt.fomutils.ModeMatch(normal=[1,0,0], ds1=DY, ds2=DZ, **fom_mode_fields)

if NOT_PARALLEL:
    #####################################################################################
    # define additional domains for field reporting (field domain 0 is for fom calculation)
    #####################################################################################
    # horizontal slice through middle of waveguides
    field_monitor_slice1 = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2, Z/2,
                                                        DX, DY, DZ)
    # vertical slice through symmetry plane
    field_monitor_slice2 = emopt.misc.DomainCoordinates(0, X, 0, 0, 0, Z,
                                                        DX, DY, DZ)

    # vertical slice through output waveguide
    sim.field_domains = [fom_slice, field_monitor_slice1,
                         field_monitor_slice2]

    #####################################################################################
    # Setup the AdjointMethod object needed for gradient calculations
    #####################################################################################
    am = MMISplitterAdjointMethod(sim, fom_slice, mode_match)
    
    #am.check_gradient(am.init)


    #####################################################################################
    # Setup and run the optimization
    #####################################################################################
    # L-BFGS-B will print out the iteration number and FOM value
    fom_list = []
    callback = lambda parms : plot_update(parms, fom_list, sim, am)
    max_iters = 30
    initial_params = Params.initial_values()
    opt = emopt.optimizer.Optimizer(am, initial_params, Nmax=max_iters, opt_method='L-BFGS-B',
                                    callback_func=callback, bounds=Params.bounds())
    print("**************calling optimizer")
    stdout.flush()  # for MPI
    #fom = am.fom(Params.init)
    final_fom, final_params = opt.run()

    if interactive and NOT_PARALLEL:
        input("Press [enter] to continue.")

print(f"******* rank={RANK} ********* script done *********************")
