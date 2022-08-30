"""Solve for the modes of electromagnetic waveguides in 2D and 3D.

Waveguide modes can be computed by setting up a generalized eigenvalue problem
corresponding to the source-free Maxwell's equations assuming a solution to
:math:`\mathbf{E}` and :math:`\mathbf{H}` which is proportional to :math:`e^{i
k_z z}`, i.e.

.. math::
    \\nabla \\times e^{i k_z z} \mathbf{E} - i \\mu_r \\omega
    e^{i k_z z}\\mathbf{H} = 0

    \\nabla \\times e^{i k_z z} \mathbf{H} + i \\epsilon_r \\omega
    e^{i k_z z}\\mathbf{E} = 0

where we have used the non-dimensionalized Maxwell's equations. These equations
can be written in the form

.. math::
    A x = n_z B x

where :math:`A` contains the discretized curls and material values, :math:`B` is
singular matrix containing only 1s and 0s, and :math:`n_z` is the effective
index of the mode whose field components are contained in :math:`x`.  Although
formulating the problem like this results in a sparse matrix with ~2x the
number of values compared to other formulations discussed in the literature[1],
it has the great advantage that the equations remain very simple which
simplifies the code. This formulation also makes it almost trivial to implement
anisotropic materials (tensors) in the future, if desired.

In addition to solving for the fields of a waveguide's modes, we can also
compute the current sources which excite only that mode. This can be used in
conjunction with :class:`emopt.fdfd.FDFD` to simulated waveguide structures
which are particularly interesting for applications in silicon photonics, etc.

References
----------
[1] A. B. Fallahkhair, K. S. Li and T. E. Murphy, "Vector Finite Difference
Modesolver for Anisotropic Dielectric Waveguides", J. Lightwave Technol. 26(11),
1423-1431, (2008).
"""
from __future__ import absolute_import
# Initialize petsc first
from builtins import range
from builtins import object
import sys, slepc4py
from future.utils import with_metaclass
slepc4py.init(sys.argv)

from .defs import FieldComponent
from .misc import info_message, warning_message, error_message, \
NOT_PARALLEL, run_on_master, MathDummy, RANK

from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class ModeFullVector(object):
    """Solve for the modes for a 2D slice of a 3D structure.

    Parameters
    ----------
    wavelength : float
        The wavelength of the modes.
    ds : float
        The grid spacing in the mode field (y) direction.
    eps : numpy.ndarray
        The array containing the slice of permittivity for which the modes are
        calculated.
    n0 : float (optional)
        The 'guess' for the effective index around which the modes are
        computed. In general, this value should be larger than the index of the
        mode you are looking for. (default = 1.0)
    neigs : int (optional)
        The number of modes to compute. (default = 1)
    backwards : bool
        Defines whether or not the mode propagates in the forward +x direction
        (False) or the backwards -x direction (True). (default = False)

    Attributes
    ----------
    wavelength : float
        The wavelength of the solved modes.
    neff : list of floats
        The list of solved effective indices
    n0 : float
        The effective index near which modes are found
    neigs : int
        The number of modes to solve for.
    dir : int
        Direction of mode propagation (1 = forward, -1 = backward)
    bc : str
        The boundary conditions used. The possible boundary conditions are:
            0 -- Perfect electric conductor (top and bottom)
            E -- Electric field symmetry (bottom) and PEC (top)
            H -- Magnetic field symmetry (bottom) and PEC (top)

    Methods
    -------
    build(self)
        Build the system of equations and prepare the mode solver for the solution
        process.
    solve(self)
        Solve for the modes of the structure.
    get_field(self, i, component)
        Get the desired raw field component of the i'th mode.
    get_field_interp(self, i, component)
        Get the desired interpolated field component of the i'th mode
    get_mode_number(self, i):
        Estimate the number X of the given TE_X mode.
    find_mode_index(self, X):
        Find the index of a TE_X mode with the desired X.
    get_source(self, i, ds1, ds2, ds3=0.0)
        Get the source current distribution for the i'th mode.
    """

    def __init__(self, wavelength, eps, domain, n0=1.0, neigs=1, \
                 backwards=False, verbose=True):

        info_message('ModeFullVector __init__')
        self.n0 = n0
        self.neigs = neigs
        self.wavelength = wavelength

        self.eps = eps
        self.domain = domain

        self._fshape = domain.shape

        # figure out how 2D domain is oriented
        if(domain.Nx == 1):
            self.ndir = 'x'
            self._M = domain.Nz
            self._N = domain.Ny
            self.dx = domain.dy
            self.dy = domain.dz
        elif(domain.Ny == 1):
            self.ndir = 'y'
            self._M = domain.Nz
            self._N = domain.Nx
            self.dx = domain.dx
            self.dy = domain.dz
        elif(domain.Nz == 1):
            self.ndir = 'z'
            self._M = domain.Ny
            self._N = domain.Nx
            self.dx = domain.dx
            self.dy = domain.dy
        else:
            raise AttributeError('3D domains are not supported for 2D mode ' \
                                 'calculations!')

        self.verbose = verbose

        # this minus sign needs attention... It doesnt really make sense,
        # currently
        if(backwards):
            self._dir = -1.0
        else:
            self._dir = 1.0

        # non-dimensionalization for spatial variables
        self.R = self.wavelength/(2*np.pi)

        # Solve problem of the form Ax = nBx
        # define A and B matrices here
        # 6 fields
        Nfields = 6
        self._A = PETSc.Mat()
        self._A.create(PETSc.COMM_WORLD)
        self._A.setSizes([Nfields*self._M*self._N, Nfields*self._M*self._N])
        self._A.setType('aij')
        self._A.setUp()

        self._B = PETSc.Mat()
        self._B.create(PETSc.COMM_WORLD)
        self._B.setSizes([Nfields*self._M*self._N, Nfields*self._M*self._N])
        self._B.setType('aij')
        self._B.setUp()

        # setup the solver
        self._solver = SLEPc.EPS()
        self._solver.create()

        # we need to set up the spectral transformation so that it doesnt try
        # to invert B
        st = self._solver.getST()
        st.setType('sinvert')

        # Let's use MUMPS for any system solving since it is fast
        ksp = st.getKSP()
        #ksp.setType('gmres')
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')

        # backwards compatibility
        try:
            pc.setFactorSolverPackage('mumps')
        except AttributeError as ae:
            pc.setFactorSolverType('mumps')

        # setup vectors for the solution
        self._x = []
        self._neff = np.zeros(neigs, dtype=np.complex128)
        vr, wr = self._A.getVecs()
        self._x.append(vr)

        for i in range(neigs-1):
            self._x.append(self._x[0].copy())

        ib, ie = self._A.getOwnershipRange()
        self.ib = ib
        self.ie = ie
        #indset = self._A.getOwnershipIS()

        #self._ISEx = indset[0].createBlock(self._M*self._N, [0])
        #self._ISEy = indset[0].createBlock(self._M*self._N, [1])
        #self._ISEz = indset[0].createBlock(self._M*self._N, [2])
        #self._ISHx = indset[0].createBlock(self._M*self._N, [3])
        #self._ISHy = indset[0].createBlock(self._M*self._N, [4])
        #self._ISHz = indset[0].createBlock(self._M*self._N, [5])

        # Boundary conditions default to H symmetry
        self._bc = ['H', 'H']

    @property
    def neff(self):
        return self._neff

    @neff.setter
    def neff(self, value):
        warning_message('neff cannot be set by the user.', \
                        module='emopt.modes')

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc):
        if(bc[0] not in ['E', 'H']):
            error_message(f"Boundary condition type '{bc[0]}' not found. Use E or H.")
        if(bc[1] not in ['E', 'H']):
            error_message(f"Boundary condition type '{bc[1]}' not found. Use E or H.")

        self._bc = bc

    def build(self):
        """Build the system of equations and prepare the mode solver for the solution
        process.

        In order to solve for the eigen modes, we must first assemble the
        relevant matrices :math:`A` and :math:`B` for the generalized
        eigenvalue problem given by :math:`A x = n_x B x` where :math:`n_x` is
        the eigenvalue and :math:`x` is the vector containing the eigen modes.

        Notes
        -----
        This function is run on all nodes.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Building mode solver system matrix...')

        dx = self.dx/self.R # non-dimensionalize
        dy = self.dy/self.R # non-dimensionalize

        odx = 1.0/dx
        ody = 1.0/dy

        A = self._A
        B = self._B
        eps = self.eps
        M = self._M
        N = self._N

        bc = self._bc

        # handle coordinate permutations for 3D slices. This is necessary since
        # the solver assumes the mode propagates in the z direction however
        # slices pointing in other directions may be provided when used in
        # conjunction with the 3D solver.
        i0 = self.domain.i1   # [i,j,k] are in domain space
        j0 = self.domain.j1
        k0 = self.domain.k1
        print(f"rank {RANK} self.ndir={self.ndir}")
        sys.stdout.flush()

        if NOT_PARALLEL:
            print(f"*************** building eps and mu arrays on node 0 ***********************************")
            sys.stdout.flush()
            if(self.ndir == 'x'):     #                                                                      z,y (3D domain)
                #                                                                                       ->   y,x (2D mode)
                eps_x = eps.get_values(k0, k0+1, j0, j0+N, i0, i0+M, koff=0.0, joff=0.5, ioff=0.0, arr=None)[:,:,0]
                eps_y = eps.get_values(k0, k0+1, j0, j0+N, i0, i0+M, koff=0.0, joff=0.0, ioff=0.5, arr=None)[:,:,0]
                eps_z = eps.get_values(k0, k0+1, j0, j0+N, i0, i0+M, koff=0.0, joff=0.0, ioff=0.0, arr=None)[:,:,0]
            elif(self.ndir == 'y'):
                # TODO: FIXME
                get_eps_x = lambda x,y : eps.get_value(k0+y,     j0, i0+x)
                get_eps_y = lambda x,y : eps.get_value(k0+y+0.5, j0, i0+x-0.5)
                get_eps_z = lambda x,y : eps.get_value(k0+y,     j0, i0+x-0.5)
            elif(self.ndir == 'z'):
                # TODO: does this need 1/2 cell offsets???
                eps_x = eps.get_values(k0, k0+N, j0, j0+M, i0, i0+1, koff=0.0, joff=0.0, ioff=0.0, arr=None)[0,:,:]
                eps_y = eps_x;
                eps_z = eps_x;
        else:
            eps_x = None
            eps_y = None
            eps_z = None
        
        comm = MPI.COMM_WORLD
        eps_x = comm.bcast(eps_x, root=0)
        eps_y = comm.bcast(eps_y, root=0)
        eps_z = comm.bcast(eps_z, root=0)

        print(f"************************************** rank {RANK} self.ib={self.ib}, self.id={self.ie}");
        print(f"eps_x.shape={eps_x.shape}")
        sys.stdout.flush()

        for I in range(self.ib, self.ie):
            A[I,I] = 0.0
            B[I,I] = 0.0

            # (stuff) = n_z B H_y
            if(I < N*M):
                y = int((I-0*M*N)/N)
                x = (I-0*M*N) - y * N

                JHz1 = 5*M*N + y*N + x
                JHz0 = 5*M*N + (y-1)*N + x
                JEx = y*N + x
                JHy = 4*M*N + y*N + x

                # derivative of Hz
                if(y > 0): A[I, JHz0] = -ody
                A[I, JHz1] = ody

                # Ex
                A[I, JEx] = 1j*eps_x[y,x]

                # Setup the LHS B matrix
                B[I,JHy] = 1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(y == 0):
                    if(bc[1] == 'E'):  # H_z has odd symmetry at y=0
                        A[I, JHz1] = 2*ody
                    else:  # H_z has even symmetry at y=0
                        A[I, JHz1] = 0

            # (stuff) = n_z B H_x
            elif(I < 2*N*M):
                y = int((I-1*M*N)/N)
                x = (I-1*M*N) - y * N

                JHz1 = 5*N*M + y*N + x
                JHz0 = 5*N*M + y*N + (x-1)

                JEy = M*N + y*N + x
                JHx = 3*M*N + y*N + x

                # derivative of Hz
                if(x > 0): A[I, JHz0] = odx
                A[I, JHz1] = -odx

                # Ey
                A[I, JEy] = 1j*eps_y[y,x]

                # Setup the LHS B matrix
                B[I,JHx] = -1j*self._dir

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == 'E'):
                        A[I, JHz1] = -2*odx
                    else:
                        A[I, JHz1] = 0

            # (stuff) = n_z B E_z
            elif(I < 3*N*M):
                y = int((I-2*M*N)/N)
                x = (I-2*M*N) - y * N

                JEy0 = M*N + y*N + x
                JEy1 = M*N + y*N + (x+1)

                JEx0 = y*N + x
                JEx1 = (y+1)*N + x

                JHz = 5*M*N + y*N + x

                # derivative of Ey
                A[I, JEy0] = -odx
                if(x < N-1):
                    A[I,JEy1] = odx

                # derivative of Ex
                A[I, JEx0] = ody
                if(y < M-1):
                    A[I, JEx1] = -ody

                # Hz at x,y
                A[I, JHz] = -1j  # mu=1

            # (stuff) = n_z B E_y
            elif(I < 4*N*M):
                y = int((I-3*M*N)/N)
                x = (I-3*M*N) - y * N

                JEz0 = 2*M*N + y*N + x
                JEz1 = 2*M*N + (y+1)*N + x

                JHx = 3*M*N + y*N + x
                JEy = M*N + y*N+x

                # derivative of Ez
                A[I,JEz0] = -ody
                if(y < M-1):
                    A[I,JEz1] = ody

                # Hx at x,y
                A[I,JHx] = -1j # mu=1

                # Setup the LHS B matrix
                B[I,JEy] = 1j*self._dir

            # (stuff) = n_z B E_x
            elif(I < 5*N*M):
                y = int((I-4*M*N)/N)
                x = (I-4*M*N) - y * N

                JEz0 = 2*M*N + y*N + x
                JEz1 = 2*M*N + y*N + (x+1)

                JHy = 4*M*N + y*N + x
                JEx = y*N + x

                # derivative of Ez
                A[I,JEz0] = odx
                if(x < N-1): A[I,JEz1] = -odx

                # Hy at x,y
                A[I,JHy] = -1j  # mu=1

                # Setup the LHS B matrix
                B[I,JEx] = -1j*self._dir

            # (stuff) = Hz (zero)
            elif(I < 6*M*N):
                y = int((I-5*M*N)/N)
                x = (I-5*M*N) - y * N

                JHy0 = 4*M*N + y*N + (x-1)
                JHy1 = 4*M*N + y*N + x

                JHx0 = 3*M*N + (y-1)*N + x
                JHx1 = 3*M*N + y*N + x

                JEz = 2*M*N + y*N + x

                # derivative of Hy
                if(x > 0): A[I, JHy0] = -odx
                A[I,JHy1] = odx

                # derivative of Hx
                if(y > 0): A[I, JHx0] = ody
                A[I, JHx1] = -ody

                # Ez
                A[I, JEz] = 1j*eps_z[y,x]

                #############################
                # enforce boundary conditions
                #############################
                if(x == 0):
                    if(bc[0] == 'E'):
                        A[I,JHy1] = 2*odx
                    else:
                        A[I,JHy1] = 0  

                if(y == 0):
                    if(bc[1] == 'E'):
                        A[I, JHx1] = -2*ody
                    else:
                        A[I, JHx1] = 0

        print(f"************************************** node {RANK} assembling A");
        print(f"eps_x.shape={eps_x.shape}")
        sys.stdout.flush()
        self._A.assemble()
        print(f"************************************** node {RANK} assembling B");
        print(f"eps_x.shape={eps_x.shape}")
        sys.stdout.flush()
        self._B.assemble()
        print(f"************************************** node {RANK} done assembling B");
        sys.stdout.flush()

    def solve(self):
        """Solve for the modes of the structure.

        Notes
        -----
        This function is run on all nodes.
        """
        if(self.verbose and NOT_PARALLEL):
            info_message('Solving...')

        # Setup the solve options. We are solving a generalized non-hermitian
        # eigenvalue problem (GNHEP)
        self._solver.setOperators(self._A, self._B)
        self._solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        self._solver.setDimensions(self.neigs, PETSc.DECIDE)
        self._solver.setTarget(self.n0) # "guess" for effective index
        self._solver.setTolerances(1e-9, 100000)
        self._solver.setFromOptions()

        # Solve Ax=nBx using SLEPc.
        # Internally, we use a direct solve (MUMPS) to handle the heavy
        # lifting.
        self._solver.solve()
        nconv = self._solver.getConverged()

        if(nconv < self.neigs):
            warning_message('%d eigenmodes were requested, however only %d ' \
                            'eigenmodes were found.' % (self.neigs, nconv), \
                            module='emopt.modes')

        # nconv can be bigger than the desired number of eigen values
        if(nconv > self.neigs):
            neigs = self.neigs
        else:
            neigs = nconv

        for i in range(neigs):
            self._neff[i] = self._solver.getEigenvalue(i)
            self._solver.getEigenvector(i, self._x[i])

    def build_gaussian_beam(self, center, mfd):
        """Directly build the fields for a gaussian beam with its waist at the mode's 
        domain, travelling in the direction according to the constructor's 'backwards'
        flag.

        center:  (x,y) in mode coordinates of waist center
        mfd:     beam waist diameter (intensity 1/e**2 of peak)
        """
        i0 = self.domain.i1   # [i,j,k] are in domain space
        j0 = self.domain.j1
        k0 = self.domain.k1
        M = self._M
        N = self._N
        if NOT_PARALLEL:
            if(self.verbose and NOT_PARALLEL):
                info_message("*************** building gaussian beam ******************")

            if(self.ndir == 'x'):     #                                                                      z,y (3D domain)
                #                                                                                       ->   y,x (2D mode)
                eps = self.eps.get_values(k0, k0+1, j0, j0+N, i0, i0+M, arr=None)[:,:,0]
                xmesh,ymesh = np.meshgrid(self.domain._y, self.domain._z)
            elif(self.ndir == 'y'):
                # TODO: FIXME
                assert(False)
            elif(self.ndir == 'z'):
                # TODO: FIXME
                assert(False)
            eta = np.sqrt(1.0/eps)  # mu=1
            rsq = (xmesh-center[0])**2 + (ymesh-center[1])**2
            Ex = np.exp(-rsq/(mfd/2)**2)
            Ey = np.zeros_like(rsq, dtype=complex)
            Ez = np.zeros_like(rsq, dtype=complex)
            Hx = np.zeros_like(rsq, dtype=complex)
            Hy = Ex / eta
            Hz = np.zeros_like(rsq, dtype=complex)
            local_x = np.concatenate((Ex,Ey,Ez,Hx,Hy,Hz)).ravel()
            self._x[0].setValues(range(len(local_x)), local_x)

        self._x[0].assemble()

        
    def __permute_field_component(self, component):
        ## Permute the field components to account for planes with non-z normal
        # directions.
        if(self.ndir == 'x'):
            if(component == FieldComponent.Ex): component = FieldComponent.Ez
            elif(component == FieldComponent.Ey): component = FieldComponent.Ex
            elif(component == FieldComponent.Ez): component = FieldComponent.Ey
            elif(component == FieldComponent.Hx): component = FieldComponent.Hz
            elif(component == FieldComponent.Hy): component = FieldComponent.Hx
            elif(component == FieldComponent.Hz): component = FieldComponent.Hy
        elif(self.ndir == 'y'):
            if(component == FieldComponent.Ex): component = FieldComponent.Ey
            elif(component == FieldComponent.Ey): component = FieldComponent.Ez
            elif(component == FieldComponent.Ez): component = FieldComponent.Ex
            elif(component == FieldComponent.Hx): component = FieldComponent.Hy
            elif(component == FieldComponent.Hy): component = FieldComponent.Hz
            elif(component == FieldComponent.Hz): component = FieldComponent.Hx
        elif(self.ndir == 'z'):
            pass # No need to permute!

        return component

    def get_field(self, i, component, permute=True, squeeze=False):
        """Get the desired raw field component of the i'th mode.

        Notes
        -----
        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeFullVector.get_field_interp`

        :func:`.ModeFullVector.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ex, Ey, Ez, Hx, Hy, Hz)
        permute : bool
            Permute the field components according to the normal direction of
            the supplied domain. Because all computation is internally
            performed assuming the mode propagates in the z direction, the
            field components need to be permuted in order to produce the
            desired result. (default = True)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            mode field.
        """
        M = self._M
        N = self._N

        # Permute components
        if(permute):
            component = self.__permute_field_component(component)

        # fields are split up among processes --> we only request data from
        # processors which store the desired field component. We do this by
        # finding out the initial and final indices of the field vector in each
        # process which stores the desired field component.
        if(component == FieldComponent.Ex):
            if(self.ib >= M*N):
                I0 = 0
                I1 = 0
            else:
                I0 = 0
                if(self.ie >= M*N):
                    I1 = M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Ey):
            if(self.ib >= 2*M*N or self.ie < M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < M*N):
                    I0 = M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 2*M*N):
                    I1 = 2*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Ez):
            if(self.ib >= 3*M*N or self.ie < 2*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 2*M*N):
                    I0 = 2*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 3*M*N):
                    I1 = 3*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hx):
            if(self.ib >= 4*M*N or self.ie < 3*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 3*M*N):
                    I0 = 3*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 4*M*N):
                    I1 = 4*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hy):
            if(self.ib >= 5*M*N or self.ie < 4*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 4*M*N):
                    I0 = 4*M*N - self.ib
                else:
                    I0 = 0
                if(self.ie >= 5*M*N):
                    I1 = 5*M*N-self.ib
                else:
                    I1 = self.ie-self.ib
        elif(component == FieldComponent.Hz):
            if(self.ie < 4*M*N):
                I0 = 0
                I1 = 0
            else:
                if(self.ib < 5*M*N):
                    I0 = 5*M*N - self.ib
                else:
                    I0 = 0

                I1 = self.ie-self.ib
        else:
            raise ValueError('Unrecongnized field componenet "%s". The allowed'
                             'field components are Ex, Ey, Ez, Hx, Hy, Hz.' % (component))

        # gather result to root process
        comm = MPI.COMM_WORLD
        x = self._x[i].getArray()[I0:I1]
        x_full = comm.gather(x, root=0)

        #scatter, x_full = PETSc.Scatter.toZero(x)
        #scatter.scatter(x, x_full, False, PETSc.Scatter.Mode.FORWARD)

        # return result on root process
        if(NOT_PARALLEL):
            x_assembled = np.concatenate(x_full)
            field = np.reshape(x_assembled, (M,N))
            if(squeeze):
                return field
            else:
                return np.reshape(field, self._fshape)
        else:
            return MathDummy()

    def get_field_interp(self, i, component, squeeze=False):
        """Get the desired interpolated field component of the i'th mode.

        In general, this function should be preferred over
        :func:`.ModeFullVector.get_field`.

        In general, you may wish to solve for more than one mode.  In order to
        get the desired mode, you must specify its index.  If you do not know
        the index but you do know the desired mode number, then
        :func:`.ModeFullVector.find_mode_index` may be used to determine the index of
        the desired mode.

        Notes
        -----
        The fields are solved for on a grid made up of compressed 2D Yee cells.
        The fields are thus interpolated at the center of this Yee cell (which
        happens to coincide with the position of Hz)

        This function only returns a non-None result on the master node. On all
        other nodes, None is returned.

        See Also
        --------
        :func:`.ModeFullVector.get_field`

        :func:`.ModeFullVector.find_mode_index`

        Parameters
        ----------
        i : int
            The index of the desired mode
        component : str
            The desired field component (Ex, Ey, Ez, Hx, Hy, Hz)

        Returns
        -------
        numpy.ndarray or None
            (Master node only) an array containing the desired component of the
            interpolated mode field.
        """
        if(self.ndir == 'x'):
            field = self.__get_field_interp_x(i, component)
        elif(self.ndir == 'y'):
            field = self.__get_field_interp_y(i, component)
        elif(self.ndir == 'z'):
            field = self.__get_field_interp_z(i, component)

        # Cant reshape in the next part unless we are rank 0
        if(not NOT_PARALLEL):
            return MathDummy()

        if(squeeze):
            return field
        else:
            return np.reshape(field, self._fshape)

    def __get_field_interp_x(self, i, component):
        ## Interpolate the fields assuming the mode propagates in the +z
        ## direction.
        ##
        ## The fields have to be interpolated in order to calculate quantities
        ## like power and energy properly. Because this mode solver has been
        ## designed in order to be compatible with the emopt.fdfd.FDFD_3D and
        ## emopt.fdtd.FDTD classes, interpolation has to be handled carefully
        ## depending on the direction that the mode propagates. In this case, we
        ## interpolate the fields about Ey, which corresponds to Ez in FDFD/FDTD.
        f_raw = self.get_field(i, component, squeeze=True)

        # zero padding is equivalent to including boundary values outside of
        # the metal boundaries. This is needed to compute the interpolated
        # values.
        f_raw = np.pad(f_raw, 1, 'constant', constant_values=0)

        # Permute components
        component = self.__permute_field_component(component)

        bc = self._bc

        # interpolate and return the fields on the rank 0 nodes. Consult the
        # All field components are interpolated onto the Ez grid.
        if(NOT_PARALLEL):
            if(component == FieldComponent.Ex): # Ex --> average along x & y
                Ex = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Ex[:,1:] += f_raw[:,0:-1]
                Ex[0:-1,1:] += f_raw[1:,0:-1]
                Ex[0:-1,:] += f_raw[1:,:]
                return Ex[1:-1, 1:-1]/4.0

            elif(component == FieldComponent.Ey): # Ey --> don't average
                return f_raw[1:-1, 1:-1]

            elif(component == FieldComponent.Ez): # Ez --> average along y
                Ez = np.copy(f_raw)

                Ez[0:-1, :] += Ez[1:, :]
                return Ez[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hx): # Hx --> don't average
                return f_raw[1:-1, 1:-1]

            elif(component == FieldComponent.Hy): # Hy --> average along x & y
                Hy = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hy[:,1:] += f_raw[:,0:-1]
                Hy[0:-1,1:] += f_raw[1:,0:-1]
                Hy[0:-1,:] += f_raw[1:,:]
                return Hy[1:-1, 1:-1]/4.0

            elif(component == FieldComponent.Hz): # Hz --> average along x
                Hz = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hz[:,1:] += f_raw[:,0:-1]
                return Hz[1:-1, 1:-1]/2.0
        else:
            return MathDummy()

    def __get_field_interp_y(self, i, component):
        ## Interpolate the fields assuming the mode propagates in the +z
        ## direction.
        ##
        ## The fields have to be interpolated in order to calculate quantities
        ## like power and energy properly. Because this mode solver has been
        ## designed in order to be compatible with the emopt.fdfd.FDFD_3D and
        ## emopt.fdtd.FDTD classes, interpolation has to be handled carefully
        ## depending on the direction that the mode propagates. In this case, we
        ## interpolate the fields about Ex, which corresponds to Ez in FDFD/FDTD.
        f_raw = self.get_field(i, component, squeeze=True)

        # zero padding is equivalent to including boundary values outside of
        # the metal boundaries. This is needed to compute the interpolated
        # values.
        f_raw = np.pad(f_raw, 1, 'constant', constant_values=0)

        # Permute components
        component = self.__permute_field_component(component)

        bc = self._bc

        # interpolate and return the fields on the rank 0 nodes. Consult the
        # All field components are interpolated onto the Ez grid.
        if(NOT_PARALLEL):
            if(component == FieldComponent.Ex): # Ex --> don't average
                return fraw[1:-1, 1:-1]

            elif(component == FieldComponent.Ey): # Ey --> average x & y
                Ey = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Ey[:, 0:-1] += Ey[:, 1:]
                Ey[1:, :] += Ey[0:-1, :]
                Ey[1:, 0:-1] += Ey[0:-1, 1:]
                return Ey[1:-1, 1:-1]/4.0

            elif(component == FieldComponent.Ez): # Ez --> average along x
                Ez = np.copy(f_raw)

                Ez[:, 0:-1] += Ez[:, 1:]
                return Ez[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hx): # Hx --> average along x & y
                Hx = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Hx[:, 0:-1]  += Hx[:, 1:]
                Hx[1:, :]    += Hx[0:-1, :]
                Hx[1:, 0:-1] += Hx[0:-1, 1:]
                return Hx[1:-1, 1:-1]/4.0

            elif(component == FieldComponent.Hy): # Hy --> don't average
                return f_raw

            elif(component == FieldComponent.Hz): # Hz --> average along x
                Hz = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Hz[1:,:] += f_raw[0:-1, :]
                return Hz[1:-1, 1:-1]/2.0
        else:
            return MathDummy()


    def __get_field_interp_z(self, i, component):
        ## Interpolate the fields assuming the mode propagates in the +z
        ## direction.
        ##
        ## The fields have to be interpolated in order to calculate quantities
        ## like power and energy properly. Because this mode solver has been
        ## designed in order to be compatible with the emopt.fdfd.FDFD_3D and
        ## emopt.fdtd.FDTD classes, interpolation has to be handled carefully
        ## depending on the direction that the mode propagates. In this case, we
        ## interpolate the fields about Ez, which corresponds to Ez in FDFD/FDTD.
        f_raw = self.get_field(i, component, squeeze=True)

        # zero padding is equivalent to including boundary values outside of
        # the metal boundaries. This is needed to compute the interpolated
        # values.
        f_raw = np.pad(f_raw, 1, 'constant', constant_values=0)

        # Permute components
        component = self.__permute_field_component(component)

        bc = self._bc

        # interpolate and return the fields on the rank 0 nodes. Consult the
        # All field components are interpolated onto the Ez grid.
        if(NOT_PARALLEL):
            if(component == FieldComponent.Ex): # Ex --> average along x
                Ex = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Ex[:,1:] += f_raw[:,0:-1]
                return Ex[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Ey): # Ey --> average along y
                Ey = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Ey[1:,:] += f_raw[0:-1,:]
                return Ey[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Ez): # Ez --> dont average
                return f_raw[1:-1, 1:-1]

            elif(component == FieldComponent.Hx): # Hx --> average along y
                Hx = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]

                Hx[1:,:] += f_raw[0:-1,:]
                return Hx[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hy): # Hy --> average along x
                Hy = np.copy(f_raw)
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hy[:,1:] += f_raw[:,0:-1]
                return Hy[1:-1, 1:-1]/2.0

            elif(component == FieldComponent.Hz): # Hz --> average along x & y
                Hz = np.copy(f_raw)
                if(bc[1] == 'E'): f_raw[0,:] = -1*f_raw[1,:]
                elif(bc[1] == 'H'): f_raw[0,:] = f_raw[1,:]
                if(bc[0] == 'E'): f_raw[:,0] = -1*f_raw[:,1]
                elif(bc[0] == 'H'): f_raw[:,0] = f_raw[:,1]

                Hz[:,1:] += f_raw[:,0:-1]
                Hz[1:,:] += f_raw[0:-1,:]
                Hz[1:, 1:] += f_raw[0:-1,0:-1]
                return Hz[1:-1, 1:-1]/4.0
        else:
            return MathDummy()

    def component_energy(self, i):
        """Get the fraction of energy stored in each field component.

        Parameters
        ----------
        i : int
            The index of the mode to analyze

        Returns
        -------
        [float, float, float, float, float, float]
            The list of energy fractions corresponding to Ex, Ey, Ez, Hx, Hy,
            Hz
        """
        eps = self.eps.get_values_in(self.domain)

        Ex = self.get_field(i, FieldComponent.Ex, squeeze=True)
        WEx = np.sum(eps.real*np.abs(Ex)**2)
        del Ex

        Ey = self.get_field(i, FieldComponent.Ey, squeeze=True)
        WEy = np.sum(eps.real*np.abs(Ey)**2)
        del Ey

        Ez = self.get_field(i, FieldComponent.Ez, squeeze=True)
        WEz = np.sum(eps.real*np.abs(Ez)**2)
        del Ez

        Hx = self.get_field(i, FieldComponent.Hx, squeeze=True)
        WHx = np.sum(np.abs(Hx)**2)  # mu=1
        del Hx

        Hy = self.get_field(i, FieldComponent.Hy, squeeze=True)
        WHy = np.sum(np.abs(Hy)**2)  # mu=1
        del Hy

        Hz = self.get_field(i, FieldComponent.Hz, squeeze=True)
        WHz = np.sum(np.abs(Hz)**2)  # mu=1
        del Hz

        Wtot = WEx+WEy+WEz+WHx+WHy+WHz
        return [WEx/Wtot, WEy/Wtot, WEz/Wtot,
                WHx/Wtot, WHy/Wtot, WHz/Wtot]

    def get_mode_number(self, i):
        """Determine the mode number.

        This is based on the number of phase crossings.

        Todo
        ----
        Implement this function...

        Parameters
        ----------
        i : int
            The index of the mode to analyze.

        Returns
        -------
        int, int
            The numbers X and Y of the mode.
        """
        pass

    def find_mode_index(self, P, Q):
        """Find the index of the index of the mode with X horizontal phase
        transitions and Y vertical phase transitions.

        Parameters
        ----------
        P : int
            The number of horizontal phase transitions
        Q : int
            The number of vertical phase transitions

        Returns
        -------
        int
            The index of the mode with the desired number.
        """
        for i in range(self.neigs):
            p, q = self.get_mode_number(i)
            if(p == P and q == Q):
                return i

        warning_message('Desired mode number was not found.', 'emopt.modes')
        return 0

    def get_source(self, i, dx, dy, dz):
        """Get the source current distribution for the i'th mode.

        Notes
        -----
        1) Source calculations are only supported for Material3D structures.
        2) The sources are computed assuming the mode is propagating in the z
        direction. In order to support modes propagating in different
        directions, the spatial coordinates are permuted. Fortunately, the
        underlying dislocated grids are invariant under this transformation.
        This is only necessary for calculating the material cross-sections.

        TODO
        ----
        Implement in parallelized manner.

        Parameters
        ----------
        i : int
            Index of the mode for which the corresponding current sources are
            desired.
        dx : float
            Grid spacing along x direction.
        dy : float
            Grid spacing along y direction
        dz : float
            Grid spacing along z direction

        Returns
        -------
        tuple of numpy.ndarray
            (On master node) The tuple (Jx, Jy, Jz, Mx, My, Mz) containing arrays of the
            source distributions.
        """

        neff = self.neff[i]
        bc = self._bc

        # setup arrays for storing the calculated current sources
        # These are only computed and stored on the rank 0 process 
        if(NOT_PARALLEL):
            Jx = np.zeros([self._M, self._N], dtype=np.complex128)
            Jy = np.zeros([self._M, self._N], dtype=np.complex128)
            Jz = np.zeros([self._M, self._N], dtype=np.complex128)
            Mx = np.zeros([self._M, self._N], dtype=np.complex128)
            My = np.zeros([self._M, self._N], dtype=np.complex128)
            Mz = np.zeros([self._M, self._N], dtype=np.complex128)

            eps = np.zeros([self._M, self._N], dtype=np.complex128)
        else:
            Jx = None; Jy = None; Jz = None
            Mx = None; My = None; Mz = None

        # Handle coordinate permutations. This allows us to support launching
        # modes in any cartesian direction. At the end, we will once again
        # permute the calculated current density components to match the users
        # supplied coordinate system
        if(self.ndir == 'x'):
            mydx = dy
            mydy = dz
            mydz = dx
        elif(self.ndir == 'y'):
            mydx = dz
            mydy = dx
            mydz = dy
        elif(self.ndir == 'z'):
            mydx = dx
            mydy = dy
            mydz = dz

        # phase factor
        dx = mydx/self.R
        dy = mydy/self.R
        dz = mydz/self.R
        ekz = np.exp(1j*self._dir*neff*dz/2)

        # Get field data
        Ex = self.get_field(i, FieldComponent.Ex, permute=False, squeeze=True)
        Ey = self.get_field(i, FieldComponent.Ey, permute=False, squeeze=True)
        Ez = self.get_field(i, FieldComponent.Ez, permute=False, squeeze=True)
        Hx = self.get_field(i, FieldComponent.Hx, permute=False, squeeze=True)
        Hy = self.get_field(i, FieldComponent.Hy, permute=False, squeeze=True)
        Hz = self.get_field(i, FieldComponent.Hz, permute=False, squeeze=True)

        # normalize power to ~1.0 -- not really necessary
        S = 0.5*mydx*mydy*np.sum(Ex*np.conj(Hy)-Ey*np.conj(Hx))
        Ex = Ex/np.sqrt(S); Ey = Ey/np.sqrt(S); Ez = Ez/np.sqrt(S)
        Hx = Hx/np.sqrt(S); Hy = Hy/np.sqrt(S); Hz = Hz/np.sqrt(S)

        if(NOT_PARALLEL):
            ## Calculate contribution of Ex
            Ex = np.pad(Ex, 1, 'constant', constant_values=0)
            My += Ex[1:-1, 1:-1]*ekz/dz

            ## Calculate contribution of Ey
            Ey = np.pad(Ey, 1, 'constant', constant_values=0)
            Mx += -Ey[1:-1, 1:-1]*ekz/dz

            ## Calculate contribution of Ez
            Ez = np.pad(Ez, 1, 'constant', constant_values=0)
            eps = self.eps.get_values_in(self.domain, squeeze=True)
            Jz += 1j*eps*Ez[1:-1, 1:-1]
            Mx += (Ez[2:, 1:-1] - Ez[1:-1, 1:-1])/dy
            My += -1*(Ez[1:-1, 2:] - Ez[1:-1, 1:-1])/dx

            ## Calculate contribution of Hx
            Hx = np.pad(Hx, 1, 'constant', constant_values=0)

            # Handle boundary conditions
            if(bc[1] == 'E'): Hx[0,:] = -1*Hx[1,:]
            elif(bc[1] == 'H'): Hx[0,:] = Hx[1,:]

            Jy += Hx[1:-1, 1:-1]/dz
            Jz += -1*(Hx[1:-1, 1:-1] - Hx[0:-2, 1:-1])/dy
            Mx += -1j*Hx[1:-1, 1:-1]  # mu=1

            ## Calculate contribution of Hy
            Hy = np.pad(Hy, 1, 'constant', constant_values=0)

            # Handle boundary conditions
            if(bc[0] == 'E'): Hy[:,0] = -1*Hy[:,1]
            elif(bc[0] == 'H'): Hy[:,0] = Hy[:,1]

            Jx += -Hy[1:-1, 1:-1]/dz
            Jz += (Hy[1:-1, 1:-1] - Hy[1:-1, 0:-2])/dx
            My += -1j*Hy[1:-1, 1:-1]  # mu=1

            ## Calculate contribution of Hz
            # There isn't one

            # reshape the output to make things seemless
            Jx = np.reshape(Jx, self.domain.shape)
            Jy = np.reshape(Jy, self.domain.shape)
            Jz = np.reshape(Jz, self.domain.shape)
            Mx = np.reshape(Mx, self.domain.shape)
            My = np.reshape(My, self.domain.shape)
            Mz = np.reshape(Mz, self.domain.shape)

        # permute (if necessary) and return the results
        if(self.ndir == 'x'):
            return Jz, Jx, Jy, Mz, Mx, My
        elif(self.ndir == 'y'):
            return Jy, Jz, Jx, My, Mz, Mx
        elif(self.ndir == 'z'):
            return Jx, Jy, Jz, Mx, My, Mz

# Done modes.py!
