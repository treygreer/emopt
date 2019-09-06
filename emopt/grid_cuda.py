"""Python interface for grid smoothing code.

Grid smoothing refers to the process of mapping a continuously-defined shapes with
sharp boundaries to a rectangular grid in a continuous manner.  This process results
in a rectangular grid whose material distribution matches the defined shape
except at the boundaries where the effective material value is between the
material value within the bound shape and that of the surrounding medium, hence
making the boundaries appear as if they have been 'smoothed' out.

Ensuring that this mapping from 'real' space to the discretized rectangular
grid is continuous (i.e. small changes in the underlying boundaries produce
small changes in the material distribution of the grid) is very important to
sensitivity analysis. Gradients computed using an adjoint method, in
particular, will be inaccurate if changes to the material distribution occur in
discrete jumps which are too large.

Grid smoothing can be accomplished in a variety of ways.  The implementation
here is very general for sets of shapes which do not require changes in
topology (creation and elimination of holes, etc). It relies on representing
boundaries using polygons and then computing the smoothed grid by applying a
series of boolean subtraction operations (in c++).
"""
from __future__ import absolute_import

from builtins import range
from builtins import object
from .grid_cuda_ctypes import libGrid
import numpy as np
import scipy
from ctypes import c_int, c_double

from .misc import DomainCoordinates
from .misc import warning_message

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class GridMaterial3D(object):

    def __init__(self, X, Y, Z, Nx, Ny, Nz, grid):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.grid = grid

    def get_value(self, k, j ,i):
        return self.grid[i,j,k]


class PolyMat(object):

    def __init__(self, xs, ys, material_value=1.0):
        assert(len(xs) == len(ys))
        self._object = libGrid.PolyMat_new(np.array(xs, dtype=np.float64),
                                           np.array(ys, dtype=np.float64),
                                           len(xs),
                                           material_value.real, material_value.imag)
    def __del__(self):
        libGrid.PolyMat_delete(self._object)

class Material3D(object):
    """Define a general interface for 3D Material distributions.

    Methods
    -------
    get_value(self, x, y, z)
        Get the value of the material distribution at (x,y,z)
    get_values(self, k1, k2, j1, j2, i1, i2, arr=None)
        Get the values of the material distribution within a set of array
        indicesa set of array indices.
    get_values_on(self, domain)
        Get the values of the material distribution within a domain.
    """

    def __init__(self):
        self._object = None

    def get_value(self, x, y, z):
        """Get the value of the material distribution at (x,y,z).

        Parameters
        ----------
        x : int or float
            The (fractional) x index.
        y : int or float
            The (fractional) y index.
        z : int or float
            The (fractional) z index

        Returns
        -------
        complex128
            The complex material value at the desired location.
        """
        value = np.array([0], dtype=np.complex128)
        libGrid.Material3D_get_value(self._object, value, x, y, z)

        return value

    def get_values(self, k1, k2, j1, j2, i1, i2, sx=0, sy=0, sz=0, arr=None,
                   reshape=True):
        """Get the values of the material distribution within a set of array
        indicesa set of array indices.

        Parameters
        ----------
        k1 : int
            The lower integer bound on x of the desired region
        k2 : int
            The upper integer bound on x of the desired region
        j1 : int
            The lower integer bound on y of the desired region
        j2 : int
            The upper integer bound on y of the desired region
        i1 : int
            The lower integer bound on y of the desired region
        i2 : int
            The upper integer bound on y of the desired region
        arr : numpy.ndarray (optional)
            The array with dimension (m2-m1)x(n2-n1) with type np.complex128
            which will store the retrieved material distribution. If None, a
            new array will be created. (optional = None)

        Returns
        -------
        numpy.ndarray
            The retrieved complex material distribution.
        """
        Nx = k2-k1
        Ny = j2-j1
        Nz = i2-i1

        if(type(arr) == type(None)):
            arr = np.zeros(Nx*Ny*Nz, dtype=np.complex128)
        else:
            arr = np.ravel(arr)

        libGrid.Material3D_get_values(self._object, arr, k1, k2, j1, j2, i1,
                                      i2, sx, sy, sz)

        # This might result in an expensive copy operation, unfortunately
        if(reshape):
            arr = np.reshape(arr, [Nz, Ny, Nx])

        return arr

    def get_values_in(self, domain, sx=0, sy=0, sz=0, arr=None, squeeze=False):
        """Get the values of the material distribution within a domain.

        Parameters
        ----------
        domain : emopt.misc.DomainCoordinates
            The domain in which the material distribution is retrieved.
        sx : float (optional)
            The partial index shift in the x direction
        sy : float (optional)
            The partial index shift in the y direction
        sz : float (optional)
            The partial index shift in the z direction
        arr : np.ndarray (optional)
            The array in which the retrieved material distribution is stored.
            If None, a new array is instantiated (default = None)
        squeeze : bool (optional)
            If True, eliminate length-1 dimensions from the resulting array.
            This only affects 1D and 2D domains. (default = False)

        Returns
        -------
        numpy.ndarray
            The retrieved material distribution which lies in the domain.
        """
        i1 = domain.i.start
        i2 = domain.i.stop
        j1 = domain.j.start
        j2 = domain.j.stop
        k1 = domain.k.start
        k2 = domain.k.stop

        vals = self.get_values(k1, k2, j1, j2, i1, i2, sx, sy, sz, arr)

        if(squeeze): return np.squeeze(vals)
        else: return vals

class ConstantMaterial3D(Material3D):
    """A uniform constant 3D material.

    Parameters
    ----------
    value : complex
        The constant material value.

    Attributes
    ----------
    material_value : complex
        The constant material value
    """
    def __init__(self, value):
        self._material_value = value
        self._object = libGrid.ConstantMaterial3D_new(value.real, value.imag)

    def __del__(self):
        libGrid.ConstantMaterial3D_delete(self._object)

    @property
    def material_value(self):
        return self._material_value

    @material_value.setter
    def material_value(self, new_value):
        libGrid.ConstantMaterial3D_set_material(self._object,
                                              new_value.real,
                                              new_value.imag)
        self._material_value = new_value

class StructuredMaterial3D(Material3D):
    """Create a 3D material consisting of one or more polygons
    with thickness along z.

    Currently StructuredMaterial3D only supports layered slab structures.

    Notes
    -----
    When used for defining the material distribution for a simulation, the
    dimensions supplied will typically match the dimensions of the simulation.

    Parameters
    ----------
    [X, Y, Z]: floats
        The x,y,z width of the underlying grid.
    [dx, dy, dz]: floats
        The grid spacing of the underlying grid
    polymat_zspans: list of [polymat, zmin, zmax] lists

    """
    def __init__(self, XYZ, dxdydz, polymat_zspans):
        self._object = libGrid.StructuredMaterial3D_new(XYZ[0], XYZ[1], XYZ[2],
                                                        dxdydz[0], dxdydz[1], dxdydz[2])
        self._polygons = []
        for (polymat, zmin, zmax) in polymat_zspans:
            self._polygons.append(polymat)
            libGrid.StructuredMaterial3D_add_polymat(self._object, polymat._object,
                                                     zmin, zmax)

    def __del__(self):
        libGrid.StructuredMaterial3D_delete(self._object)

