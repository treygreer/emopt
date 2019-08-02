"""Define an interface for accessing the grid library written in c++."""

from ctypes import *
import os
import numpy as np
from numpy.ctypeslib import ndpointer

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

dir_path = os.path.dirname(os.path.realpath(__file__))

so_path = ''.join([dir_path, '/Grid_cuda.so'])
libGrid = cdll.LoadLibrary(so_path)

# some useful definitions
c_complex_2D_p = ndpointer(np.complex128, ndim=2, flags='C')
c_complex_1D_p = ndpointer(np.complex128, ndim=1, flags='C')
c_double_p = ndpointer(np.double, ndim=1, flags='C')

####################################################################################
# Polygon configuration
####################################################################################
libGrid.Polygon_new.argtypes = [c_double_p, c_double_p, c_int, c_double, c_double]
libGrid.Polygon_new.restype = c_void_p
libGrid.Polygon_delete.argtypes = [c_void_p]
libGrid.Polygon_delete.restype = None

#####################################################################################
# Material3D configuration 
#####################################################################################
libGrid.Material3D_get_value.argtypes = [c_void_p, c_complex_1D_p, c_double,
                                         c_double, c_double]
libGrid.Material3D_get_value.restype = None
libGrid.Material3D_get_values.argtypes = [c_void_p, c_complex_1D_p, c_int, c_int, c_int, c_int,
                                          c_int, c_int, c_double, c_double,
                                          c_double]
libGrid.Material3D_get_values.restype = None

####################################################################################
# ConstantMaterial3D configuration
####################################################################################
libGrid.ConstantMaterial3D_new.argtypes = [c_double, c_double]
libGrid.ConstantMaterial3D_new.restype = c_void_p

####################################################################################
# StructuredMaterial3D configuration
####################################################################################
libGrid.StructuredMaterial3D_new.argtypes = [c_double, c_double, c_double,
                                             c_double, c_double, c_double]
libGrid.StructuredMaterial3D_new.restype = c_void_p
libGrid.StructuredMaterial3D_delete.argtypes = [c_void_p]
libGrid.StructuredMaterial3D_delete.restype = None
libGrid.StructuredMaterial3D_add_polygon.argtypes = [c_void_p, c_void_p,
                                                     c_double, c_double]
libGrid.StructuredMaterial3D_add_polygon.restype = None


