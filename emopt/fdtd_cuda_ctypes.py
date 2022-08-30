"""Provides a ctypes interface between C++ FDTD_cuda library and python."""

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

so_path = ''.join([dir_path, '/FDTD_cuda.so'])
libFDTD = cdll.LoadLibrary(so_path)

# useful defs
c_complex_p = ndpointer(np.complex128, ndim=1, flags='C')
c_double_p = ndpointer(np.double, ndim=1, flags='C')

# class for c++ FDTD structure
class CPP_COMPLEX128(Structure):
    _fields_ = [("real", c_double),
                ("imag", c_double)]

class CPP_FDTD(Structure):
    _fields_ = [("Ex", POINTER(c_double)),
                ("Ey", POINTER(c_double)),
                ("Ez", POINTER(c_double)),
                ("Hx", POINTER(c_double)),
                ("Hy", POINTER(c_double)),
                ("Hz", POINTER(c_double)),
                ("eps_x", POINTER(c_double)),
                ("eps_y", POINTER(c_double)),
                ("eps_z", POINTER(c_double))]

#######################################################
# ctypes interface definition
#######################################################
libFDTD.FDTD_new.argtypes = [c_int, c_int, c_int]
libFDTD.FDTD_new.restype = POINTER(CPP_FDTD)

libFDTD.FDTD_set_wavelength.argtypes = [c_void_p, c_double]
libFDTD.FDTD_set_wavelength.restype = None

libFDTD.FDTD_set_physical_dims.argtypes = [c_void_p,
                                           c_double, c_double, c_double,
                                           c_double, c_double, c_double]
libFDTD.FDTD_set_physical_dims.restype = None

libFDTD.FDTD_set_dt.argtypes = [c_void_p, c_double]
libFDTD.FDTD_set_dt.restype = None

libFDTD.FDTD_update.argtypes = [c_void_p, c_double, c_int]
libFDTD.FDTD_update.restype = None

libFDTD.FDTD_set_pml_widths.argtypes = [c_void_p, c_int, c_int,
                                                  c_int, c_int,
                                                  c_int, c_int]
libFDTD.FDTD_set_pml_widths.restype = None

libFDTD.FDTD_set_pml_properties.argtypes = [c_void_p, c_double, c_double,
                                                      c_double, c_double]
libFDTD.FDTD_set_pml_properties.restype = None

libFDTD.FDTD_build_pml.argtypes = [c_void_p]
libFDTD.FDTD_build_pml.argtypes = None

libFDTD.FDTD_set_t0_arrays.argtypes = [c_void_p,
                                       c_complex_p, c_complex_p, c_complex_p,
                                       c_complex_p, c_complex_p, c_complex_p]
libFDTD.FDTD_set_t0_arrays.restype = None

libFDTD.FDTD_set_t1_arrays.argtypes = [c_void_p,
                                       c_complex_p, c_complex_p, c_complex_p,
                                       c_complex_p, c_complex_p, c_complex_p]
libFDTD.FDTD_set_t1_arrays.restype = None

libFDTD.FDTD_calc_phase_3T.argtypes = [c_double, c_double, c_double,
                                       c_double, c_double, c_double]
libFDTD.FDTD_calc_phase_3T.restype = c_double

libFDTD.FDTD_calc_amplitude_3T.argtypes = [c_double, c_double, c_double,
                                           c_double, c_double, c_double,
                                           c_double]
libFDTD.FDTD_calc_amplitude_3T.restype = c_double

libFDTD.FDTD_capture_t0_fields.argtypes = [c_void_p]
libFDTD.FDTD_capture_t0_fields.restype = None

libFDTD.FDTD_capture_t1_fields.argtypes = [c_void_p]
libFDTD.FDTD_capture_t1_fields.restype = None

libFDTD.FDTD_calc_complex_fields_3T.argtypes = [c_void_p, c_double, c_double,
                                                c_double]
libFDTD.FDTD_calc_complex_fields_3T.restype = None

libFDTD.FDTD_add_source.argtypes = [c_void_p,
                                    c_complex_p, c_complex_p, c_complex_p,
                                    c_complex_p, c_complex_p, c_complex_p,
                                    c_int, c_int, c_int,
                                    c_int, c_int, c_int,
                                    c_bool]
libFDTD.FDTD_add_source.restype = None

libFDTD.FDTD_clear_sources.argtypes = [c_void_p]
libFDTD.FDTD_clear_sources.restype = None

libFDTD.FDTD_set_source_properties.argtypes = [c_void_p, c_double, c_double]
libFDTD.FDTD_set_source_properties.restype = None

libFDTD.FDTD_src_func_t.argtypes = [c_void_p, c_double, c_double]
libFDTD.FDTD_src_func_t.restype = c_double

libFDTD.FDTD_set_bc.argtypes = [c_void_p, c_char_p]
libFDTD.FDTD_set_bc.restype = None

