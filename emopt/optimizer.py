"""
This :class:`.Optimizer` class provides a simple wrapper around scipy.minimize.optimize
which allows you optimize an electromagnetic structure given an arbitrary
(user-defined) set of design parameters. The :class:`.Optimizer` class
minimizes a figure of merit defined in an
:class:`emopt.adjoint_method.AdjointMethod` object and takes advantage of the
gradient computed by the supplied
:class:`emopt.adjoint_method.AdjointMethod` object.


Examples
--------
The :class:`.Optimizer` is used approximately as follows:

.. doctest::
    import emopt.fdfd
    import emopt.adjoint_method
    import emopt.optimizer

    # Setup the simulation object
    sim = ...

    # Define a custom adjoint method class and instantiate it
    am = MyAdjointMethod(...)

    # Define a callback function
    def my_callback(params, ...):
        ...

    callback_func = lambda params : my_callback(params, other_inputs)

    # Specify initial guess for the design parameters
    design_params = ....

    # Create the optimizer object
    opt = Optimizer(sim, am, design_params, callback=callback_func)

    # run the optimization
    opt.run()
"""
from __future__ import absolute_import

from builtins import object
from .misc import info_message, warning_message, error_message

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class Optimizer(object):
    """Handles the optimization of an electromagnetic structure.

    Given a set of design variables, a figure of merit, and a gradient, any
    electromagnetic structure can be optimized regardless of the underlying
    implementations of the simulator. With these quantities defined, it is in
    theory quite easy to run an optimization.

    Currently, the optimization code
    is based on scipy.optimize.minimize, which is not parallel/MPI-compatible.
    As is such, this class manages the interface between the sequential scipy
    calls and the parallel components of EMOpt (like running simulations).

    Fully parallelizing the optimization code should be possible using petsc.
    However, parallelizing the gradient computation is quite tricky for the
    most general case of arbitrary design variables.  The process of computing
    gradients in paralle is significantly simplified when the material in each
    grid cell is an independent design variable (i.e. grayscale topology
    optimization). This type of problem, however, is not the core goal of
    EMOpt. This functionality may be added in the future.

    Parameters
    ----------
    sim : emopt.fdfd.FDFD
        Simulation object
    am : emopt.adjoint_method.AdjointMethod
        Object containing problem-specific implementation of AdjointMethod
    p0 : numpy.ndarray or list
        Initial guess for design parameters of system
    callback_func : function
        Function which accepts the current design variables as the only
        argument. This function is called after each iteration of the
        optimization.  By default, no callback function is used.
    opt_method : str
        Optimization method to use.  The recommended options are: CG, BFGS,
        L-BFGS-B, TNC, SLSQP. (default='BFGS')
    Nmax : int
        Maximum number of interations of optimization method before process is
        terminated. (default=1000)
    tol : float
        Minimum change in figure of merit below which the optimization will
        complete. (default=1e-5)
    bounds : list of tuples
        List of tuples containing two floats which specify the lower and upper
        bounds on each design variable.  This is not compatible with all
        optimization methods.  Consult the scipy.optimize.minimize
        documentation for details. (default=None)

    Attributes
    ----------
    am : emopt.adjoint_method.AdjointMethod
        The adjoint method object for calculating FOM and gradient
    p0 : numpy.ndarray or list
        The initial guess for design variables
    callback : function
        The callback function to call after each optimization iteration.
    Nmax : int
        The maximum number of iterations
    tol : float
        The minimum change in figure of merit under which optimization
        terminates
    bounds : list of 2-tuple
        The list of bounds to put on design variables in the formate (minv, maxv)

    Methods
    -------
    run(self)
        Run the optimization.
    run_sequence(self, sim, am)
        Define the sequence of figure of merit and gradient calls for the
        optimization.
    """

    def __init__(self, am, p0, callback_func=None, opt_method='BFGS', \
                 Nmax=1000, tol=1e-5, bounds=None, scipy_verbose=True):
        self.am = am

        self.p0 = p0

        if(callback_func is None):
            self.callback = lambda p : None
        else:
            self.callback = callback_func

        self.opt_method = opt_method
        self.Nmax = Nmax
        self.tol = tol
        self.bounds = bounds
        self.scipy_verbose = scipy_verbose

    def run(self):
        """Run the optimization.

        Returns
        -------
        float
            The final figure of merit
        numpy.array
            The optimized design parameters
        """

        self.am.fom(self.p0)
        self.callback(self.p0)
        result = minimize(self.am.fom, self.p0, method=self.opt_method,
                          jac=self.am.gradient, callback=self.callback,
                          tol=self.tol, bounds=self.bounds,
                          options={'maxiter':self.Nmax, \
                                   'disp':self.scipy_verbose})

        return result.fun, result.x

