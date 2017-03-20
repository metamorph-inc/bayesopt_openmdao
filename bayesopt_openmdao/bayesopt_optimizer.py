#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import bayesopt

from six import itervalues, iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from collections import OrderedDict

class BayesoptOptimizer(Driver):
    def __init__(self):
        """Initialize the ScipyOptimizer."""

        super(BayesoptOptimizer, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = False

        # User Options
        # self.options.add_option('optimizer', 'SLSQP', values=_optimizers,
        #                         desc='Name of optimizer to use')
        self.options.add_option('n_iterations', 200, lower=0,
                                desc='Number of iterations.')
        self.options.add_option('n_inner_iterations', 500, lower=0,
                                desc='Number of iterations.')
        self.options.add_option('n_iter_relearn', 5, lower=0,
                                desc='Number of iterations.')
        self.options.add_option('n_init_samples', 2, lower=0,
                                desc='Number of iterations.')
        self.options.add_option('surr_name', "sGaussianProcess",
                                desc='Number of iterations.')
        self.options.add_option('noise', 1e-6, lower=0,
                                desc='Noise')
        self.options.add_option('disp', True,
                                desc='Set to False to prevent printing of Scipy '
                                'convergence messages')

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()

        self.metadata = None
        self._problem = None
        self.result = None
        self.exit_flag = 0
        self.grad_cache = None
        self.con_cache = None
        self.con_idx = OrderedDict()
        self.cons = None
        self.objs = None

    def _setup(self):
        super(BayesoptOptimizer, self)._setup()

    def run(self, problem):
        """Optimize the problem using your choice of Scipy optimizer.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        # Metadata Setup
        self.metadata = create_local_meta(None, "BayesOpt")
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        # Initial Run
        with problem.root._dircontext:
            problem.root.solve_nonlinear(metadata=self.metadata)

        pmeta = self.get_desvar_metadata()
        self.params = list(pmeta)
        self.objs = list(self.get_objectives())
        con_meta = self.get_constraint_metadata()
        self.cons = list(con_meta)
        self.con_cache = self.get_constraints()

        self.opt_settings['disp'] = self.options['disp']

        bopt_params = {}
        bopt_params['n_iterations'] = self.options['n_iterations']
        bopt_params['n_inner_iterations'] = self.options['n_inner_iterations']
        bopt_params['n_iter_relearn'] = self.options['n_iter_relearn']
        bopt_params['n_init_samples'] = self.options['n_init_samples']
        bopt_params['noise'] = self.options['noise']
        bopt_params['surr_name'] = self.options['surr_name']


        # Size Problem
        nparam = 0
        for param in itervalues(pmeta):
            nparam += param['size']
        x_init = np.empty(nparam)
        i = 0

        # Initial Parameters
        lower_bounds = []
        upper_bounds = []

        for name, val in iteritems(self.get_desvars()):
            size = pmeta[name]['size']
            x_init[i:i+size] = val
            i += size

            # Bounds if our optimizer supports them
            meta_low = pmeta[name]['lower']
            meta_high = pmeta[name]['upper']
            for j in range(0, size):

                if isinstance(meta_low, np.ndarray):
                    p_low = meta_low[j]
                else:
                    p_low = meta_low

                if isinstance(meta_high, np.ndarray):
                    p_high = meta_high[j]
                else:
                    p_high = meta_high

                lower_bounds.append(p_low)
                upper_bounds.append(p_high)

        # optimize
        self._problem = problem

        min_value, xout, error = bayesopt.optimize(self._objfunc, len(lower_bounds), np.asarray(lower_bounds), np.asarray(upper_bounds), bopt_params)

        # Run one more iteration, at the computed minimum
        self._objfunc(xout)

        self._problem = None
        self.result = min_value # TODO: what is this supposed to return?
        self.exit_flag = 1 # TODO: handle optimization failure?

        if self.options['disp']:
            print('Optimization Complete')
            print('-'*35)

    def _objfunc(self, x_new):
        """ Function that evaluates and returns the objective function. Model
        is executed here.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """

        system = self.root
        metadata = self.metadata

        # Pass in new parameters
        i = 0
        for name, meta in self.get_desvar_metadata().items():
            size = meta['size']
            self.set_desvar(name, x_new[i:i+size])
            i += size

        self.iter_count += 1
        update_local_meta(metadata, (self.iter_count,))

        with system._dircontext:
            system.solve_nonlinear(metadata=metadata)

        # Get the objective function evaluations
        for name, obj in self.get_objectives().items():
            f_new = obj
            break

        self.con_cache = self.get_constraints()

        # Record after getting obj and constraints to assure it has been
        # gathered in MPI.
        self.recorders.record_iteration(system, metadata)

        #print("Functions calculated")
        #print(x_new)
        #print(f_new)

        return f_new

    def _confunc(self, x_new, name, idx):
        """ Function that returns the value of the constraint function
        requested in args. Note that this function is called for each
        constraint, so the model is only run when the objective is evaluated.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

        cons = self.con_cache
        meta = self._cons[name]

        # Equality constraints
        bound = meta['equals']
        if bound is not None:
            if isinstance(bound, np.ndarray):
                bound = bound[idx]
            return bound - cons[name][idx]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        upper = meta['upper']
        lower = meta['lower']
        if lower is None or dbl_side:
            if isinstance(upper, np.ndarray):
                upper = upper[idx]
            return upper - cons[name][idx]
        else:
            if isinstance(lower, np.ndarray):
                lower = lower[idx]
            return cons[name][idx] - lower

    def _gradfunc(self, x_new):
        """ Function that evaluates and returns the objective function.
        Gradients for the constraints are also calculated and cached here.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to parameter array.
        """

        grad = self.calc_gradient(self.params, self.objs+self.cons,
                                  return_format='array')
        self.grad_cache = grad

        #print("Gradients calculated")
        #print(x_new)
        #print(grad[0, :])

        return grad[0, :]

    def _congradfunc(self, x_new, name, idx):
        """ Function that returns the cached gradient of the constraint
        function. Note, scipy calls the constraints one at a time, so the
        gradient is cached when the objective gradient is called.

        Args
        ----
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Gradient of the constraint function wrt all params.
        """

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

        grad = self.grad_cache
        meta = self._cons[name]
        grad_idx = self.con_idx[name] + idx + 1

        #print("Constraint Gradient returned")
        #print(x_new)
        #print(name, idx, grad[grad_idx, :])

        # Equality constraints
        if meta['equals'] is not None:
            return -grad[grad_idx, :]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        if meta['lower'] is None or dbl_side:
            return -grad[grad_idx, :]
        else:
            return grad[grad_idx, :]
