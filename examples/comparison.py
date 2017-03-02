#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import csv
import time

from bayesopt_openmdao.bayesopt_optimizer import BayesoptOptimizer

from openmdao.api import IndepVarComp, Component, Problem, Group, ScipyOptimizer

_DIMENSIONS = 4

class RosenbrockMultiDim(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self, dimensions):
        super(RosenbrockMultiDim, self).__init__()

        self._dimensions = dimensions

        for i in range(self._dimensions):
            self.add_param('x{0}'.format(i), val=0.0)

        self.add_output('f', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        """ See second multidimensional generalization at https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        r_sum = 0.0

        for i in range(self._dimensions - 1):
            x_i = params['x{0}'.format(i)]
            x_i1 = params['x{0}'.format(i + 1)]

            component = 100.0 * ((x_i1 - x_i**2.0)**2.0) + ((x_i - 1)**2.0)

            r_sum += component

        unknowns["f"] = r_sum

        print("Evaluated function and got", unknowns['f'])
    
def bayesopt_optimize(iteration_count):
    dimensions = 7

    top = Problem()
    root = top.root = Group()

    root.add('p', RosenbrockMultiDim(_DIMENSIONS))

    top.driver = BayesoptOptimizer()
    top.driver.options["n_iterations"] = iteration_count
    top.driver.options["n_iter_relearn"] = 20
    top.driver.options["n_init_samples"] = 40
    top.driver.options["n_inner_iterations"] = 1000
    top.driver.options["surr_name"] = "sGaussianProcessML"
    top.driver.add_objective('p.f')

    for i in range(_DIMENSIONS):
        componentName = 'p{0}'.format(i)
        variableName = 'x{0}'.format(i)
        portName = '{0}.{1}'.format(componentName, variableName)
        root.add(componentName, IndepVarComp(variableName, 0.0))
        root.connect(portName, 'p.{0}'.format(variableName))
        top.driver.add_desvar(portName, lower=-5, upper=5)

    top.setup()

    top.run()

    print('\n')
    print('Minimum of %f found' % (top['p.f']))
    return top['p.f']

def cobyla_optimize(iteration_count):
    top = Problem()
    root = top.root = Group()

    root.add('p', RosenbrockMultiDim(_DIMENSIONS))

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['maxiter'] = iteration_count
    top.driver.add_objective('p.f')

    for i in range(_DIMENSIONS):
        componentName = 'p{0}'.format(i)
        variableName = 'x{0}'.format(i)
        portName = '{0}.{1}'.format(componentName, variableName)
        root.add(componentName, IndepVarComp(variableName, 0.0))
        root.connect(portName, 'p.{0}'.format(variableName))
        top.driver.add_desvar(portName, lower=-50, upper=50)

    top.setup()

    top.run()

    result_x = []
    for i in range(_DIMENSIONS):
        result_x.append(top['p.x{0}'.format(i)])

    print('\n')
    print('Minimum of {0} found at {1}'.format(top['p.f'], result_x))
    return top['p.f']

def main():
    iteration_counts_to_test = [10, 20, 50, 100, 200, 500, 1000]
    methods_to_test = [('COBYLA', cobyla_optimize, 10), ('BayesOpt', bayesopt_optimize, 10)]
    # For each iteration_count that we want to test, run COBYLA once and Bayesopt ten times (it's nondeterministic)
    with open('results.csv', 'wb') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["Method", "Iterations", "Minimum Found", "Time"])

        for iteration_count in iteration_counts_to_test:
            for method_name, method, run_count in methods_to_test:
                for i in range(run_count):
                    startTime = time.clock()
                    minimum = method(iteration_count)
                    stopTime = time.clock()
                    csvWriter.writerow([method_name, iteration_count, minimum, stopTime - startTime])
                    csvFile.flush()


if __name__ == "__main__":
    main()
