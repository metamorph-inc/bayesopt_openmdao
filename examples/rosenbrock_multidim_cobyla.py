#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

from openmdao.api import IndepVarComp, Component, Problem, Group, ScipyOptimizer

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

        #print("Evaluated function and got", unknowns['f'])
    
def main():
    print("Bayesopt OpenMDAO Optimizer example")

    dimensions = 7

    top = Problem()
    root = top.root = Group()

    root.add('p', RosenbrockMultiDim(dimensions))

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['maxiter'] = 50000
    top.driver.add_objective('p.f')

    for i in range(dimensions):
        componentName = 'p{0}'.format(i)
        variableName = 'x{0}'.format(i)
        portName = '{0}.{1}'.format(componentName, variableName)
        root.add(componentName, IndepVarComp(variableName, 0.0))
        root.connect(portName, 'p.{0}'.format(variableName))
        top.driver.add_desvar(portName, lower=-50, upper=50)

    top.setup()

    top.run()

    result_x = []
    for i in range(dimensions):
        result_x.append(top['p.x{0}'.format(i)])

    print('\n')
    print('Minimum of {0} found at {1}'.format(top['p.f'], result_x))


if __name__ == "__main__":
    main()
