# BayesOpt OpenMDAO Driver

## Installation (Windows)

 1. `pip install bayesopt`
 2. Install this package with `pip install -e .` (or `python -m pip install -e .` if using the Tonka Python).

## Usage

The BayesOpt driver can be instantiated like any OpenMDAO driver:

    from bayesopt_openmdao.bayesopt_optimizer import BayesoptOptimizer

    # ...

    top = Problem()
    root = top.root = Group()

    # ...

    top.driver = BayesoptOptimizer()
    top.driver.options["n_iterations"] = 200

    top.driver.add_desvar('p1.x', lower=-40, upper=50)
    top.driver.add_desvar('p2.y', lower=-20, upper=50)
    top.driver.add_objective('p.f_xy')

    top.setup()

Supported options will be passed through to BayesOpt.  These currently include:

  * `n_iterations` - Number of iterations of BayesOpt
  * `n_inner_iterations` - maximum number of iterations to optimize the acquisition function
  * `n_iter_relearn` - Number of iterations between re-learning kernel parameters (iterations where relearning happens take longer to compute)
  * `n_init_samples` - Number of initial samples when learning the preliminary model of the target function.  Each sample requires a target function evaluation.
  * `surr_name` - Name of the surrogate function

For more details, see the [BayesOpt documentation](https://rmcantin.bitbucket.io/html/usemanual.html).

## Examples

Example code is located in the `examples` subdirectory.

  * `optimizer.py` - simple paraboloid optimization problem, with two independent variables.
  * `cobyla_opt.py` - same problem as `optimizer.py`, using COBYLA instead of BayesOpt.
  * `rosenbrock_multidim.py` - optimization problem using the rosenbrock test function, with a configurable number of independent variables (change `dimensions` in its main function).
  * `rosenbrock_multidem_cobyla` - same as above, using COBYLA instead of BayesOpt.
  * `comparison.py` - automated comparison test between COBYLA and BayesOpt, using a set of parameters that seems to work reasonably well.  Varies the number of independent variables and maximum number of samples to evaluate (BayesOpt will always evaluate this many samples; COBYLA can complete with fewer samples if its tolerance is reached).
