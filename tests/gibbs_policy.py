from __future__ import print_function
from builtins import range
from rlpy.Domains import GridWorld
from rlpy.Representations import Tabular
from scipy.optimize import check_grad, approx_fprime
from rlpy.Policies.gibbs import GibbsPolicy
import numpy as np


def test_fdcheck_dlogpi():
    """Finite differences check for the dlogpi of the gibbs policy"""
    logger = Logger()
    domain = GridWorld()
    representation = Tabular(logger=logger, domain=domain, discretization=20)
    policy = GibbsPolicy(representation=representation, logger=logger)

    def f(theta, s, a):
        policy.representation.theta = theta
        return np.log(policy.prob(s, a))

    def df(theta, s, a):
        policy.representation.theta = theta
        return policy.dlogpi(s, a)

    def df_approx(theta, s, a):
        return approx_fprime(theta, f, 1e-10, s, a)

    thetas = np.random.rand(10, len(representation.theta))
    for i in range(10):
        s = np.array([np.random.randint(4), np.random.randint(5)])
        a = np.random.choice(domain.possibleActions(s))
        for theta in thetas:
            # print "s", s
            # print "a", a
            # print "f", f(theta, s, a)
            # print "df", df(theta, s, a)
            # print "df_approx", df_approx(theta, s, a)
            error = check_grad(f, df, theta, s, a)
            print(error)
            assert np.abs(error) < 1e-6
