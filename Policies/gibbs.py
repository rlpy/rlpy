from Policy import Policy, Logger
import numpy as np
from Tools import randSet, discrete_sample


class GibbsPolicy(Policy):
    """
    Gibbs policy for finite number of actions

    Warning: assumes that the features for each action are stacked, i.e.,
    a feature vector consists of |A| identical stacked vectors.
    """

    def pi(self, s):
        p = self.probabilities(s)
        return discrete_sample(p)

    def dlogpi(self, s, a):

        #TODO Finite differences check!
        v = self.probabilities(s)
        n = self.representation.features_num
        phi = self.representation.phi(s)
        res = -np.outer(v, phi)
        res.shape = self.representation.theta.shape
        res[a * n:(a + 1) * n] += phi
        #res.shape = self.representation.theta.shape
        return res

    def prob(self, s, a):
        """
        probability of chosing action a given the state s
        """
        v = self.probabilities(s)
        return v[a]

    @property
    def theta(self):
        return self.representation.theta

    @theta.setter
    def theta(self, v):
        self.representation.theta = v

    def probabilities(self, s):

        phi = self.representation.phi(s)
        n = self.representation.features_num
        v = np.exp(np.dot(self.representation.theta.reshape(-1, n), phi))
        v /= v.sum()
        return v

if __name__ == "__main__":

    # Finite differences check of dlogpi
    from Domains import GridWorld
    from Representations import Tabular
    from scipy.optimize import check_grad, approx_fprime

    MAZE = './Domains/GridWorldMaps/4x5.txt'
    NOISE = .3
    logger = Logger()
    domain = GridWorld(MAZE, noise=NOISE, logger=logger)
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
        a = randSet(domain.possibleActions(s))
        for theta in thetas:
            print "s", s
            print "a", a
            #print "f", f(theta, s, a)
            #print "df", df(theta, s, a)
            #print "df_approx", df_approx(theta, s, a)
            print "Error", check_grad(f, df, theta, s, a)
