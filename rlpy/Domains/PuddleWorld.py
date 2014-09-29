"""Puddle world domain (navigation task)."""

from .Domain import Domain
import numpy as np
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class PuddleWorld(Domain):

    """
    Implementation of the puddle world benchmark as described in references
    below.



    **STATE:** 2-dimensional vector, *s*, each dimension is continuous in [0,1]\n
    **ACTIONS:** [right, up, left, down] - NOTE it is not possible to loiter.\n
    **REWARD:** 0 for goal state, -1 for each step, and an additional penalty
        for passing near puddles.

    **REFERENCE:**

    .. seealso::
        Jong, N. & Stone, P.: Kernel-based models for reinforcement learning, ICML (2006)

    .. seealso::
        Sutton, R. S.: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding, NIPS(1996)

    """

    discount_factor = 1.  # discout factor
    domain_fig = None
    valfun_fig = None
    polfun_fig = None

    episodeCap = 1000
    puddles = np.array([[[0.1, .75], [.45, .75]], [[.45, .4], [.45, .8]]])
    continuous_dims = np.arange(2)
    statespace_limits = np.array([[0., 1.]] * 2)

    actions = 0.05 * \
        np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype="float")
    actions_num = 4

    def __init__(self, noise_level=.01, discount_factor=1.):
        self.noise_level = noise_level
        self.discount_factor = discount_factor
        super(PuddleWorld, self).__init__()
        self.reward_map = np.zeros((100, 100))
        self.val_map = np.zeros((100, 100))
        self.pi_map = np.zeros((100, 100))
        a = np.zeros((2))
        for i, x in enumerate(np.linspace(0, 1, 100)):
            for j, y in enumerate(np.linspace(0, 1, 100)):
                a[0] = x
                a[1] = y
                self.reward_map[j, i] = self._reward(a)

    def s0(self):
        self.state = self.random_state.rand(2)
        while self.isTerminal():
            self.state = self.random_state.rand(2)
        return self.state.copy(), False, self.possibleActions()

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return s.sum() > 0.95 * 2

    def possibleActions(self, s=0):
        return np.arange(self.actions_num)

    def step(self, a):
        a = self.actions[a]
        ns = self.state + a + self.random_state.randn() * self.noise_level
        # make sure we stay inside the [0,1]^2 region
        ns = np.minimum(ns, 1.)
        ns = np.maximum(ns, 0.)
        self.state = ns.copy()
        return self._reward(ns), ns, self.isTerminal(), self.possibleActions()

    def _reward(self, s):
        if self.isTerminal(s):
            return 0  # goal state reached
        reward = -1
        # compute puddle influence
        d = self.puddles[:, 1, :] - self.puddles[:, 0, :]
        denom = (d ** 2).sum(axis=1)
        g = ((s - self.puddles[:, 0, :]) * d).sum(axis=1) / denom
        g = np.minimum(g, 1)
        g = np.maximum(g, 0)
        dists = np.sqrt(((self.puddles[:, 0, :] + g * d - s) ** 2).sum(axis=1))
        dists = dists[dists < 0.1]
        if len(dists):
            reward -= 400 * (0.1 - dists[dists < 0.1]).max()
        return reward

    def showDomain(self, a=None):
        s = self.state
        # Draw the environment
        if self.domain_fig is None:
            self.domain_fig = plt.figure("Domain")
            self.reward_im = plt.imshow(self.reward_map, extent=(0, 1, 0, 1),
                                        origin="lower")
            self.state_mark = plt.plot(s[0], s[1], 'kd', markersize=20)
            plt.draw()
        else:
            self.domain_fig = plt.figure("Domain")
            self.state_mark[0].set_data([s[0]], [s[1]])
            plt.draw()

    def showLearning(self, representation):
        a = np.zeros((2))
        for i, x in enumerate(np.linspace(0, 1, 100)):
            for j, y in enumerate(np.linspace(0, 1, 100)):
                a[0] = x
                a[1] = y
                self.val_map[j,
                             i] = representation.V(
                    a,
                    self.isTerminal(a),
                    self.possibleActions())
                self.pi_map[j,
                            i] = representation.bestAction(
                    a,
                    self.isTerminal(a),
                    self.possibleActions())

        if self.valfun_fig is None:
            self.valfun_fig = plt.figure("Value Function")
            plt.clf()
            self.val_im = plt.imshow(self.val_map, extent=(0, 1, 0, 1),
                                     origin="lower")
            plt.colorbar()
        else:
            self.valfun_fig = plt.figure("Value Function")
            self.val_im.set_data(self.val_map)
            self.val_im.autoscale()
        plt.draw()

        if self.polfun_fig is None:
            self.polfun_fig = plt.figure("Policy")
            plt.clf()
            self.pol_im = plt.imshow(self.pi_map, extent=(0, 1, 0, 1),
                                     origin="lower", cmap="4Actions")
        else:
            self.polfun_fig = plt.figure("Policy")
            self.pol_im.set_data(self.pi_map)
            self.pol_im.autoscale()
        plt.draw()


class PuddleGapWorld(PuddleWorld):

    def _reward(self, s):
        r = super(PuddleGapWorld, self)._reward(s)
        if s[1] < .67 and s[1] >= .6:
            r = -1
        return r
