"""Puddle world domain (navigation task)."""

from Domain import Domain
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from Tools import memory
from Tools.progressbar import ProgressBar

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class PuddleWorld(Domain):
    """
    Implementation of the puddle world benchmark as described in references
    below.



    **STATE:** 2-dimensional vector, *s*, each dimension is continuous in [0,1]

    **ACTIONS:** [right, up, left, down] - NOTE it is not possible to loiter.

    **REWARD:** 0 for goal state, -1 for each step, and an additional penalty
        for passing near puddles.

    **REFERENCE:**

    .. seealso::
        Jong, N. & Stone, P.: Kernel-based models for reinforcement learning, ICML (2006)

    .. seealso::
        Sutton, R. S.: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding, NIPS(1996)

    """

    gamma = .95  # discout factor
    domain_fig = None
    valfun_fig = None
    polfun_fig = None

    episodeCap = 1000
    puddles = np.array([[[0.1, .75], [.45, .75]], [[.45, .4], [.45, .8]]])
    continuous_dims = np.arange(2)
    statespace_limits = np.array([[0., 1.]] * 2)

    actions = 0.05 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype="float")
    actions_num = 4

    def __init__(self, logger=None, noise_level=.01, gamma=.99):
        def gk(key):
            key["policy"] = key["policy"].__class__.__name__, key["policy"].__getstate__()
            key["self"] = copy(key["self"].__getstate__())
            lst = ["random_state", "V", "val_im", "pol_im", "pi_map", "valfun_fig", "polfun_fig",
                   "logger", "reward_map", "val_map", "state", "transition_samples", "d", "denom"]
            for l in lst:
                if l in key["self"]:
                    del key["self"][l]
            return key
        self.V = memory.cache(self.V, get_key=gk)
        self.transition_samples = memory.cache(self.transition_samples, get_key=gk)
        self.noise_level = noise_level
        self.gamma = gamma
        super(PuddleWorld, self).__init__(logger)
        self.reward_map = np.zeros((100, 100))
        self.val_map = np.zeros((100, 100))
        self.pi_map = np.zeros((100, 100))

        # precompute temporary values for speed in _reward
        self.d = self.puddles[:, 1, :] - self.puddles[:, 0, :]
        self.denom = (self.d ** 2).sum(axis=1)

        a = np.zeros((2))
        for i, x in enumerate(np.linspace(0, 1, 100)):
            for j, y in enumerate(np.linspace(0, 1, 100)):
                a[0] = x
                a[1] = y
                self.reward_map[j, i] = self._reward(a)
        if self.logger:
            self.logger.log("Noise level:\t{0}".format(self.noise_level))
        self.s0()
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

    #from Tools.run import lprofiler
    #@lprofiler
    def _reward(self, s):
        if self.isTerminal(s):
            return 0  # goal state reached
        reward = -1
        # compute puddle influence
        g = ((s - self.puddles[:, 0, :]) * self.d).sum(axis=1) / self.denom
        g = np.minimum(g, 1)
        g = np.maximum(g, 0)
        dists = np.sqrt(((self.puddles[:, 0, :] + g * self.d - s) ** 2).sum(axis=1))
        dists = dists[dists < 0.1]
        if len(dists):
            reward -= 400 * (0.1 - dists).max()
        return reward

    def showDomain(self, a=None):
        s = self.state
        #Draw the environment
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
                self.val_map[j, i] = representation.V(a, self.isTerminal(a), self.possibleActions())
                #self.pi_map[j, i] = representation.bestAction(a, self.isTerminal(a), self.possibleActions())

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

    def sample_V(self, s, policy, num_traj, max_len_traj):
        ret = 0.
        for i in xrange(num_traj):
            _, _, rt = self.sample_trajectory(s, policy, max_len_traj)
            ret += np.sum(rt * np.power(self.gamma, np.arange(len(rt))))
        return ret / num_traj

    def stationary_distribution(self, policy, discretization=20, num_traj=1000):
        counts = np.zeros((discretization, discretization))

        for i in xrange(num_traj):
            self.s0()
            s, _, _ = self.sample_trajectory(self.state, policy, self.episodeCap)
            k = (s * discretization).astype("int")
            k[k == discretization] = discretization - 1
            for j in xrange(k.shape[0]):
                counts[k[j,0], k[j,1]] += 1
        return counts.flatten()

    def model_for_partitioning(self, policy, partitioning, num_parts, num_traj=1000):
        stat = self.state.copy()
        ran_stat = copy(self.random_state)
        P = np.zeros((num_parts, num_parts))
        R = np.zeros((num_parts))
        d = np.zeros((num_parts))
        k = 0
        for i in xrange(num_traj):
            self.s0()
            s, _, r = self.sample_trajectory(self.state, policy, self.episodeCap)
            t = self.isTerminal()
            pid = partitioning(s[-1].flatten(), t)
            #d[pid] += 1
            if t:
                # drive the value of terminal states to 0
                P[pid, pid] += 1

            for j in xrange(len(s) - 2, 0, -1):
                cid = partitioning(s[j].flatten(), False)
                P[cid, pid] += 1
                d[cid] += 1
                R[cid] += r[j]
                pid = cid
                k += 1
        norm_per_state = P.sum(axis=1)
        mask = (norm_per_state != 0)
        norm_per_state = norm_per_state[mask]
        R[mask] /= norm_per_state
        P[mask, :] /= norm_per_state[:, None]
        d /= k
        self.state = stat
        self.random_state = ran_stat
        return P, R, d

    def V(self, policy, discretization=20, max_len_traj=200, num_traj=1000):
        dis = 1. / discretization
        xtest = np.mgrid[0:1:dis,0:1:dis].reshape(2,-1)
        R = np.zeros(xtest.shape[1])
        xtest_term = np.zeros(xtest.shape[1], dtype="bool")
        with ProgressBar() as p:
            for i in xrange(xtest.shape[1]):
                p.update(i, xtest.shape[1], "Sample V")
                if self.isTerminal(xtest[:,i]):
                    R[i] = 0.
                    xtest_term[i] = True
                else:
                    R[i] = self.sample_V(xtest[:,i], policy, num_traj, max_len_traj)
        stationary_dis = self.stationary_distribution(policy, discretization=discretization,
                                                      num_traj=num_traj)
        return R, xtest, xtest_term, stationary_dis

    def approx_Phi(self, representation, policy, max_len_traj=200, num_traj=1000):
        """
        approximate the feature matrix
        @param d: [ndarray with num_states entries] stationary state distribution
        """
        Phi = np.zeros((num_traj * max_len_traj, representation.features_num))
        for i in xrange(num_traj):
            s0, _, _ = self.s0()
            s, _, _ = self.sample_trajectory(s0, policy, max_len_traj)
            for j in xrange(max_len_traj):
                Phi[i*max_len_traj + j] = representation
class PuddleGapWorld(PuddleWorld):
    def _reward(self, s):
        r = super(PuddleGapWorld, self)._reward(s)
        if s[1] < .67 and s[1] >= .6:
            r = -1
        return r
