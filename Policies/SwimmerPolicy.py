import Policies.Policy

import numpy as np
from scipy.io import loadmat
from Tools import __rlpy_location__, cartesian
import os

class SwimmerPolicy(Policies.Policy.Policy):
    """
    Fixed deterministic policy for a 3-link swimmer.
    It is a discretized version from Yuval Tassas MATLAB implmentation
    The policy seems to be almost optimal.

    .. seealso::
        Tassa, Y., Erez, T., & Smart, B. (2007).
        *Receding Horizon Differential Dynamic Programming.*
        In Advances in Neural Information Processing Systems.
    """

    default_location = os.path.join(__rlpy_location__, "Policies", "swimmer3.mat")

    def __init__(self, representation, logger=None, filename=default_location, epsilon=0.1):

        self.logger = logger
        self.representation = representation
        E = loadmat(filename)["E"]
        self.locs = E["Locs"][0][0]
        self.nlocs = E["Nlocs"][0][0]
        self.scale = E["scale"][0][0]
        self.gains = E["Gains"][0][0]
        self.d = 3
        self.eGreedy = False
        self.epsilon = epsilon
        d = self.d
        # incidator variables for angles in a state representation
        self.angles = np.zeros(2 + self.d * 2 + 1, dtype=np.bool)
        self.angles[2:2+self.d-1] = True
        self.actions = cartesian((d - 1) * [[-2., 0., 2]])
        self.actions_num = len(self.actions)

    def pi(self,s, terminal, p_actions):
        coin = np.random.rand()
        if coin < self.epsilon:
            return np.random.choice(p_actions)
        else:
            if self.eGreedy:
                b_actions = self.representation.bestActions(s, terminal, p_actions)
                return np.random.choice(b_actions)
            else:
                return self.pi_sam(s, terminal, p_actions)
    """
    def turnOffExploration(self):
        self.old_epsilon = self.epsilon
        self.epsilon = 0
        self.eGreedy = True

    def turnOnExploration(self):
        self.epsilon = self.old_epsilon
        self.eGreedy = False
    """
    def pi_sam(self, s, terminal, p_actions):
        #import ipdb; ipdb.set_trace()
        # d        = size(x,1);
        # x_a = [x(~angles,:); sin(x(angles,:)); cos(x(angles,:))];
        x_a = np.hstack((s[self.angles == False], np.sin(s[self.angles]), np.cos(s[self.angles])))

        # M        = (1:d);
        M = np.arange(len(self.angles))
        idx = np.argsort(np.hstack((M[self.angles == False], M[self.angles], M[self.angles] + .5)))
        # [v, ix]  = sort([M(~angles) M(angles) M(angles)+0.5]);
        # x_a      = x_a(ix,:);
        x_a = x_a[idx]
        Nx = np.dot(self.scale, x_a)
        Ndiff = Nx[:,None] - self.nlocs
        dist = np.sum(Ndiff**2, axis=0)
        w = np.argmin(dist)

        diff = s - self.locs[:, w]
        diff[self.angles] = np.mod(diff[self.angles] + np.pi, 2 * np.pi) - np.pi
        k = np.hstack((np.ones(1), diff.flatten()))
        u = np.dot(self.gains[:,:,w], k)
        dd = np.sum((u[None, :] - self.actions[p_actions])**2, axis=1)
        aid = np.argmin(dd)
        return p_actions[aid]

