#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


__author__ = "Christoph Dann"

import os
import sys
#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT + '/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Domain import Domain
import numpy as np
import matplotlib.pyplot as plt


class PuddleWorld(Domain):
    """
    Implementation of the puddle world benchmark as described in

    Jong, N. & Stone, P.: Kernel-based models for reinforcement learning, ICML (2006)
    and
    Sutton, R. S.: Generalization in Reinforcement Learning:
    Successful Examples Using Sparse Coarse Coding, NIPS(1996)

    the state is a 2-dimensional vector s \in [0,1]^2
    """

    gamma = 1.  # discout factor
    domain_fig = None
    valfun_fig = None
    polfun_fig = None

    episodeCap = 1000
    puddles = np.array([[[0.1, .75], [.45, .75]], [[.45, .4], [.45, .8]]])
    continuous_dims = np.arange(2)
    statespace_limits = np.array([[0., 1.]] * 2)

    actions = 0.05 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype="float")
    actions_num = 4

    def __init__(self, logger=None, noise_level=.01, gamma=1.):
        self.noise_level = noise_level
        self.gamma = gamma
        super(PuddleWorld, self).__init__(logger)
        self.reward_map = np.zeros((100, 100))
        self.val_map = np.zeros((100, 100))
        self.pi_map = np.zeros((100, 100))
        a = np.zeros((2))
        for i, x in enumerate(np.linspace(0, 1, 100)):
            for j, y in enumerate(np.linspace(0, 1, 100)):
                a[0] = x
                a[1] = y
                self.reward_map[j, i] = self.reward(a)
        if self.logger:
            self.logger.log("Noise level:\t{0}".format(self.noise_level))

    def s0(self):
        self.state = self.rand_state.rand(2)
        while self.isTerminal(self.state):
            self.state = self.rand_state.rand(2)
        return self.state.copy()

    def isTerminal(self, s):
        return s.sum() > 0.95 * 2

    def possibleActions(self, s):
        return np.arange(self.actions_num)

    def step(self, a):
        a = self.actions[a]
        ns = self.state + a + self.rand_state.randn() * self.noise_level
        # make sure we stay inside the [0,1]^2 region
        ns = np.minimum(ns, 1.)
        ns = np.maximum(ns, 0.)
        self.state = ns.copy()
        return (self.reward(ns), ns, self.isTerminal(ns))

    def reward(self, s):
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

    def showDomain(self, s, a=None):
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
                self.val_map[j, i] = representation.V(a)
                self.pi_map[j, i] = representation.bestAction(a)

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

if __name__ == "__main__":
    h = PuddleWorld()
    h.test(200)
