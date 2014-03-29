"""epsilon-Greedy Policy"""

import Policies.Policy
from Tools import __rlpy_location__
import numpy as np
import pickle
import copy

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class eGreedy(Policies.Policy.Policy):
    epsilon = None
    old_epsilon = None
    forcedDeterministicAmongBestActions = None  # This boolean variable is used to avoid random selection among actions with the same values

    def __init__(self, representation, logger, epsilon=.1,
                 forcedDeterministicAmongBestActions=False):
        self.epsilon = epsilon
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        super(eGreedy, self).__init__(representation, logger)
        if self.logger:
            self.logger.log("=" * 60)
            self.logger.log("Policy: eGreedy")
            self.logger.log("Epsilon\t\t{0}".format(self.epsilon))

    def pi(self, s, terminal, p_actions):
        coin = np.random.rand()
        # print "coin=",coin
        if coin < self.epsilon:
            return np.random.choice(p_actions)
        else:
            b_actions = self.representation.bestActions(s, terminal, p_actions)
            if self.forcedDeterministicAmongBestActions:
                return b_actions[0]
            else:
                return np.random.choice(b_actions)

    def prob(self, s, terminal, p_actions):
        p = np.ones(len(p_actions)) / len(p_actions)
        p *= self.epsilon
        b_actions = self.representation.bestActions(s, terminal, p_actions)
        if self.forcedDeterministicAmongBestActions:
            p[b_actions[0]] += (1 - self.epsilon)
        else:
            p[b_actions] += (1 - self.epsilon) / len(b_actions)
        return p

    def turnOffExploration(self):
        self.old_epsilon = self.epsilon
        self.epsilon = 0

    def turnOnExploration(self):
        self.epsilon = self.old_epsilon


class StoredPolicy(eGreedy):

    """Loads a pickled representation from a file and acts epsilon-greedy according to the
    represented Q-function"""

    __author__ = "Christoph Dann"

    def __init__(self, filename, epsilon=0.):
        """
        filename: location of the pickled representation
        """
        self.filename = filename
        self.epsilon = epsilon
        fn = self.filename.replace("__rlpy_location__", __rlpy_location__)
        with open(fn) as f:
            self.representation = pickle.load(f)
        self.forcedDeterministicAmongBestActions = False

    def __getstate__(self):
        b = copy.copy(self.__dict__)
        del b["representation"]
        return b
