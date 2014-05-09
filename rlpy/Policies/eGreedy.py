"""epsilon-Greedy Policy"""

from .Policy import Policy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class eGreedy(Policy):
    """
    Greedy policy with epsilon-probability for uniformly random exploration.

    The policy acts greedily according to the value function with probability
    1-epsilon and choses a uniformly random action otherwise.

    """

    epsilon = None
    old_epsilon = None
    # This boolean variable is used to avoid random selection among actions
    # with the same values
    forcedDeterministicAmongBestActions = None

    def __init__(self, representation, epsilon=.1,
                 forcedDeterministicAmongBestActions=False):
        self.epsilon = epsilon
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        super(eGreedy, self).__init__(representation)

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
