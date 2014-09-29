"""Uniform Random policy"""

from .Policy import Policy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class UniformRandom(Policy):
    """
    Select an action uniformly at random from those available in a particular
    state.

    """

    def __init__(self, representation, seed=1):
        super(UniformRandom, self).__init__(representation, seed)
    
    def prob(self, s, terminal, p_actions):
        p = np.ones(len(p_actions)) / len(p_actions)
        return p

    def pi(self, s, terminal, p_actions):
        return self.random_state.choice(p_actions)
