"""Uniform Random policy"""

from Policy import Policy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class UniformRandom(Policy):
    def pi(self, s, terminal, p_actions):
        return np.random.choice(p_actions)
