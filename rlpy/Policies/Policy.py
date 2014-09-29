"""Policy base class"""

from rlpy.Tools import className, discrete_sample
import numpy as np
import logging
from abc import ABCMeta, abstractmethod

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Policy(object):

    """The Policy determines the discrete action that an
    :py:class:`~rlpy.Agents.Agent.Agent` will take  given its
    :py:class:`~rlpy.Representations.Representation.Representation`.

    The Agent learns about the :py:class:`~rlpy.Domains.Domain.Domain`
    as the two interact.
    At each step, the Agent passes information about its current state
    to the Policy; the Policy uses this to decide what discrete action the
    Agent should perform next (see :py:meth:`~rlpy.Policies.Policy.Policy.pi`) \n

    The Policy class is a base class that provides the basic framework for all
    policies. It provides the methods and attributes that allow child classes
    to interact with the Agent and Representation within the RLPy library. \n

    .. note::
        All new policy implementations should inherit from Policy.

    """

    __netaclass__ = ABCMeta
    representation = None
    DEBUG = False
    # A seeded numpy random number generator
    random_state = None

    def __init__(self, representation, seed=1):
        """
        :param representation: the :py:class:`~rlpy.Representations.Representation.Representation`
            to use in learning the value function.

        """
        self.representation = representation
        # An object to record the print outs in a file
        self.logger = logging.getLogger("rlpy.Policies." + self.__class__.__name__)
        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState(seed=seed)

    def init_randomization(self):
        """
        Any stochastic behavior in __init__() is broken out into this function
        so that if the random seed is later changed (eg, by the Experiment),
        other member variables and functions are updated accordingly.

        """
        pass

    @abstractmethod
    def pi(self, s, terminal, p_actions):
        """
        *Abstract Method:*\n Select an action given a state.

        :param s: The current state
        :param terminal: boolean, whether or not the *s* is a terminal state.
        :param p_actions: a list / array of all possible actions in *s*.
        """
        raise NotImplementedError

    def turnOffExploration(self):
        """
        *Abstract Method:* \n Turn off exploration (e.g., epsilon=0 in epsilon-greedy)
        """
        pass
    # [turnOffExploration code]

    # \b ABSTRACT \b METHOD: Turn exploration on. See code
    # \ref Policy_turnOnExploration "Here".
    # [turnOnExploration code]
    def turnOnExploration(self):
        """
        *Abstract Method:* \n
        If :py:meth:`~rlpy.Policies.Policy.Policy.turnOffExploration` was called
        previously, reverse its effects (e.g. restore epsilon to its previous,
        possibly nonzero, value).
        """
        pass

    def printAll(self):
        """ Prints all class information to console. """
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value


class DifferentiablePolicy(Policy):

    __metaclass__ = ABCMeta

    def pi(self, s, terminal, p_actions):
        """Sample action from policy"""
        p = self.probabilities(s, terminal)
        return discrete_sample(p)

    @abstractmethod
    def dlogpi(self, s, a):
        """derivative of the log probabilities of the policy"""
        return NotImplementedError

    def prob(self, s, a):
        """
        probability of chosing action a given the state s
        """
        v = self.probabilities(s, False)
        return v[a]

    @property
    def theta(self):
        return self.representation.weight_vec

    @theta.setter
    def theta(self, v):
        self.representation.weight_vec = v

    @abstractmethod
    def probabilities(self, s, terminal):
        """
        returns a vector of num_actions length containing the normalized
        probabilities for taking each action given the state s
        """
        return NotImplementedError
