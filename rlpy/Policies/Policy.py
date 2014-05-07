"""Policy base class"""

from rlpy.Tools import className
import logging

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Policy(object):

    """The Policy determines the discrete action that an
    :py:class:`~Agents.Agent.Agent` will take  given its
    :py:class:`~Representations.Representation.Representation`.

    The Agent learns about the :py:class:`~Domains.Domain.Domain`
    as the two interact.
    At each step, the Agent passes information about its current state
    to the Policy; the Policy uses this to decide what discrete action the
    Agent should perform next (see :py:meth:`~Policies.Policy.Policy.pi`) \n

    The Policy class is a base class that provides the basic framework for all
    policies. It provides the methods and attributes that allow child classes
    to interact with the Agent and Representation within the RLPy library. \n

    .. note::
        All new policy implementations should inherit from Policy.

    """

    representation = None
    DEBUG = False

    def __init__(self, representation):
        """
        :param representation: the :py:class:`~Representations.Representation.Representation`
            to use in learning the value function.

        """
        self.representation = representation
        # An object to record the print outs in a file
        self.logger = logging.getLogger("rlpy.Policies." + self.__class__.__name__)

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
        If :py:meth:Policies.Policy.Policy.turnOffExploration` was called
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

    def collectSamples(self, samples):
        """
        DEPRECATED


        Return matrices of S,A,NS,R,T where each row of each numpy 2d-array
        is a sample by following the current policy.

        - S: (#samples) x (# state space dimensions)
        - A: (#samples) x (1) int [we are storing actionIDs here, integers]
        - NS:(#samples) x (# state space dimensions)
        - R: (#samples) x (1) float
        - T: (#samples) x (1) bool

        See :py:meth:`Agents.Agent.Agent.Q_MC` and :py:meth:`Agents.Agent.Agent.MC_episode`
        """
        domain = self.representation.domain
        S = empty(
            (samples,
             self.representation.domain.state_space_dims),
            dtype=type(domain.s0()))
        A = empty((samples, 1), dtype='uint16')
        NS = S.copy()
        T = A.copy()
        R = empty((samples, 1))

        sample = 0
        eps_length = 0
        # So the first sample forces initialization of s and a
        terminal = True
        while sample < samples:
            if terminal or eps_length > self.representation.domain.episodeCap:
                s = domain.s0()
                a = self.pi(s)

            # Transition
            r, ns, terminal = domain.step(a)
            # Collect Samples
            S[sample] = s
            A[sample] = a
            NS[sample] = ns
            T[sample] = terminal
            R[sample] = r

            sample += 1
            eps_length += 1
            s = ns
            a = self.pi(s)

        return S, A, NS, R, T
