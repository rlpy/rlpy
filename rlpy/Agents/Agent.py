"""Standard Control Agent. """

from abc import ABCMeta, abstractmethod
import numpy as np
import logging

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Agent(object):

    """Learning Agent for obtaining good policices.

    The Agent receives observations from the Domain and performs actions to
    obtain some goal.

    The Agent interacts with the Domain in discrete timesteps.
    At each timestep the Agent receives some observations from the Domain
    which it uses to update its Representation of the Domain
    (ie, on each call to its :py:meth:`~Agents.Agent.Agent.learn` function).
    It then uses its Policy to select an action to perform.
    This process (observe, update, act) repeats nuntil some goal or fail state,
    determined by the Domain, is reached. At this point the
    :py:class:`~Experiments.Experiment.Experiment` determines
    whether the agent starts over or has its current policy tested
    (without any exploration).

    :py:class:`~Agents.Agent.Agent` is a base class that provides the basic
    framework for all RL Agents. It provides the methods and attributes that
    allow child classes to interact with the
    :py:class:`~Domains.Domain.Domain`,
    :py:class:`~Representations.Representation.Representation`,
    :py:class:`~Policies.Policy.Policy`, and
    :py:class:`~Experiments.Experiment.Experiment` classes within the
    RLPy library.

    .. note::
        All new agent implementations should inherit from this class.

    """

    __metaclass__ = ABCMeta
    # The Representation to be used by the Agent
    representation = None
    # The domain the agent interacts with
    domain = None
    # The policy to be used by the agent
    policy = None
    #: The eligibility trace, which marks states as eligible for a learning
    #: update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the
    #: parameter lambda is set. See:
    #: http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace = []
    #: A simple object that records the prints in a file
    logger = None
    #: Used by some alpha_decay modes
    episode_count = 0

    def __init__(self, representation, policy, domain, **kwargs):
        """initialization.

        :param representation: the :py:class:`~Representation.Representation.Representation`
            to use in learning the value function.
        :param policy: the :py:class:`~Policies.Policy.Policy` to use when selecting actions.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param initial_alpha: Initial learning rate to use (where applicable)

        .. warning::
            ``initial_alpha`` should be set to 1 for automatic learning rate;
            otherwise, initial_alpha will act as a permanent upper-bound on alpha.

        :param alpha_decay_mode: The learning rate decay mode (where applicable)
        :param boyan_N0: Initial Boyan rate parameter (when alpha_decay_mode='boyan')

        """
        self.representation = representation
        self.policy = policy
        self.domain = domain
        self.logger = logging.getLogger("rlpy.Agents." + self.__class__.__name__)



    @abstractmethod
    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        """
        This function receives observations of a single transition and
        learns from it.

        .. note::
            Each inheriting class (Agent) must implement this method.

        :param s: original state
        :param p_actions: possible actions in the original state
        :param a: action taken
        :param r: obtained reward
        :param ns: next state
        :param np_actions: possible actions in the next state
        :param na: action taken in the next state
        :param terminal: boolean indicating whether next state (ns) is terminal
        """
        return NotImplementedError

    def episodeTerminated(self):
        """
        This function adjusts all necessary elements of the agent at the end of
        the episodes.

        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.

        """
        # Increase the number of episodes
        self.episode_count += 1

        # Update the eligibility traces if they exist:
        # Set eligibility Traces to zero if it is end of the episode
        if hasattr(self, 'eligibility_trace'):
                self.eligibility_trace = np.zeros_like(self.eligibility_trace)
        if hasattr(self, 'eligibility_trace_s'):
                self.eligibility_trace_s = np.zeros_like(self.eligibility_trace_s)


class DescentAlgorithm(object):
    """
    Abstract base class that contains step-size control methods for (stochastic)
    descent algorithms such as TD Learning, Greedy-GQ etc.
    """

    # The initial learning rate. Note that initial_alpha should be set to
    # 1 for automatic learning rate; otherwise, initial_alpha will act as
    # a permanent upper-bound on alpha.
    initial_alpha       = 0.1
    #: The learning rate
    alpha               = 0
    #: Only used in the 'dabney' method of learning rate update.
    #: This value is updated in the updateAlpha method. We use the rate
    #: calculated by [Dabney W 2012]: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
    candid_alpha        = 0
    #: The eligibility trace, which marks states as eligible for a learning
    #: update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the
    #: parameter lambda is set. See:
    #: http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace   = []
    #: A simple object that records the prints in a file
    logger              = None
    #: Used by some alpha_decay modes
    episode_count       = 0
    # Decay mode of learning rate. Options are determined by valid_decay_modes.
    alpha_decay_mode    = 'dabney'
    # Valid selections for the ``alpha_decay_mode``.
    valid_decay_modes   = ['dabney','boyan','const','boyan_const']
    #  The N0 parameter for boyan learning rate decay
    boyan_N0            = 1000

    def __init__(self, initial_alpha = 0.1, alpha_decay_mode='dabney', boyan_N0=1000, **kwargs):
        """
        :param policy: the :py:class:`~Policies.Policy.Policy` to use when selecting actions.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param initial_alpha: Initial learning rate to use (where applicable)

        .. warning::
            ``initial_alpha`` should be set to 1 for automatic learning rate;
            otherwise, initial_alpha will act as a permanent upper-bound on alpha.

        :param alpha_decay_mode: The learning rate decay mode (where applicable)
        :param boyan_N0: Initial Boyan rate parameter (when alpha_decay_mode='boyan')

        """
        self.initial_alpha = initial_alpha
        self.alpha      = initial_alpha
        self.alpha_decay_mode = alpha_decay_mode.lower()
        self.boyan_N0   = boyan_N0

        # Note that initial_alpha should be set to 1 for automatic learning rate; otherwise,
        # initial_alpha will act as a permanent upper-bound on alpha.
        if self.alpha_decay_mode == 'dabney':
            self.initial_alpha = 1.0
            self.alpha = 1.0

        super(DescentAlgorithm, self).__init__(**kwargs)

    def updateAlpha(self,phi_s, phi_prime_s, eligibility_trace_s, gamma, nnz, terminal):
        """Computes a new learning rate (alpha) for the agent based on
        ``self.alpha_decay_mode``.

        :param phi_s: The feature vector evaluated at state (s)
        :param phi_prime_s: The feature vector evaluated at the new state (ns) = (s')
        :param eligibility_trace_s: Eligibility trace using state only (no copy-paste)
        :param gamma: The discount factor for learning
        :param nnz: The number of nonzero features
        :param terminal: Boolean that determines if the step is terminal or not

        """

        if self.alpha_decay_mode == 'dabney':
        # We only update alpha if this step is non-terminal; else phi_prime becomes
        # zero and the dot product below becomes very large, creating a very small alpha
            if not terminal:
                #Automatic learning rate: [Dabney W. 2012]
                self.candid_alpha = np.abs(np.dot(gamma * phi_prime_s - phi_s,
                                                   eligibility_trace_s))
                #  http://people.cs.umass.edu/~wdabney/papers/alphaBounds.p
                self.candid_alpha = 1 / (self.candid_alpha*1.) if self.candid_alpha != 0 else np.inf
                self.alpha = np.minimum(self.alpha,self.candid_alpha)
            # else we take no action
        elif self.alpha_decay_mode == 'boyan':
            self.alpha = self.initial_alpha * (self.boyan_N0 + 1.) / (self.boyan_N0 + (self.episode_count+1) ** 1.1)
            self.alpha /= np.sum(np.abs(phi_s))  # divide by l1 of the features; note that this method is only called if phi_s != 0
        elif self.alpha_decay_mode == 'boyan_const':
            self.alpha = self.initial_alpha * (self.boyan_N0 + 1.) / (self.boyan_N0 + (self.episode_count+1) ** 1.1)
        elif self.alpha_decay_mode == "const":
            self.alpha = self.initial_alpha
        else:
            self.logger.log("Unrecognized decay mode ")

    def episodeTerminated(self):
        """
        This function adjusts all necessary elements of the agent at the end of
        the episodes.

        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.

        """
        #Increase the number of episodes
        self.episode_count += 1
        self.representation.episodeTerminated()
        #super(DescentAlgorithm, self).episodeTerminated()



