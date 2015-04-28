"""Standard Control Agent. """

from abc import ABCMeta, abstractmethod
import numpy as np
import logging

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Agent(object):

    """Learning Agent for obtaining good policices.

    The Agent receives observations from the Domain and incorporates their
    new information into the representation, policy, etc. as needed.

    In a typical Experiment, the Agent interacts with the Domain in discrete
    timesteps.
    At each Experiment timestep the Agent receives some observations from the Domain
    which it uses to update the value function Representation of the Domain
    (ie, on each call to its :py:meth:`~rlpy.Agents.Agent.Agent.learn` function).
    The Policy is used to select an action to perform.
    This process (observe, update, act) repeats until some goal or fail state,
    determined by the Domain, is reached. At this point the
    :py:class:`~rlpy.Experiments.Experiment.Experiment` determines
    whether the agent starts over or has its current policy tested
    (without any exploration).

    :py:class:`~rlpy.Agents.Agent.Agent` is a base class that provides the basic
    framework for all RL Agents. It provides the methods and attributes that
    allow child classes to interact with the
    :py:class:`~rlpy.Domains.Domain.Domain`,
    :py:class:`~rlpy.Representations.Representation.Representation`,
    :py:class:`~rlpy.Policies.Policy.Policy`, and
    :py:class:`~rlpy.Experiments.Experiment.Experiment` classes within the
    RLPy library.

    .. note::
        All new agent implementations should inherit from this class.

    """

    __metaclass__ = ABCMeta
    # The Representation to be used by the Agent
    representation = None
    #: discount factor determining the optimal policy
    discount_factor = None
    #: The policy to be used by the agent
    policy = None
    #: The eligibility trace, which marks states as eligible for a learning
    #: update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the
    #: parameter lambda is set. See:
    #: http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace = []
    #: A simple object that records the prints in a file
    logger = None
    #: number of seen episodes
    episode_count = 0
    # A seeded numpy random number generator
    random_state = None

    def __init__(self, policy, representation, discount_factor, seed=1, **kwargs):
        """initialization.

        :param representation: the :py:class:`~rlpy.Representation.Representation.Representation`
            to use in learning the value function.
        :param policy: the :py:class:`~rlpy.Policies.Policy.Policy` to use when selecting actions.
        :param discount_factor: the discount factor of the optimal policy which should be
            learned
        :param initial_learn_rate: Initial learning rate to use (where applicable)

        .. warning::
            ``initial_learn_rate`` should be set to 1 for automatic learning rate;
            otherwise, initial_learn_rate will act as a permanent upper-bound on learn_rate.

        :param learn_rate_decay_mode: The learning rate decay mode (where applicable)
        :param boyan_N0: Initial Boyan rate parameter (when learn_rate_decay_mode='boyan')

        """
        self.representation = representation
        self.policy = policy
        self.discount_factor = discount_factor
        self.logger = logging.getLogger("rlpy.Agents." + self.__class__.__name__)

        # a new stream of random numbers for each agent
        self.random_state = np.random.RandomState(seed=seed)

    def init_randomization(self):
        """
        Any stochastic behavior in __init__() is broken out into this function
        so that if the random seed is later changed (eg, by the Experiment),
        other member variables and functions are updated accordingly.

        """
        pass

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

        # Set eligibility Traces to zero if it is end of the episode
        if hasattr(self, 'eligibility_trace'):
                self.eligibility_trace = np.zeros_like(self.eligibility_trace)


class DescentAlgorithm(object):
    """
    Abstract base class that contains step-size control methods for (stochastic)
    descent algorithms such as TD Learning, Greedy-GQ etc.
    """

    # The initial learning rate. Note that initial_learn_rate should be set to
    # 1 for automatic learning rate; otherwise, initial_learn_rate will act as
    # a permanent upper-bound on learn_rate.
    initial_learn_rate       = 0.1
    #: The learning rate
    learn_rate               = 0
    #: The eligibility trace, which marks states as eligible for a learning
    #: update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the
    #: parameter lambda is set. See:
    #: http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace   = []
    #: A simple object that records the prints in a file
    logger              = None
    #: Used by some learn_rate_decay modes
    episode_count       = 0
    # Decay mode of learning rate. Options are determined by valid_decay_modes.
    learn_rate_decay_mode    = 'dabney'
    # Valid selections for the ``learn_rate_decay_mode``.
    valid_decay_modes   = ['dabney','boyan','const','boyan_const']
    #  The N0 parameter for boyan learning rate decay
    boyan_N0            = 1000

    def __init__(self, initial_learn_rate = 0.1, learn_rate_decay_mode='dabney', boyan_N0=1000, **kwargs):
        """
        :param initial_learn_rate: Initial learning rate to use (where applicable)

        .. warning::
            ``initial_learn_rate`` should be set to 1 for automatic learning rate;
            otherwise, initial_learn_rate will act as a permanent upper-bound on learn_rate.

        :param learn_rate_decay_mode: The learning rate decay mode (where applicable)
        :param boyan_N0: Initial Boyan rate parameter (when learn_rate_decay_mode='boyan')

        """
        self.initial_learn_rate = initial_learn_rate
        self.learn_rate      = initial_learn_rate
        self.learn_rate_decay_mode = learn_rate_decay_mode.lower()
        self.boyan_N0   = boyan_N0

        # Note that initial_learn_rate should be set to 1 for automatic learning rate; otherwise,
        # initial_learn_rate will act as a permanent upper-bound on learn_rate.
        if self.learn_rate_decay_mode == 'dabney':
            self.initial_learn_rate = 1.0
            self.learn_rate = 1.0

        super(DescentAlgorithm, self).__init__(**kwargs)

    def updateLearnRate(self, phi, phi_prime, eligibility_trace,
					discount_factor, nnz, terminal):
        """Computes a new learning rate (learn_rate) for the agent based on
        ``self.learn_rate_decay_mode``.

        :param phi: The feature vector evaluated at state (s) and action (a)
        :param phi_prime_: The feature vector evaluated at the new state (ns) = (s') and action (na)
        :param eligibility_trace: Eligibility trace
        :param discount_factor: The discount factor for learning (gamma)
        :param nnz: The number of nonzero features
        :param terminal: Boolean that determines if the step is terminal or not

        """

        if self.learn_rate_decay_mode == 'dabney':
            # We only update learn_rate if this step is non-terminal; else phi_prime becomes
            # zero and the dot product below becomes very large, creating a very
            # small learn_rate
            if not terminal:
                # Automatic learning rate: [Dabney W. 2012]
                # http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
                candid_learn_rate = np.dot(discount_factor * phi_prime - phi,
                                                   eligibility_trace)
                if candid_learn_rate < 0:
                    self.learn_rate = np.minimum(self.learn_rate,-1.0/candid_learn_rate)
        elif self.learn_rate_decay_mode == 'boyan':
            self.learn_rate = self.initial_learn_rate * \
                (self.boyan_N0 + 1.) / \
                (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
            # divide by l1 of the features; note that this method is only called if phi != 0
            self.learn_rate /= np.sum(np.abs(phi))
        elif self.learn_rate_decay_mode == 'boyan_const':
            # New little change from not having +1 for episode count
            self.learn_rate = self.initial_learn_rate * \
                (self.boyan_N0 + 1.) / \
                (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
        elif self.learn_rate_decay_mode == "const":
            self.learn_rate = self.initial_learn_rate
        else:
            self.logger.warn("Unrecognized decay mode ")

    def episodeTerminated(self):
        """
        This function adjusts all necessary elements of the agent at the end of
        the episodes.

        .. note::
            Every Agent must call this function at the end of the learning if the
            transition led to terminal state.

        """
        # Increase the number of episodes
        self.episode_count += 1
        self.representation.episodeTerminated()
        super(DescentAlgorithm, self).episodeTerminated()
