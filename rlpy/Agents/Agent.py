"""Standard Control Agent. """

from rlpy.Tools import className, incrementalAverageUpdate
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

    # The Representation to be used by the Agent
    representation = None
    # The domain the agent interacts with
    domain = None
    # The policy to be used by the agent
    policy = None
    # The initial learning rate. Note that initial_alpha should be set to
    # 1 for automatic learning rate; otherwise, initial_alpha will act as
    # a permanent upper-bound on alpha.
    initial_alpha = 0.1
    #: The learning rate
    alpha = 0
    #: Only used in the 'dabney' method of learning rate update.
    #: This value is updated in the updateAlpha method. We use the rate
    #: calculated by [Dabney W 2012]: http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
    candid_alpha = 0
    #: The eligibility trace, which marks states as eligible for a learning
    #: update. Used by \ref Agents.SARSA.SARSA "SARSA" agent when the
    #: parameter lambda is set. See:
    #: http://www.incompleteideas.net/sutton/book/7/node1.html
    eligibility_trace = []
    #: A simple object that records the prints in a file
    logger = None
    #: Used by some alpha_decay modes
    episode_count = 0
    # Decay mode of learning rate. Options are determined by valid_decay_modes.
    alpha_decay_mode = 'dabney'
    # Valid selections for the ``alpha_decay_mode``.
    valid_decay_modes = ['dabney', 'boyan', 'const', 'boyan_const']
    #  The N0 parameter for boyan learning rate decay
    boyan_N0 = 1000

    def __init__(self, representation, policy, domain,
                 initial_alpha=0.1, alpha_decay_mode='dabney', boyan_N0=1000):
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
        self.initial_alpha = initial_alpha
        self.alpha = initial_alpha
        self.alpha_decay_mode = alpha_decay_mode.lower()
        self.boyan_N0 = boyan_N0
        self.logger = logging.getLogger("rlpy.Agents." + self.__class__.__name__)

        # Check to make sure selected alpha_decay mode is valid
        if not self.alpha_decay_mode in self.valid_decay_modes:
            errMsg = "Invalid decay mode selected:" + self.alpha_decay_mode \
                + "; valid choices are: " + str(self.valid_decay_modes)
            self.logger.error(errMsg)
        #  Note that initial_alpha should be set to 1 for automatic learning rate; otherwise,
        #  initial_alpha will act as a permanent upper-bound on alpha.
        if self.alpha_decay_mode == 'dabney':
            self.initial_alpha = 1.0
            self.alpha = 1.0

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

    def updateAlpha(self, phi_s, phi_prime_s,
                    eligibility_trace_s, gamma, nnz, terminal):
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
        # zero and the dot product below becomes very large, creating a very
        # small alpha
            if not terminal:
                # Automatic learning rate: [Dabney W. 2012]
                # http://people.cs.umass.edu/~wdabney/papers/alphaBounds.p
                self.candid_alpha = abs(
                    np.dot(gamma * phi_prime_s - phi_s, eligibility_trace_s))
                self.candid_alpha    = 1 / \
                    (self.candid_alpha * 1.) if self.candid_alpha != 0 else np.inf
                self.alpha = min(self.alpha, self.candid_alpha)
            # else we take no action
        elif self.alpha_decay_mode == 'boyan':
            # New little change from not having +1 for episode count
            self.alpha = self.initial_alpha * \
                (self.boyan_N0 + 1.) / \
                (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
            # divide by l1 of the features; note that this method is only
            # called if phi_s != 0
            self.alpha /= sum(abs(phi_s))
        elif self.alpha_decay_mode == 'boyan_const':
            # New little change from not having +1 for episode count
            self.alpha = self.initial_alpha * \
                (self.boyan_N0 + 1.) / \
                (self.boyan_N0 + (self.episode_count + 1) ** 1.1)
        elif self.alpha_decay_mode == "const":
            self.alpha = self.initial_alpha
        else:
            self.logger.warn("Unrecognized decay mode ")

    def MC_episode(self, s=None, a=None, tolerance=0):
        """
        Run a single monte-carlo simulation episode from state *s* with action
        *a* following the current policy of the agent.
        See :py:meth:`~Agents.Agent.Agent.Q_MC`.

        :param s: The state used in the simulation
        :param a: The action used in the simulation
        :param tolerance: If the tolerance is set to a non-zero value, episodes
            will be stopped once the additive value to the sum of rewards drops
            below this threshold.  ie, this is the lower bound on acceptable performance.
        :return eps_return: Sum of rewards
        :return eps_length: Length of the Episode
        :return eps_term: Specifies the terminal condition of the episode: 0 (stopped due to length), >0 (stopped due to a terminal state)
        :return eps_discounted_return: Sum of discounted rewards.

        """
        eps_length = 0
        eps_return = 0
        eps_discounted_return = 0
        eps_term = 0
        if s is None:
            s = self.domain.s0()
        if a is None:
            a = self.policy.pi(s)
        while not eps_term and eps_length < self.domain.episodeCap:
            r, ns, eps_term = self.domain.step(a)
            s = ns
            eps_return += r
            eps_discounted_return += self.representation.domain.gamma ** eps_length * \
                r
            eps_length += 1
            a = self.policy.pi(s)
            if self.representation.domain.gamma ** eps_length < tolerance:
                break
        return eps_return, eps_length, eps_term, eps_discounted_return

    def Q_MC(self, s, a, MC_samples=1000, tolerance=0):
        """
        Use Monte-Carlo samples with the fixed policy to evaluate the Q(s,a).
        See :py:meth:`~Agents.Agent.Agent.MC_episode`.

        :param s: The state used in the simulation
        :param a: The action used in the simulation
        :param tolerance: If the tolerance is set to a non-zero value, episodes will be stopped once the additive value to the sum of rewards drops below this threshold
        :param MC_samples: Number of samples to be used to evaluated the Q value
        :return: Q_avg, Averaged sum of discounted rewards = estimate of the Q.

        """

        Q_avg = 0
        for i in np.arange(MC_samples):
            # print "MC Sample:", i
            _, _, _, Q = self.MC_episode(s, a, tolerance)
            Q_avg = incrementalAverageUpdate(Q_avg, Q, i + 1)
        return Q_avg

    def evaluate(self, samples, MC_samples, output_file):
        #TODO check if method is still used, otherwise delete
        """

        Evaluate the current policy for fixed number of samples and store them
        in a numpy 2-d array: (#samples) x (|S|+2), where the two appended
        columns correspond to the action (integer) and Q(s,a). \n
        Saves the data generated in a file. \n

        :param samples: The number of samples (s,a)
        :param MC_samples: The number of MC simulations used to estimate Q(s,a)
        :param output_file: The file in which the data is saved.

        :return: the data generated and stored in the output_file

        """

        # if gamma^steps falls bellow this number the MC-Chain will terminate
        # since it will not have much impact in evaluation of Q
        tolerance = 1e-10
        cols = self.domain.state_space_dims + 2
        DATA = np.empty((samples, cols))
        terminal = True
        steps = 0
        s = 0.  # will be overwritten anyway
        while steps < samples:
            s = self.domain.s0() if terminal or steps % self.domain.episodeCap == 0 else s
            a = self.policy.pi(s)

            # Store the corresponding Q
            Q = self.Q_MC(s, a, MC_samples, tolerance)
            DATA[steps, :] = np.hstack((s, [a, Q]))
            r, s, terminal = self.domain.step(a)
            steps += 1

            self.logger.debug(
                "Sample " + str(steps) + ":" + str(s) + " " + str(a) + " " + str(Q))

        np.save(output_file, DATA)
        return DATA

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
            if self.lambda_:
                self.eligibility_trace = np.zeros(
                    self.representation.features_num *
                    self.domain.actions_num)
                self.eligibility_trace_s = np.zeros(
                    self.representation.features_num)
