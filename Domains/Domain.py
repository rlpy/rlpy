"""Domain base class"""
import numpy as np
import Tools
from copy import copy
from Tools.progressbar import ProgressBar
from collections import Counter
__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Domain(object):
    """
    The Domain controls the environment in which the
    :py:class:`~Agents.Agent.Agent` resides as well as the reward function the
    Agent is subject to.

    The Agent interacts with the Domain in discrete timesteps called
    *episodes* (see :py:meth:`~Domains.Domain.Domain.step`).
    At each step, the Agent informs the Domain what indexed action it wants to
    perform.  The Domain then calculates the effects this action has on the
    environment and updates its internal state accordingly.
    It also returns the new state to the agent, along with a reward/penalty,
    and whether or not the episode is over (thus resetting the agent to its
    initial state).

    This process repeats until the Domain determines that the Agent has either
    completed its goal or failed.
    The :py:class:`~Experiments.Experiment.Experiment` controls this cycle.

    Because Agents are designed to be agnostic to the Domain that they are
    acting within and the problem they are trying to solve, the Domain needs
    to completely describe everything related to the task. Therefore, the
    Domain must not only define the observations that the Agent receives,
    but also the states it can be in, the actions that it can perform, and the
    relationships between the three.

    The Domain class is a base clase that provides the basic framework for all
    Domains. It provides the methods and attributes that allow child classes
    to interact with the Agent and Experiment classes within the RLPy library.
    Domains should also provide methods that provide visualization of the
    Domain itself and of the Agent's learning
    (:py:meth:`~Domains.Domain.Domain.showDomain` and
    :py:meth:`~Domains.Domain.Domain.showLearning` respectively) \n
    All new domain implementations should inherit from :py:class:`~Domains.Domain.Domain`.

    .. note::
        Though the state *s* can take on almost any value, if a dimension is not
        marked as 'continuous' then it is assumed to be integer.

    """
    #: The discount factor by which rewards are reduced
    gamma = .9
    #: The number of possible states in the domain
    states_num = 0  # was None
    #: The number of Actions the agent can perform
    actions_num = 0  # was None
    #: Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    statespace_limits = []  # was None
    #: Limits of each dimension of a discrete state space. This is the same as statespace_limits, without the extra -.5, +.5 added to each dimension
    discrete_statespace_limits = []  # was None
    #: Number of dimensions of the state space
    state_space_dims = 0  # was None
    #: List of the continuous dimensions of the domain
    continuous_dims = []
    #: The cap used to bound each episode (return to state 0 after)
    episodeCap = None
    #: A simple object that records the prints in a file
    logger = None

    def __init__(self, logger=None):
        if logger is None:
            logger = Tools.Logger()
        self.logger = logger
        self.state_space_dims = len(self.statespace_limits)
        self.gamma = float(self.gamma)  # To make sure type of gamma is float. This will later on be used in LSPI to force A matrix to be float
        # For discrete domains, limits should be extended by half on each side so that the mapping becomes identical with continuous states
        # The original limits will be saved in self.discrete_statespace_limits
        self._extendDiscreteDimensions()
        if self.continuous_dims == []:
            self.states_num = int(np.prod(self.statespace_limits[:, 1]
                                  - self.statespace_limits[:, 0]))
        else:
            self.states_num = np.inf

        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState()

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        res = """{self.__class__}:
------------
Dimensions: {self.state_space_dims}
|S|:        {self.states_num}
|A|:        {self.actions_num}
Episode Cap:{self.episodeCap}
Gamma:      {self.gamma}
""".format(self=self)
        return res

    def show(self, a=None, representation=None):
        """
        Shows a visualization of the current state of the domain and that of
        learning.

        See :py:meth:`~Domains.Domain.Domain.showDomain()` and
        :py:meth:`~Domains.Domain.Domain.showLearning()`,
        both called by this method.

        .. note::
            Some domains override this function to allow an optional *s*
            parameter to be passed, which overrides the *self.state* internal
            to the domain; however, not all have this capability.

        :param a: The action being performed
        :param representation: The learned value function
            :py:class:`~Representation.Representation.Representation`.

        """
        self.showDomain(a=a)
        self.showLearning(representation=representation)

    def showDomain(self, a=0):
        """
        *Abstract Method:*\n
        Shows a visualization of the current state of the domain.

        :param a: The action being performed.

        """
        pass

    def showLearning(self, representation):
        """
        *Abstract Method:*\n
        Shows a visualization of the current learning,
        usually in the form of a gridded value function and policy.
        It is thus really only possible for 1 or 2-state domains.

        :param representation: the learned value function
            :py:class:`~Representation.Representation.Representation`
            to generate the value function / policy plots.

        """
        pass

    def s0(self):
        """
        Begins a new episode and returns the initial observed state of the Domain.
        Sets self.state accordingly.

        :return: A numpy array that defines the initial domain state.

        """
        raise NotImplementedError("Children need to implement this method")

    def possibleActions(self, s=None):
        """
        The default version returns an enumeration of all actions [0, 1, 2...].
        We suggest overriding this method in your domain, especially if not all
        actions are available from all states.

        :param s: The state to query for possible actions
            (overrides self.state if ``s != None``)

        :return: A numpy array containing every possible action in the domain.

        .. note::

            *These actions must be integers*; internally they may be handled
            using other datatypes.  See :py:meth:`~Tools.GeneralTools.vec2id`
            and :py:meth:`~Tools.GeneralTools.id2vec` for converting between
            integers and multidimensional quantities.

        """
        return np.arange(self.actions_num)


    # TODO: change 'a' to be 'aID' to make it clearer when we refer to
    # actions vs. integer IDs of actions?  They aren't always interchangeable.
    def step(self, a):
        """
        *Abstract Method:*\n
        Performs the action *a* and updates the Domain
        state accordingly.
        Returns the reward/penalty the agent obtains for
        the state/action pair determined by *Domain.state*  and the parameter
        *a*, the next state into which the agent has transitioned, and a
        boolean determining whether a goal or fail state has been reached.

        .. note::

            Domains often specify stochastic internal state transitions, such
            that the result of a (state,action) pair might vary on different
            calls (see also the :py:meth:`~Domains.Domain.Domain.sampleStep`
            method).
            Be sure to look at unique noise parameters of each domain if you
            require deterministic transitions.


        :param a: The action to perform.

        .. warning::

            The action *a* **must** be an integer >= 0, and might better be
            called the "actionID".  See the class description
            :py:class:`~Domains.Domain.Domain` above.

        :return: The tuple (r, ns, t, p_actions) =
            (Reward [value], next observed state, isTerminal [boolean])

        """
        raise NotImplementedError("Each domain needs to implement this method")

    def isTerminal(self):
        """
        Returns ``True`` if the current Domain.state is a terminal one, ie,
        one that ends the episode.  This often results from either a failure
        or goal state being achieved.\n
        The default definition does not terminate.

        :return: ``True`` if the state is a terminal state, ``False`` otherwise.

        """
        return False

    def _extendDiscreteDimensions(self):
        """
        Offsets discrete dimensions by 0.5 so that binning works properly.

        .. warning::

            This code is used internally by the Domain base class.
            **It should only be called once**

        """
        # Store the original limits for other types of calculations
        self.discrete_statespace_limits = self.statespace_limits
        self.statespace_limits = self.statespace_limits.astype('float')
        for d in xrange(self.state_space_dims):
            if d not in self.continuous_dims:
                self.statespace_limits[d, 0] += -.5
                self.statespace_limits[d, 1] += +.5

    def sample_trajectory(self, s_init, policy, max_len_traj):
        r = np.zeros(max_len_traj)
        s = np.zeros((max_len_traj, self.state_space_dims))
        a = np.zeros(max_len_traj, dtype="int")
        self.s0()
        self.state = s_init.copy()

        ns = self.state
        pa = self.possibleActions()
        t = self.isTerminal()

        for i in range(max_len_traj - 1):
            if t:
                break
            s[i] = ns
            a[i] = policy.pi(s[i], t, pa)
            r[i], ns, t, pa = self.step(a[i])
            if t:
                break
        return s[:i+1], a[:i+1], r[:i+1]


    def transition_samples(self, policy, num_samples):
        stat = self.state.copy()
        ran_stat = copy(self.random_state)

        R = np.empty(num_samples)
        S = np.empty((num_samples, self.state_space_dims))
        S_term = np.zeros(num_samples, dtype="bool")
        Sn = np.empty((num_samples, self.state_space_dims))
        Sn_term = np.zeros(num_samples, dtype="bool")

        term = True
        for i in range(num_samples):
            if term:
                s, term, pa = self.s0()
            a = policy.pi(s, term, pa)
            r, sn, sn_term, pa = self.step(a)
            S[i] = s
            Sn[i] = sn
            Sn_term[i] = sn_term
            S_term[i] = term
            R[i] = r
            s, term = sn, sn_term
        self.state = stat
        self.random_state = ran_stat
        return S, S_term, Sn, Sn_term, R

    def sampleStep(self, a, num_samples):
        """
        Sample a set number of next states and rewards from the domain.
        This function is used when state transitions are stochastic;
        deterministic transitions will yield an identical result regardless
        of *num_samples*, since repeatedly sampling a (state,action) pair
        will always yield the same tuple (r,ns,terminal).
        See :py:meth:`~Domains.Domain.Domain.step`.

        :param a: The action to attempt
        :param num_samples: The number of next states and rewards to be sampled.

        :return: A tuple of arrays ( S[], A[] ) where
            *S* is an array of next states,
            *A* is an array of rewards for those states.

        """
        next_states = []
        rewards = []
        s = self.state.copy()
        for i in xrange(num_samples):
            r, ns, terminal = self.step(a)
            self.state = s.copy()
            next_states.append(ns)
            rewards.append(r)

        return np.array(next_states), np.array(rewards)


class BigDiscreteDomain(object):

    @property
    def num_dim(self):
        return self.statespace_limits.shape[0]

    batch_V_sampling = 1

    """
    def num_V_batches(self, policy, num_traj,  discretization):
        xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj)
        return int(np.ceil(float(xtest.shape[1]) / self.batch_V_sampling))

    def precompute_V_batch_cache(self, discretization, policy, max_len_traj, num_traj, batch):
        xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj)
        rng = slice(batch * self.batch_V_sampling, min((batch + 1) * self.batch_V_sampling, xtest.shape[1]))
        print "Compute batch", batch, "with indices", rng
        # no need to return stuff. should be cached anyway
        self.sample_V_batch(xtest[:,rng], policy, num_traj, max_len_traj)

    """

    def V(self, policy, discretization=None, max_len_traj=200, num_traj=1000, num_traj_stationary=1000):
        stat_dis = self.stationary_distribution(policy, num_traj=num_traj_stationary)
        #xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj_stationary)
        stationary_dis = np.zeros(len(stat_dis), dtype="int")
        R = np.zeros(len(stat_dis))
        xtest = np.zeros((self.num_dim, len(stat_dis)), dtype="int64")
        xtest_term = np.zeros(xtest.shape[1], dtype="bool")
        if self.batch_V_sampling == 1:
            with ProgressBar() as p:
                i = 0
                for k, count in stat_dis.iteritems():
                    xtest[:,i] = np.array(k)
                    stationary_dis[i] = count
                    p.update(i, xtest.shape[1], "Sample V")
                    if self.isTerminal(xtest[:,i]):
                        R[i] = 0.
                        xtest_term[i] = True
                    else:
                        R[i] = self.sample_V(xtest[:,i], policy, num_traj, max_len_traj)
                    i += 1
        else:
            # batch mode to be able to do caching with smaller batches and
            # precompute V in parallel with cluster jobs
            for k, count in stat_dis.iteritems():
                xtest[:,i] = np.array(k)
                stationary_dis[i] = count
                i += 1
            for i in xrange(0, xtest.shape[1], self.batch_V_sampling):
                rng = slice(i * self.batch_V_sampling, min((i + 1) * self.batch_V_sampling, xtest.shape[1]))

                R[rng], xtest_term[rng] = self.sample_V_batch(xtest[:,rng], policy, num_traj, max_len_traj)
        return R, xtest, xtest_term, stationary_dis
    """
    def sample_V_batch(self, xtest, policy, num_traj, max_len_traj):
        R = np.zeros(xtest.shape[1])
        xtest_term = np.zeros(xtest.shape[1], dtype="bool")
        with ProgressBar() as p:
            for i in xrange(xtest.shape[1]):
                p.update(i, xtest.shape[1], "Sample V batch")
                if self.isTerminal(xtest[:,i]):
                    R[i] = 0.
                    xtest_term[i] = True
                else:
                    R[i] = self.sample_V(xtest[:,i], policy, num_traj, max_len_traj)
        return R, xtest_term
    """
    def sample_V(self, s, policy, num_traj, max_len_traj):
        ret = 0.
        for i in xrange(num_traj):
            _, _, rt = self.sample_trajectory(s, policy, max_len_traj)
            ret += np.sum(rt * np.power(self.gamma, np.arange(len(rt))))
        return ret / num_traj

    def stationary_distribution(self, policy, num_traj=1000):
        counts = Counter()

        for i in xrange(num_traj):
            self.s0()
            s, _, _ = self.sample_trajectory(self.state, policy, self.episodeCap)
            for j in xrange(s.shape[0]):
                counts[tuple(s[j,:])] += 1
        return counts



class ContinuousDomain(object):

    @property
    def num_dim(self):
        return self.statespace_limits.shape[0]

    batch_V_sampling = 1

    """
    def num_V_batches(self, policy, num_traj,  discretization):
        xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj)
        return int(np.ceil(float(xtest.shape[1]) / self.batch_V_sampling))

    def precompute_V_batch_cache(self, discretization, policy, max_len_traj, num_traj, batch):
        xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj)
        rng = slice(batch * self.batch_V_sampling, min((batch + 1) * self.batch_V_sampling, xtest.shape[1]))
        print "Compute batch", batch, "with indices", rng
        # no need to return stuff. should be cached anyway
        self.sample_V_batch(xtest[:,rng], policy, num_traj, max_len_traj)

    """

    def V(self, policy, discretization=20, max_len_traj=200, num_traj=1000, num_traj_stationary=1000):
        xtest, stationary_dis = self.test_states(policy, discretization=discretization, num_traj=num_traj_stationary)
        R = np.zeros(xtest.shape[1])
        xtest_term = np.zeros(xtest.shape[1], dtype="bool")
        if self.batch_V_sampling == 1:
            with ProgressBar() as p:
                for i in xrange(xtest.shape[1]):
                    p.update(i, xtest.shape[1], "Sample V")
                    if self.isTerminal(xtest[:,i]):
                        R[i] = 0.
                        xtest_term[i] = True
                    else:
                        R[i] = self.sample_V(xtest[:,i], policy, num_traj, max_len_traj)
        else:
            # batch mode to be able to do caching with smaller batches and
            # precompute V in parallel with cluster jobs
            for i in xrange(0, xtest.shape[1], self.batch_V_sampling):
                rng = slice(i * self.batch_V_sampling, min((i + 1) * self.batch_V_sampling, xtest.shape[1]))
                R[rng], xtest_term[rng] = self.sample_V_batch(xtest[:,rng], policy, num_traj, max_len_traj)
        return R, xtest, xtest_term, stationary_dis
    """
    def sample_V_batch(self, xtest, policy, num_traj, max_len_traj):
        R = np.zeros(xtest.shape[1])
        xtest_term = np.zeros(xtest.shape[1], dtype="bool")
        with ProgressBar() as p:
            for i in xrange(xtest.shape[1]):
                p.update(i, xtest.shape[1], "Sample V batch")
                if self.isTerminal(xtest[:,i]):
                    R[i] = 0.
                    xtest_term[i] = True
                else:
                    R[i] = self.sample_V(xtest[:,i], policy, num_traj, max_len_traj)
        return R, xtest_term
    """
    def sample_V(self, s, policy, num_traj, max_len_traj):
        ret = 0.
        for i in xrange(num_traj):
            _, _, rt = self.sample_trajectory(s, policy, max_len_traj)
            ret += np.sum(rt * np.power(self.gamma, np.arange(len(rt))))
        return ret / num_traj

    def test_states(self, policy, discretization=21, num_traj=3000):
        stationary_dis = self.stationary_distribution(policy, discretization=discretization,
                                                      num_traj=num_traj)
        dis = (self.statespace_limits[:,1] - self.statespace_limits[:,0]) / discretization
        num_dim = self.statespace_limits.shape[0]
        a = [slice(self.statespace_limits[i,0], self.statespace_limits[i,1], dis[i]) for i in xrange(num_dim)]
        xtest = np.mgrid[a].reshape(num_dim,-1)
        xtest += dis[:, None] / 2
        k = self.discrete_idx(xtest.T, discretization)
        for j in xrange(xtest.shape[1]):
            assert(stationary_dis[tuple(k[j])] == stationary_dis.flatten()[j])
        stat = stationary_dis.flatten()
        mask = stat > 0
        return xtest.T[mask].T, stat[mask]

    def stationary_distribution(self, policy, discretization=20, num_traj=1000):
        counts = np.zeros([discretization] * self.num_dim)

        for i in xrange(num_traj):
            self.s0()
            s, _, _ = self.sample_trajectory(self.state, policy, self.episodeCap)
            k = self.discrete_idx(s, discretization)
            for j in xrange(k.shape[0]):
                t = tuple(k[j])
                counts[t] += 1
        return counts


    def discrete_idx(self, s, discretization):
        w  = (self.statespace_limits[:,1] - self.statespace_limits[:,0])
        k = s - self.statespace_limits[:,0]
        k /= w
        k = (k * discretization).astype("int")
        k[k == discretization] = discretization - 1
        return k
