"""Domain base class"""
import numpy as np
import logging
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Domain(object):

    """
    The Domain controls the environment in which the
    :py:class:`~rlpy.Agents.Agent.Agent` resides as well as the reward function the
    Agent is subject to.

    The Agent interacts with the Domain in discrete timesteps called
    *episodes* (see :py:meth:`~rlpy.Domains.Domain.Domain.step`).
    At each step, the Agent informs the Domain what indexed action it wants to
    perform.  The Domain then calculates the effects this action has on the
    environment and updates its internal state accordingly.
    It also returns the new state to the agent, along with a reward/penalty,
    and whether or not the episode is over (thus resetting the agent to its
    initial state).

    This process repeats until the Domain determines that the Agent has either
    completed its goal or failed.
    The :py:class:`~rlpy.Experiments.Experiment.Experiment` controls this cycle.

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
    (:py:meth:`~rlpy.Domains.Domain.Domain.showDomain` and
    :py:meth:`~rlpy.Domains.Domain.Domain.showLearning` respectively) \n
    All new domain implementations should inherit from :py:class:`~rlpy.Domains.Domain.Domain`.

    .. note::
        Though the state *s* can take on almost any value, if a dimension is not
        marked as 'continuous' then it is assumed to be integer.

    """
    #: The discount factor by which rewards are reduced
    discount_factor = .9
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
    # A seeded numpy random number generator
    random_state = None

    def __init__(self):
        self.logger = logging.getLogger("rlpy.Domains." + self.__class__.__name__)
        self.state_space_dims = len(self.statespace_limits)
        # To make sure type of discount_factor is float. This will later on be used in
        # LSPI to force A matrix to be float
        self.discount_factor = float(self.discount_factor)
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

    def init_randomization(self):
        """
        Any stochastic behavior in __init__() is broken out into this function
        so that if the random seed is later changed (eg, by the Experiment),
        other member variables and functions are updated accordingly.

        """
        pass

    def __str__(self):
        res = """{self.__class__}:
------------
Dimensions: {self.state_space_dims}
|S|:        {self.states_num}
|A|:        {self.actions_num}
Episode Cap:{self.episodeCap}
Gamma:      {self.discount_factor}
""".format(self=self)
        return res

    def show(self, a=None, representation=None):
        """
        Shows a visualization of the current state of the domain and that of
        learning.

        See :py:meth:`~rlpy.Domains.Domain.Domain.showDomain()` and
        :py:meth:`~rlpy.Domains.Domain.Domain.showLearning()`,
        both called by this method.

        .. note::
            Some domains override this function to allow an optional *s*
            parameter to be passed, which overrides the *self.state* internal
            to the domain; however, not all have this capability.

        :param a: The action being performed
        :param representation: The learned value function
            :py:class:`~rlpy.Representation.Representation.Representation`.

        """
        self.saveRandomState()
        self.showDomain(a=a)
        self.showLearning(representation=representation)
        self.loadRandomState()

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
            :py:class:`~rlpy.Representation.Representation.Representation`
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
            using other datatypes.  See :py:meth:`~rlpy.Tools.GeneralTools.vec2id`
            and :py:meth:`~rlpy.Tools.GeneralTools.id2vec` for converting between
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
            calls (see also the :py:meth:`~rlpy.Domains.Domain.Domain.sampleStep`
            method).
            Be sure to look at unique noise parameters of each domain if you
            require deterministic transitions.


        :param a: The action to perform.

        .. warning::

            The action *a* **must** be an integer >= 0, and might better be
            called the "actionID".  See the class description
            :py:class:`~rlpy.Domains.Domain.Domain` above.

        :return: The tuple (r, ns, t, p_actions) =
            (Reward [value], next observed state, isTerminal [boolean])

        """
        raise NotImplementedError("Each domain needs to implement this method")

    def saveRandomState(self):
        """
        Stores the state of the the random generator.
        Using loadRandomState this state can be loaded.
        """
        self.random_state_backup = self.random_state.get_state()

    def loadRandomState(self):
        """
        Loads the random state stored in the self.random_state_backup
        """
        self.random_state.set_state(self.random_state_backup)

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

    def sampleStep(self, a, num_samples):
        """
        Sample a set number of next states and rewards from the domain.
        This function is used when state transitions are stochastic;
        deterministic transitions will yield an identical result regardless
        of *num_samples*, since repeatedly sampling a (state,action) pair
        will always yield the same tuple (r,ns,terminal).
        See :py:meth:`~rlpy.Domains.Domain.Domain.step`.

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

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k is "logger":
                continue
            # This block bandles matplotlib transformNode objects,
                # which cannot be coped
            try:
                setattr(result, k, deepcopy(v, memo))
            except:
                # Try this: if this doesnt work, just let theat error get thrown
                try:
                    setattr(result, k, v.frozen())
                except:
                    self.logger.warning('Could not copy attribute ' + k +
                                        ' when duplicating domain.')
        return result
