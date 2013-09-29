"""Domain base class"""
import numpy as np
import Tools

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Domain(object):
    """
    The Domain controls the environment in which the \ref Agents.Agent.Agent "Agent" resides and the goal that said Agent is trying to acheive.

    The Agent interacts with the %Domain in discrete timesteps called 'episodes'. Each episode, the %Domain provides the Agent with some observations
    about its surroundings. Based on that information, the Agent informs the %Domain what indexed action it wants to perform.
    The %Domain then calculates the effects this action has on the environment and returns the new state, a reward/penalty, and whether or not the episode is over or not (thus resetting the agent to its initial state).
    This process repeats until the %Domain determines that the Agent has either completed its goal or
    failed. The \ref Experiments.Experiment.Experiment "Experiment" controls this cycle.

    Because Agents are designed to be agnostic to the %Domain that they are acting within and the problem they are trying to solve,
    the %Domain needs to completely describe everything related to the task. Therefore, the %Domain must not only define the observations
    that the Agent receives, but also the states it can be in, the actions that it can perform, and the relationships between the three.
    Note that because RL-Agents are designed around obtaining a reward, observations that the %Domain returns should include a reward.

    The \c %Domain class is a superclass that provides the basic framework for all Domains. It provides the methods and attributes
    that allow child classes to interact with the \c %Agent and \c Experiment classes within the RLPy library.
    %Domains should also provide methods that provide visualization of the %Domain itself and of the Agent's learning (showDomain and showLearning)   \n
    All new domain implementations should inherit from \c %Domain.

    .. note::
        Though the state s can take on almost any value, if a dimension is not
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
        """
        Initializes the \c %Domain object. See code
        \ref Domain_init "Here".

        """
        if logger is None:
            logger = Tools.Logger()
        self.logger = logger
        self.state_space_dims = len(self.statespace_limits)
        self.gamma = float(self.gamma)  # To make sure type of gamma is float. This will later on be used in LSPI to force A matrix to be float
        # For discrete domains, limits should be extended by half on each side so that the mapping becomes identical with continuous states
        # The original limits will be saved in self.discrete_statespace_limits
        self.extendDiscreteDimensions()
        if self.continuous_dims == []:
            self.states_num = int(np.prod(self.statespace_limits[:, 1]
                                  - self.statespace_limits[:, 0]))
        else:
            self.states_num = np.inf

        # a new stream of random numbers for each domain
        self.random_state = np.random.RandomState()

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

    def show(self, a=None, representation=None, s=None):
        """
        Shows a visualization of the current state of the domain and that of learning. See code
        \ref Domain_show "Here".
        @param a
        The action being performed
        @param representation
        The representation to show
        """
        self.showDomain(a, s=s)
        self.showLearning(representation)

    def showDomain(self, a=0, s=None):
        """
        \b ABSTRACT \b METHOD: Shows a visualization of the current state of the domain. See code
        \ref Domain_showDomain "Here".
        @param a
        The action being performed

        """
        pass

    def showLearning(self, representation):
        """
        \b ABSTRACT \b METHOD: Shows a visualization of the current learning. This visualization is usually in the form
        of a value gridded value function and policy. It is thus really only possible for 1 or 2-state domains. See code
        \ref Domain_showLearning "Here".
        @param representation
        The representation to show

        """
        pass

    def s0(self):
        """
        Begins a new episode and returns the initial observed state of the %Domain
        @return
        A numpy array that defines the initial state of the %Domain. See code
        \ref Domain_s0 "Here".

        """
        raise NotImplementedError("Children need to implement this method")

    def possibleActions(self, s=None):
        """
        Returns all actions in the domain.
        The default version returns all actions [0, 1, 2...].
        You may want to change this method in your domain if all actions are not available at all times. See code
        \ref Domain_possActions "Here".
        @return
        A numpy array that contains a list of every action in the domain.

        """
        return np.arange(self.actions_num)

    def step(self, a):
        """
        \b ABSTRACT \b METHOD: Performs an action while in a specific state and updates the domain accordingly.
        This function should return a reward that the agent acheives for the action, the next observed state that the domain/agent should be in,
        and a boolean determining whether a goal or fail state has been reached. See code
        \ref Domain_step "Here".
        @param a
        The action to perform. Note that each action outside of the domain corresponds to the index of the action. This index will be interpreted within the domain.
        @return [r,ns,t] => Reward (number), next observed state (state), isTerminal (bool)

        """
        raise NotImplementedError("Each domain needs to implement this method")

    def isTerminal(self):
        """
        Determines whether the stats s is a terminal state.
        The default definition does not terminate. Override this function to specify otherwise. See code
        \ref Domain_isTerminal "Here".
        @return
        True if the state is a terminal state, False otherwise.
        """
        return False

    def extendDiscreteDimensions(self):
        """
        Offsets discrete dimensions by 0.5 so that binning works properly. See code
        \ref Domain_extendDiscreteDimensions "Here".

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
        Sample a set number of next states and rewards from the domain. See code
        \ref sampleStep "Here".
        @param s
        The given state
        @param a
        The given action
        @param num_samples
        The number of next states and rewards to be sampled.
        @return
        [S,A] => S is an array of next states, A is an array of rewards for those states

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
