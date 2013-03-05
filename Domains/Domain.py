## \file Domain.py
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT
######################################################

from Tools import *
from pydoc import classname
## The \c Domain class controls the enviroment that the agent is acting upon.
#
# \c %Domain provides the basic framework for domains to be acted upon by the Agent class.
# All new domains should inherit from \c Domain.
#
# TODO Describe what role domains do exactly here. Agents.Agent.Agent
#
# \note Though the state s can take on almost any
# value, if a dimension is not marked as 'continuous'
# then it is assumed to be integer.

class Domain(object):
	## The discount factor. Default is .8
    gamma = .80
	## Number of states
    states_num = None
	## Number of Actions
    actions_num = None
	## Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    statespace_limits = None
	## same as above, except for discrete dimensions this array does not have the -.5,+.5 added to discrete dimensions
    discrete_statespace_limits = None
	## Number of dimensions of the state space
    state_space_dims = None
	## List of continuous dimensions of the domain. Default is empty []
    continuous_dims = []
	## The cap used to bound each episode (return to s0 after)
    episodeCap = None
	## Used to capture the text in a file
    logger = None
    ## Termination Signals of Episodes
    NOT_TERMINATED          = 0
    NOMINAL_TERMINATION     = 1
    CRITICAL_TERMINATION    = 2

    def __init__(self,logger):
        self.logger = logger
        for v in ['statespace_limits','actions_num','episodeCap']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.state_space_dims = len(self.statespace_limits)
        self.gamma = self.gamma * 1.0 # To make sure type of gamma is float. This will later on be used in LSPI to force A matrix to be float
        # For discrete domains, limits should be extended by half on each side so that the mapping becomes identical with continuous states
        # The original limits will be saved in self.discrete_statespace_limits
        self.extendDiscreteDimensions()
        if self.continuous_dims == []:
            self.states_num = int(prod(self.statespace_limits[:,1]-self.statespace_limits[:,0]))
        else:
            self.states_num = inf
        if logger:
            self.logger.line()
            self.logger.log("Domain:\t\t"+str(className(self)))
            self.logger.log("Dimensions:\t"+str(self.state_space_dims))
            self.logger.log("|S|:\t\t"+str(self.states_num))
            self.logger.log("|A|:\t\t"+str(self.actions_num))
            self.logger.log("|S|x|A|:\t\t"+str(self.actions_num*self.states_num))
            self.logger.log("Episode Cap:\t"+str(self.episodeCap))
            self.logger.log("Gamma:\t\t"+str(self.gamma))

	## Show a visualization of the current state of the domain and of the current learning
    def show(self,s,a, representation):
        self.showDomain(s,a)
        self.showLearning(representation)

    ## \b ABSTRACT \b METHOD: Show a visualization of the current state of the domain
    def showDomain(self,s,a = 0):
        pass

    ## \b ABSTRACT \b METHOD: Show a visualization of the current learning, usually in the form
    # of a value gridded value function and policy (really only possible for 1 or 2-state domains).
    def showLearning(self,representation):
        pass

    ## @return numpy array: the initial state.
    def s0(self):
        abstract

    ## The vanilla (default) version returns all actions [0, 1, 2...]
    # You may want to change this in your domain if all actions are not available at all times.
	# @return numpy array: possible actions in each state.
    def possibleActions(self,s):
        return arange(self.actions_num)

    ## \b ABSTRACT \b METHOD: Perform action a while in state s:
    # @return [r,ns,t] => Reward, next state, isTerminal (boolean)
    def step(self,s,a):
        abstract

    ## \b ABSTRACT \b METHOD: Returns r, ns, p. Each row of each output corresponds to one possibility.
    # e.g. s,a -> r[i], ns[i] with probability p[i]
    def expectedStep(self,s,a):
        pass

    ## Returns True if the state s is a terminal state, False otherwise.
    def isTerminal(self,s):
        return False # By default the domain does not terminate unless it specifies the otherwise

	## Run the environment by performing random actions for T steps
    def test(self,T):
        terminal    = True
        steps       = 0
        while steps < T:
            if terminal:
                if steps != 0: self.showDomain(s,a)
                s = self.s0uniform()
            elif steps % self.episodeCap == 0:
                s = self.s0uniform()
            a = randSet(self.possibleActions(s))
            self.showDomain(s,a)
            r,s,terminal = self.step(s, a)
            steps += 1

	## Prints the class data
    def printAll(self):
        printClass(self)

    def extendDiscreteDimensions(self):
        # Store the original limits for other types of calculations
        self.discrete_statespace_limits = self.statespace_limits
        self.statespace_limits = self.statespace_limits.astype('float')
        for d in arange(self.state_space_dims):
             if not d in self.continuous_dims:
                 self.statespace_limits[d,0] += -.5
                 self.statespace_limits[d,1] += +.5

	## Sample no_samples of next states and rewards from the domain
    def MCExpectedStep(self,s,a,no_samples):

        next_states = []
        rewards = []

        for i in arange(0,no_samples):

            r,ns,terminal = self.step(s, a)

            next_states.append(ns)
            rewards.append(r)

        return array(next_states),array(rewards)

	## Returns a state sampled uniformely from the state space
    def s0uniform(self):
        if className(self) == 'BlocksWorld':
            print "s0uniform is not supported by %s.\nFurther implementation is needed to filter impossible states." % className(self)
        if self.continuous_dims == []:
            s = empty(self.state_space_dims, dtype = integer)
        else:
            s = empty(self.state_space_dims)

        for d in arange(self.state_space_dims):
            a,b = self.statespace_limits[d]
            s[d] = random.rand()*(b-a)+a
            if not d in self.continuous_dims:
                s[d] = int(s[d])
        if len(s) == 1: s = s[0]
        return s

	## This function is used for cases when state vector has elements outside of its limits. This function simply caps each element of the state space to lie in the allowed state limits
    def saturateState(self,s):
        return bound_vec(s,self.discrete_statespace_limits)
