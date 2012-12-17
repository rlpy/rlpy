######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################

# NOTE That though the state s can take on almost any
# value, if a dimension is not marked as 'continuous'
# then it is assumed to be integer.

from Tools import *
from pydoc import classname
class Domain(object):
    gamma = .90             # Discount factor default = .9
    states_num = None       # Number of states
    actions_num = None      # Number of Actions
    statespace_limits = None# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    state_space_dims = None # Number of dimensions of the state space
    continuous_dims = []     # List of continuous dimensions of the domain, default = empty []
    episodeCap = None       # The cap used to bound each episode (return to s0 after)
    logger = None           # Used to capture the text in a file
    #Termination Signals of Episodes
    NOT_TERMINATED          = 0
    NOMINAL_TERMINATION     = 1
    CRITICAL_TERMINATION    = 2

    def __init__(self,logger):
        self.logger = logger
        for v in ['statespace_limits','actions_num','episodeCap']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.state_space_dims = len(self.statespace_limits)
        
        # For discrete domains, limits should be extended by half on each side so that the mapping becomes identical with continuous states
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
    def show(self,s,a, representation):     
        self.showDomain(s,a)
        self.showLearning(representation)
    
    ## Show a visualization of the current state of the domain
    def showDomain(self,s,a = 0):
        pass
    
    ## Show a visualization of the current learning, usually in the form
    # of a value gridded value function and policy (really only possible for 1 or 2-state domains).
    def showLearning(self,representation):
        pass
    
    ## @return: the initial state
    def s0(self):       
        abstract
        
    ## @return: the numpy array of possible actions in each state.
    # The vanilla (default) version returns all actions [0, 1, 2...]
    # You may want to change this in your domain if all actions are not available at all times.
    def possibleActions(self,s):
        return arange(self.actions_num)
    
    ## Perform action a while in state s:
    # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
    def step(self,s,a):
        abstract
        
    ## Returns r, ns, p. Each row of each output corresponds to one possibility.
    # e.g. s,a -> r[i], ns[i] with probability p[i]"""
    def expectedStep(self,s,a):
        pass 
    
    ## Returns True if the state s is a terminal state, False otherwise."""
    def isTerminal(self,s):
        abstract
    def saturateState(self,s):
        # Kemal put more info on this. Why do you need this?
        dim = len(s)
               
        for i in range(0,dim):
            if s[i] < self.statespace_limits[i,0]+0.5: s[i] = int(self.statespace_limits[i,0]+0.5)
            if s[i] > self.statespace_limits[i,1]-0.5: s[i] = int(self.statespace_limits[i,1]-0.5)
    def test(self,T):
        # Run the environment with a random Agent for T steps
        terminal    = True
        steps       = 0
        while steps < T:
            if terminal:
                if steps != 0: self.showDomain(s,a)
                s = self.s0()
            elif steps % self.episodeCap == 0:
                s = self.s0()
            a = randSet(self.possibleActions(s))
            self.showDomain(s,a)
            r,s,terminal = self.step(s, a)
            steps += 1
    def printAll(self):
        printClass(self)
    def extendDiscreteDimensions(self):
        self.statespace_limits = self.statespace_limits.astype('float')
        for d in arange(self.state_space_dims):
             if not d in self.continuous_dims:
                 self.statespace_limits[d,0] += -.5 
                 self.statespace_limits[d,1] += +.5 
    def MCExpectedStep(self,s,a,no_samples):
        
        # Sample no_samples of next states and rewards from the domain
        
        next_states = []
        rewards = []
        
        for i in range(0,no_samples):
        
            r,ns,terminal = self.step(s, a)
        
            next_states.append(ns)
            rewards.append(r)
                
        return array(next_states),array(rewards) 
    