######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from pydoc import classname
class Domain(object):
    gamma = .90             # Discount factor default = .9
    states_num = None       # Number of states
    actions_num = None      # Number of Actions
    statespace_limits = None# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    state_space_dims = None # Number of dimensions of the state space
    continous_dims = []     # List of continuous dimensions of the domain, default = None
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
        
        # For discrete domains, limits should be extended by half on each side so that the mapping becomes identical with continous states
        self.extendDiscreteDimensions()
        if self.continous_dims == []:
            self.states_num = int(prod(self.statespace_limits[:,1]-self.statespace_limits[:,0]))
        else:
            self.states_num = inf
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
    def showDomain(self,s,a = 0):
        pass
    def showLearning(self,representation):
        pass
    def s0(self):       
        # Returns the initial state
        abstract
    def possibleActions(self,s):
        # default. You may want to change it in your domain if all actions are not available at all times
        # Returns the list of possible actions in each state the vanilla version returns all of the actions
        return arange(self.actions_num)
    def step(self,s,a):
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
        abstract    
    def expectedStep(self,s,a):
        # returns r, ns, p. Each row of each output corresponds to one possibility.
        # e.g. s,a -> r[i], ns[i] with probability p[i]
        pass 
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        abstract
    def test(self,T):
        # Run the environment with a random Agent for T steps
        terminal    = True
        steps       = 0
        while steps < T:
            if terminal:
                if steps != 0: self.showDomain(s,a)
                s = self.s0()
            a = randSet(self.possibleActions(s))
            # DELETE ME
#            print 'state at step',steps,'::',s
#            print 'action selected ::',a
            
            
            self.showDomain(s,a)
            r,s,terminal = self.step(s, a)
#            print "r,s,terminal",r,s,terminal
            steps += 1
    def printAll(self):
        printClass(self)
    def extendDiscreteDimensions(self):
        self.statespace_limits = self.statespace_limits.astype('float')
        for d in arange(self.state_space_dims):
             if not d in self.continous_dims:
                 self.statespace_limits[d,0] += -.5 
                 self.statespace_limits[d,1] += .5 
    def isTerminal(self,s):
        # Returns true if state s is terminal
        abstract