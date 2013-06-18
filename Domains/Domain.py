######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
class Domain(object):
    name = ''               # Name of the Domain
    gamma = .90             # Discount factor default = .9
    states_num = None       # Number of states
    actions_num = None      # Number of Actions
    statespace_limits = []  # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
    state_space_dims = 0    # Number of dimensions of the state space
    continous_dims = []     # List of continuous dimensions of the domain, default = None
    episodeCap = 0          # The cap used to bound each episode (return to s0 after)
    #Termination Signals of Episodes
    NOT_TERMINATED          = 0
    NOMINAL_TERMINATION     = 1
    CRITICAL_TERMINATION    = 2

    def __init__(self):
        pass
    def show(self,s,a, representation):     
        self.showDomain(s,a)
        self.showLearning(representation)
    def showDomain(self,s,a = 0):
        abstract
    def showLearning(self,representation):
        abstract
    def s0(self):       
        # Returns the initial state
        abstract
    def possibleActions(self,s):
        # Returns the list of possible actions in each state the vanilla version returns all of the actions
        return arange(self.actions_num)
    def step(self,s,a):
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal
        abstract    
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
            self.showDomain(s,a)
            r,s,terminal = self.step(s, a)
            steps += 1
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
         