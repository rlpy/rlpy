######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
from Policy import *
class eGreedy(Policy):
    epsilon         = None
    old_epsilon     = None 
    forcedDeterministicAmongBestActions = None # This boolean variable is used to avoid random selection among actions with the same values
    def __init__(self,representation,logger,epsilon = .1, forcedDeterministicAmongBestActions = False):
        self.epsilon = epsilon
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        super(eGreedy,self).__init__(representation,logger)
    def pi(self,s):
        coin = random.rand()
        #print "coin=",coin
        if coin < self.epsilon:
            A = self.representation.domain.possibleActions(s)
            return randSet(A)
        else:
            A = self.representation.bestActions(s)
            if self.forcedDeterministicAmongBestActions:
                return A[0]
            else:
                return randSet(A)
    def turnOffExploration(self):
        self.old_epsilon = self.epsilon 
        self.epsilon = 0
    def turnOnExploration(self):
        self.epsilon = self.old_epsilon
         
        
        
        