import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Alborz Geramifard, Robert H. Klein at MIT, Dec. 21, 2012
######################################################
# 50-state chain described by Lagoudakis & Parr, 2003
# s0 <-> s1 <-> ... <-> s50  
# actions are left [0] and right [1]
# Reward of +1 at states 10 and 41 (indices 0 and 9)
# Actions succeed with probability .9, otherwise execute opposite action.
######################################################
class FiftyState(Domain):
    GOAL_REWARD = 1
    GOAL_STATES = [9,40]        # Indices of states with rewards
    episodeCap  = 0             # Set by the domain = min(100,rows*cols)
    MAX_RETURN  = 1             # Used for graphical normalization
    MIN_RETURN  = 0             # Used for graphical normalization
    SHIFT       = .3            # Used for graphical shifting of arrows
    RADIUS      = .05            # Used for graphical radius of states
    circles     = None          # Stores the graphical pathes for states so that we can later change their colors 
    chainSize   = 0             # Number of states in the chain
    Y           = 1             # Y values used for drawing circles
    actions_num = 2
    p_action_failure = 0.1      # Probability of taking the other (unselected) action
    #Constants in the map
    def __init__(self, chainSize=2,logger = None):
        self.chainSize          = chainSize
        self.start              = 0
        self.goal               = chainSize - 1
        self.statespace_limits  = array([[0,chainSize-1]])
        self.episodeCap         = 2*chainSize
        super(FiftyState,self).__init__(logger)
    def showDomain(self,s,a = 0):
        #Draw the environment
        if self.circles is None:
           fig = pl.figure(1, (self.chainSize*2/10.0, 2))
           ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
           ax.set_xlim(0, self.chainSize*2/10.0)
           ax.set_ylim(0, 2)
           ax.add_patch(mpatches.Circle((1/5.0+2/10.0*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
           ax.xaxis.set_visible(False)
           ax.yaxis.set_visible(False)
           self.circles = [mpatches.Circle((1/5.0+2/10.0*i, self.Y), self.RADIUS, fc="w") for i in range(self.chainSize)]
           for i in range(self.chainSize):
               ax.add_patch(self.circles[i])
               pl.show()
            
        [p.set_facecolor('w') for p in self.circles]
        [self.circles[p].set_facecolor('g') for p in self.GOAL_STATES]
        self.circles[s].set_facecolor('k')
        pl.draw()
    def step(self,s,a):
        actionFailure = (random.random() < self.p_action_failure)
        if a == 0 or (a == 1 and actionFailure): #left
            ns = max(0,s-1)
        elif a == 1 or (a == 0 and actionFailure):
            ns = min(self.chainSize-1,s+1)
        terminal = self.isTerminal(ns)
        r = self.GOAL_REWARD if s in self.GOAL_STATES else 0
        return r,ns,terminal
    def s0(self):
        return 0
    def isTerminal(self,s):
        return False # s == [[]]

if __name__ == '__main__':
    #p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
    p = FiftyState(50);
    p.test(1000)
    
    