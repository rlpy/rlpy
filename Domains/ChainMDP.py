import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Developed by Alborz Geramiard Nov 20th 2012 at MIT #
######################################################
# A simple Chain MDP
# s0 <-> s1 <-> ... <-> sn  
# actions are left [0] and right [1]
# The task is to reach sn from s0
# optimal policy is always to go right
######################################################
class ChainMDP(Domain):
    GOAL_REWARD = 0
    STEP_REWARD = -1
    gamma       = 1
    episodeCap  = 0             # Set by the domain = min(100,rows*cols)
    MAX_RETURN  = 1             # Used for graphical normalization
    MIN_RETURN  = 0             # Used for graphical normalization
    SHIFT       = .3            # Used for graphical shifting of arrows
    RADIUS      = .5            # Used for graphical radius of states
    circles     = None          # Stores the graphical pathes for states so that we can later change their colors 
    chainSize   = 0             # Number of states in the chain
    Y           = 1             # Y values used for drawing circles
    actions_num = 2
    #Constants in the map
    def __init__(self,logger, chainSize=2):
        self.chainSize          = chainSize
        self.start              = 0
        self.goal               = chainSize - 1
        self.statespace_limits  = array([[0,chainSize-1]])
        self.episodeCap         = 2*chainSize
        super(ChainMDP,self).__init__(logger)
    def showDomain(self,s,a = 0):
        #Draw the environment
        if self.circles is None:
           fig = pl.figure(1, (self.chainSize*2, 2))
           ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
           ax.set_xlim(0, self.chainSize*2)
           ax.set_ylim(0, 2)
           ax.add_patch(mpatches.Circle((1+2*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
           ax.xaxis.set_visible(False)
           ax.yaxis.set_visible(False)
           self.circles = [mpatches.Circle((1+2*i, self.Y), self.RADIUS, fc="w") for i in range(self.chainSize)]
           for i in range(self.chainSize):
               ax.add_patch(self.circles[i])
               if i != self.chainSize-1:
                    fromAtoB(1+2*i+self.SHIFT,self.Y+self.SHIFT,1+2*(i+1)-self.SHIFT, self.Y+self.SHIFT)
                    if i != self.chainSize-2: fromAtoB(1+2*(i+1)-self.SHIFT,self.Y-self.SHIFT,1+2*i+self.SHIFT, self.Y-self.SHIFT, 'r')
               fromAtoB(.75,self.Y-1.5*self.SHIFT,.75,self.Y+1.5*self.SHIFT,'r',connectionstyle='arc3,rad=-1.2')
               pl.show(block=False)
            
        [p.set_facecolor('w') for p in self.circles]
        self.circles[s].set_facecolor('k')
        pl.draw()
    def step(self,s,a):
        if a == 0: #left
            ns = max(0,s-1)
        if a == 1:
            ns = min(self.chainSize-1,s+1)
        terminal = self.isTerminal(ns)
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r,ns,terminal
    def s0(self):
        return 0
    def isTerminal(self,s):
        return (s == self.chainSize - 1)
if __name__ == '__main__':
    #p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
    p = ChainMDP(5);
    p.test(1000)
    
    