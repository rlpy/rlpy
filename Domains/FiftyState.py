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
    GOAL_STATES = [9,40] # Indices of states with rewards
    episodeCap  = 50             # Set by the domain = min(100,rows*cols)
    MAX_RETURN  = 2.5             # Used for graphical normalization
    MIN_RETURN  = 0             # Used for graphical normalization
    SHIFT       = .01            # Used for graphical shifting of arrows
    RADIUS      = .05            # Used for graphical radius of states
    circles     = None          # Stores the graphical pathes for states so that we can later change their colors 
    chainSize   = 50             # Number of states in the chain
    Y           = 1             # Y values used for drawing circles
    actions_num = 2
    p_action_failure = 0.1      # Probability of taking the other (unselected) action
    V_star      = None          # Array of optimal values at each state
    
    optimalPolicy = None        # The optimal policy for this domain
    using_optimal_policy = False # Should the domain only allow optimal actions
    
    #Plotting values
    domain_fig          = None
    value_function_fig  = None
    policy_fig          = None
    V_star_line         = None
    V_approx_line       = None
    COLORS              = ['g','k']
    LEFT = 0
    RIGHT = 1
    
    #Constants in the map
    def __init__(self,logger = None):
        self.start              = 0
        self.statespace_limits  = array([[0,self.chainSize-1]])
        super(FiftyState,self).__init__(logger)
        self.optimal_policy = array( [-1 for dummy in range(0, self.chainSize)]) # To catch errors
        self.storeOptimalPolicy()
        self.gamma = 0.8 # Set gamma to be 0.8 for this domain per L & P 2007
    def storeOptimalPolicy(self):
        self.optimal_policy[arange(self.GOAL_STATES[0])] = self.RIGHT            
        goalStateIndices = arange(1,len(self.GOAL_STATES))
        for i in goalStateIndices:
            goalState1 = self.GOAL_STATES[i-1]
            goalState2 = self.GOAL_STATES[i]
            averageState = int(mean([goalState1, goalState2]))
            self.optimal_policy[arange(goalState1, averageState)] = self.LEFT
            self.optimal_policy[arange(averageState, goalState2)] = self.RIGHT
        self.optimal_policy[arange(self.GOAL_STATES[-1], self.chainSize)] = self.LEFT
        print self.optimal_policy
    def showDomain(self,s,a = 0):
        #Draw the environment
        if self.circles is None:
           self.domain_fig = pl.subplot(3,1,1)
           pl.figure(1, (self.chainSize*2/10.0, 2))
           self.domain_fig.set_xlim(0, self.chainSize*2/10.0)
           self.domain_fig.set_ylim(0, 2)
           self.domain_fig.add_patch(mpatches.Circle((1/5.0+2/10.0*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
           self.domain_fig.xaxis.set_visible(False)
           self.domain_fig.yaxis.set_visible(False)
           self.circles = [mpatches.Circle((1/5.0+2/10.0*i, self.Y), self.RADIUS, fc="w") for i in range(self.chainSize)]
           for i in range(self.chainSize):
               self.domain_fig.add_patch(self.circles[i])
               pl.show()
            
        [p.set_facecolor('w') for p in self.circles]
        [self.circles[p].set_facecolor('g') for p in self.GOAL_STATES]
        self.circles[s].set_facecolor('k')
        pl.draw()
    def showLearning(self, representation):
        allStates = arange(0,self.chainSize)
        X   = arange(self.chainSize)*2.0/10.0-self.SHIFT
        Y   = ones(self.chainSize) * self.Y
        DY  = zeros(self.chainSize)  
        DX  = zeros(self.chainSize)
        C   = zeros(self.chainSize)

        if self.value_function_fig is None:
            self.value_function_fig = pl.subplot(3,1,2)
            self.V_star = zeros(self.chainSize) #### # Value function at each state.
            self.V_star_line = self.value_function_fig.plot(allStates,self.V_star)
            V   = [representation.V(s) for s in allStates]
            
            # Note the comma below, since a tuple of line objects is returned
            self.V_approx_line, = self.value_function_fig.plot(allStates, self.V_star, 'r-')
            self.V_star_line    = self.value_function_fig.plot(allStates, V, 'b--')
            pl.ylim([0, self.GOAL_REWARD * (len(self.GOAL_STATES)+1)]) # Maximum value function is sum of all possible rewards
            
            self.policy_fig = pl.subplot(3,1,3)
            self.policy_fig.set_xlim(0, self.chainSize*2/10.0)
            self.policy_fig.set_ylim(0, 2)
            self.arrows = pl.quiver(X,Y,DX,DY, C, cmap = 'fiftyChainActions', units='x', width=0.05,scale=.008, alpha= .8 )#headwidth=.05, headlength = .03, headaxislength = .02)
            self.policy_fig.xaxis.set_visible(False)
            self.policy_fig.yaxis.set_visible(False)
        
        V   = [representation.V(s) for s in allStates]
        pi  = [representation.bestAction(s) for s in allStates]
        #pi  = [self.optimal_policy[s] for s in allStates]
        
        DX  = [(2*a-1)*self.SHIFT*.1 for a in pi]
        
        self.V_approx_line.set_ydata(V)
        self.arrows.set_UVC(DX,DY,pi)
        pl.draw()
    def step(self,s,a):
        a = self.optimal_policy[s]
        actionFailure = (random.random() < self.p_action_failure)
        if a == self.LEFT or (a == self.RIGHT and actionFailure): #left
            ns = max(0,s-1)
        elif a == self.RIGHT or (a == self.LEFT and actionFailure):
            ns = min(self.chainSize-1,s+1)
        terminal = self.isTerminal(ns)
        r = self.GOAL_REWARD if s in self.GOAL_STATES else 0
        return r,ns,terminal
    def s0(self):
        return random.randint(0,self.chainSize)
    def isTerminal(self,s):
        return False # s == [[]]
    def possibleActions(self,s):
        if self.using_optimal_policy: return array([self.optimal_policy[s]])
        else: return arange(self.actions_num)
    

if __name__ == '__main__':
    p = FiftyState();
    p.test(1000)
    
    