import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Developed by Alborz Geramiard Nov 12th 2012 at MIT #
######################################################
# Blocks world with n blocks. Goal is to make an order list of towers
# Transitions are stochastic.
# Reward 1 at goal, .001 per move
# Episodic 
# state is a vector of length blocks. Each dimension can have 'blocks' number of possibilities.
# if the value of dimension d is d it means d is on the table (trick to save space)
# [0 1 2 3 4 0] => means all blocks on table except block 5 which is on top of block 0 
######################################################
class BlocksWorld(Domain):
    STEP_REWARD             = -.001
    GOAL_REWARD             = 1
    gamma                   = 1
    blocks                  = 0    # Total number of blocks
    towersize               = 0    # Goal tower size   
    episodeCap              = 1000
    domain_fig              = None  #Used to plot the domain
    def __init__(self, blocks = 6, towerSize = 6, noise = .3):
        self.blocks             = blocks    
        self.towerSize          = towerSize    
        self.noise              = noise 
        self.TABLE              = blocks+1
        self.actions_num        = blocks*blocks
        self.gamma              = 1
        self.statespace_limits  = tile([0,blocks-1],(blocks,1)) #Block i is on top of what? if block i is on top of block i => block i is on top of table
        self.real_states_num    = sum([nchoosek(blocks,i)*factorial(blocks-i)*pow(i,blocks-i) for i in range(blocks)]) #This is the true size of the state space refer to [Geramifard11_ICML]
        self.GOAL_STATE         = hstack(([0],arange(0,blocks-1))) # [0 0 1 2 3 .. blocks-2] meaning block 0 on the table and all other stacked on top of e
        super(BlocksWorld,self).__init__()
    def showDomain(self,s,a =0):
        #Draw the environment
        world           = zeros((self.blocks,self.blocks),'uint8')
        undrawn_blocks  = arange(self.blocks)
        while len(undrawn_blocks):
            A = undrawn_blocks[0]
            B = s[A]
            undrawn_blocks = undrawn_blocks[1:]
            if B == A: #=> A is on Table
                world[0,A] = A+1 #0 is white thats why!
            else:
                # See if B is already drawn
                i,j = findElemArray2D(B+1,world)
                if len(i):
                    world[i+1,j] = A+1 #0 is white thats why!
                else:
                    # Put it in the back of the list
                    undrawn_blocks = hstack((undrawn_blocks,[A]))
        if self.domain_fig == None:
            self.domain_fig = pl.imshow(world, cmap='BlocksWorld', origin='lower', interpolation='nearest')#,vmin=0,vmax=self.blocks)
            pl.xticks(arange(self.blocks), fontsize= FONTSIZE)
            pl.yticks(arange(self.blocks), fontsize= FONTSIZE)
            #pl.tight_layout()
            pl.axis('off')
            pl.show(block=False)
        else:
            self.domain_fig.set_data(world)
            pl.draw()   
    def showLearning(self,representation):
        pass #cant show 6 dimensional value function
    def step(self,s,a):
        [A,B] = id2vec(a,[self.blocks, self.blocks]) #move block A on top of B
        print 'taking action %d->%d' % (A,B)
        if not self.validAction(s,A,B):
            print 'State:%s, Invalid move from %d to %d' % (str(s),A,B)
            print self.possibleActions(s)
            print id2vec(self.possibleActions(s),[self.blocks, self.blocks])
            raw_input()

        if random.random_sample() < self.noise: B = A #Drop on Table
        ns          = s.copy()
        ns[A]       = B # A is on top of B now.
        terminal    = self.isTerminal(s)
        r           = self.GOAL_REWARD if terminal else self.STEP_REWARD
        print 'ns:', ns 
        return r,ns,terminal
    def expectedStep(self,s,a):
        [A,B] = id2vec(a,[self.blocks, self.blocks]) #move block A on top of B
        if not self.validAction(s,A,B):
            print 'Invalid move from %d to %d' % (A,B)
            return
        #Scenario 1: Success
        ns = s.copy()
        ns[A] = B # A is on top of B now.
        terminal    = self.isTerminal(s)
        r           = self.GOAL_REWARD if terminal else self.STEP_REWARD
    def s0(self):
        # all blocks on table
        return arange(self.blocks)
    def possibleActions(self,s):
        # return the id of possible actions
        # find empty blocks (nothing on top)
        empty_blocks    = [b for b in arange(self.blocks) if self.clear(b,s)]
        empty_num       = len(empty_blocks)
        actions         = [[a,b] for a in empty_blocks for b in empty_blocks if not self.destination_is_table(a,b) or not self.on_table(a,s)] #condition means if A sits on the table you can not pick it and put it on the table
        print 'state',s
        print "Empty Blocks", empty_blocks
        print actions
        raw_input()
        return array([vec2id(x,[self.blocks, self.blocks]) for x in actions])
    def validAction(self,s,A,B):
        #Returns true if B and A are both empty.
        return (self.clear(A,s) and (self.destination_is_table(A,B) or self.clear(B,s)))
    def isTerminal(self,s):
        return array_equal(s,self.GOAL_STATE)
    def top(self,A,s):
        #returns the block on top of block A. Return [] if nothing is on top of A
        on_A = findElemArray1D(A,s)
        on_A = setdiff1d(on_A,[A]) # S[i] = i is the key for i is on table.
        return on_A
    def clear(self,A,s):
        # returns true if block A is clear and can be moved
        return len(self.top(A,s)) == 0
    def destination_is_table(self,A,B):
        # See for move A->B, B is table
        return (A==B)
    def on_table(self,A,s):
        #returns true of A is on the table
        return s[A] == A
if __name__ == '__main__':
    random.seed(0)
    p = BlocksWorld(blocks=3,noise=0);
    p.test(1000)
    
    