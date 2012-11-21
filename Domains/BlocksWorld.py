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
    trajCap                 = 1000
    STEP_REWARD             = -.001
    GOAL_REWARD             = 1
    blocks                  = 0    # Total number of blocks
    towersize               = 0    # Goal tower size   
    episodeCap              = 100
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
        #print s
        #print a,':',A,'=>',B
        ns          = s
        if random.random_sample() < self.noise:
            B = A #Drop on Table
        
        if self.validAction(s,A,B):
            ns[A] = B # A is on top of B now.
        
        terminal    = self.isTerminal(s)
        r           = self.GOAL_REWARD if terminal else self.STEP_REWARD
         
        #print ns
        return r,ns,terminal
    def s0(self):
        # all blocks on table
        return arange(self.blocks)
    def possibleActions(self,s):
        # return the id of possible actions
        # find empty blocks (nothing on top)
        empty_blocks    = [b for b in arange(self.blocks) if (
                                                              (s[b] == b and len(findElemArray1D(b,s)) == 1) or #sitting alone on the table
                                                              len(findElemArray1D(b,s)) == 0) # nothing is on it
                           ]
        #print "Empty Blocks", empty_blocks
        empty_num       = len(empty_blocks)
        actions         = [[a,b] for a in empty_blocks for b in empty_blocks if s[a] != b]
        return array([vec2id(x,[self.blocks, self.blocks]) for x in actions])
    def validAction(self,s,A,B):
        #Returns true if B can be put on A
        position = findElemArray1D(B,s)
        return (A==B or # Destination is Table
                 len(position) == 0 or #Nothing is on block B
                 position == B # Only B is on B => B is on table
                 )
    def isTerminal(self,s):
        return array_equal(s,self.GOAL_STATE)
if __name__ == '__main__':
    #p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
    random.seed(0)
    p = BlocksWorld(noise=0);
    p.test(1000)
    
    