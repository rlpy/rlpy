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
    episodeCap              = 1000
    GOAL_STATE              = hstack(([0],arange(0,blocks-1))) # [0 0 1 2 3 .. blocks-2] meaning block 0 on the table and all other stacked on top of e
    domain_fig              = None  #Used to plot the domain
    def __init__(self, blocks = 6, towerSize = 6, noise = .1):
        self.blocks             = blocks    
        self.towerSize          = towerSize    
        self.noise              = noise 
        self.TABLE              = blocks+1
        self.actions_num        = blocks*blocks
        self.gamma              = 1
        self.statespace_limits  = array([0,blocks-1]* (blocks)) #Block i is on top of what? if block i is on top of block i => block i is on top of table
        self.states_num         = sum([nchoosek(blocks,i)*factorial(blocks-i)*pow(i,blocks-i) for i in range(blocks)])
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
                i,j = findElem(B,world)
                if len(i):
                    world[j+1,i] = A+1 #0 is white thats why!
                else:
                    # Put it in the back of the list
                    undrawn_blocks = hstack((undrawn_blocks,[A]))
        print world
        if self.domain_fig == None:
            pl.imshow(world, cmap='BlocksWorld', origin='lower', interpolation='nearest')#,vmin=0,vmax=self.blocks)
            self.domain_fig = pl.figure(1,figsize=(14, 10))
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
        terminal    = self.NOT_TERMINATED
        r           = self.STEP_REWARD
        ns          = s
        if random.random_sample() < self.NOISE:
            #Random Move  
            a = randSet(self.possibleActions(s))
        ns = s + self.ACTIONS[a]
        
        if (ns[0] < 0 or ns[0] == self.ROWS or
            ns[1] < 0 or ns[1] == self.COLS or
            self.map[ns[0],ns[1]] == self.BLOCKED):
                ns = s
        if self.map[ns[0],ns[1]] == self.GOAL:
                r = self.GOAL_REWARD
                terminal = self.NOMINAL_TERMINATION
        if self.map[ns[0],ns[1]] == self.PIT:
                r = self.PIT_REWARD
                terminal = self.CRITICAL_TERMINATION
        return r,ns,terminal
    def s0(self):
        # all blocks on table
        return arange(self.blocks)
    def possibleActions(self,s):
        possibleA = array([],uint8)
        for a in arange(self.actions_num):
            ns = s + self.ACTIONS[a]
            if (
                ns[0] < 0 or ns[0] == self.ROWS or
                ns[1] < 0 or ns[1] == self.COLS or
                self.map[ns[0],ns[1]] == self.BLOCKED):
                continue
            possibleA = append(possibleA,[a])
        return possibleA
         
if __name__ == '__main__':
    #p = PitMaze('/Domains/PitMazeMaps/ACC2011.txt');
    p = PitMaze('/PitMazeMaps/4by5.txt');
    p.test(1000)
    
    