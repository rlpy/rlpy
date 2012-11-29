#NOT FINISHED YET
#NOT FINISHED YET
#NOT FINISHED YET
#NOT FINISHED YET
#NOT FINISHED YET
import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
#State is x,y, Actions are 4 way directional with fixed noise
#each grid cell is:
# 0: empty
# 1: blocked
# 2: start
# 3: goal
# 4: pit
# The task is to reach the goal from the start while avoiding the pits
######################################################
class CartPole(Domain):
    #Rewards
    BALANCED_REWARD = 0
    UNBALANCED_REWARD = -1
    episodeCap  = 3000              # Set by the domain = min(100,rows*cols)
    NOISE = 0                       # Movement Noise
    MAX_RETURN  = 0                 # Used for graphical normalization
    MIN_RETURN  = -1                # Used for graphical normalization
    TORQUE      = 20                #Torque value applied
    actions_num        = 3
    state_space_dims   = 2
    #Constants in the map
    ACTIONS = array([-TORQUE,0,TORQUE])
    def __init__(self, noise = .1):
        path                    = os.getcwd() + mapname
        self.map                = loadtxt(path, dtype = uint8)
        self.start              = argwhere(self.map==self.START)[0]
        self.ROWS,self.COLS     = shape(self.map)
        self.states_num         = self.ROWS * self.COLS
        self.statespace_limits  = array([[0,self.ROWS-1],[0,self.COLS-1]])
        self.NOISE              = noise
        self.episodeCap         = min(self.ROWS*self.COLS,100)
        #reduce(mul,x,1)
    def showDomain(self,s,a):
       #Draw the environment
       if self.domain_fig is None:
           pl.subplot(1,2,1)
           self.domain_fig = pl.imshow(self.map, cmap='GridWorld',interpolation='nearest',vmin=0,vmax=5)
           pl.xticks(arange(self.COLS), fontsize= FONTSIZE)
           pl.yticks(arange(self.ROWS), fontsize= FONTSIZE)
           pl.tight_layout()
           pl.show(block=False)
       mapcopy = copy(self.map) 
       mapcopy[s[0],s[1]] = self.AGENT
       self.domain_fig.set_data(mapcopy)
       pl.draw()   
    def showLearning(self,representation):
        if self.valueFunction_fig is None:
            pl.subplot(1,2,2)
            self.valueFunction_fig   = pl.imshow(self.map, cmap='ValueFunction',interpolation='nearest',vmin=self.MIN_RETURN,vmax=self.MAX_RETURN) 
            pl.xticks(arange(self.COLS), fontsize=12)
            pl.yticks(arange(self.ROWS), fontsize=12)
           #Create quivers for each action. 4 in total
            X   = arange(self.ROWS)-self.SHIFT
            Y   = arange(self.COLS)
            X,Y = pl.meshgrid(X,Y) 
            DX = DY = ones(X.shape)
            C = zeros(X.shape); C[0,0] = 1 # Making sure C has both 0 and 1             
            self.upArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions')#, headwidth=1.5, headlength = 2.5, headaxislength = 2.25)
            X   = arange(self.ROWS)+self.SHIFT
            Y   = arange(self.COLS)
            X,Y = pl.meshgrid(X,Y) 
            self.downArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions')
            X   = arange(self.ROWS)
            Y   = arange(self.COLS)-self.SHIFT
            X,Y = pl.meshgrid(X,Y) 
            self.leftArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions')
            X   = arange(self.ROWS)
            Y   = arange(self.COLS)+self.SHIFT
            X,Y = pl.meshgrid(X,Y) 
            self.rightArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions')
            f = pl.gcf()
#            f.set_size_inches(10,20)
            pl.show(block=False)
            pl.tight_layout()
        V            = zeros((self.ROWS,self.COLS))
        Mask         = ones((self.COLS,self.ROWS,self.actions_num), dtype='bool') #Boolean 3 dimensional array. The third array highlights the action. Thie mask is used to see in which cells what actions should exist
        arrowSize    = zeros((self.COLS,self.ROWS,self.actions_num))
        arrowColors  = zeros((self.COLS,self.ROWS,self.actions_num),dtype= 'uint8') # 0 = suboptimal action, 1 = optimal action
        for r in arange(self.ROWS):
            for c in arange(self.COLS):
                if self.map[r,c] == self.BLOCKED: V[r,c] = 0 
                if self.map[r,c] == self.GOAL: V[r,c] = self.MAX_RETURN  
                if self.map[r,c] == self.PIT: V[r,c] =self.MIN_RETURN 
                if self.map[r,c] == self.EMPTY or self.map[r,c] == self.START:
                    s        = [r,c]
                    Qs,As    = representation.Qs(s)
                    bestA    = representation.bestActions(s)
                    V[r,c]   = max(Qs)
                    Mask[c,r,As]             = False
                    arrowColors[c,r,bestA]   = 1
#                    print r,c,Qs
                    arrowSize[c,r,As]        = vectorize(linearMap)(Qs,self.MIN_RETURN,self.MAX_RETURN,.4,2) #Vectorize creates a function that can be applied to matrixes
#                    print vectorize(linearMap)(Qs,min(Qs),max(Qs),.4,2)
        #Show Value Function
        self.valueFunction_fig.set_data(V)
        #Show Policy Up Arrows
        DX = arrowSize[:,:,0]
        DY = zeros((self.ROWS,self.COLS))  
        DX = ma.masked_array(DX, mask=Mask[:,:,0])
        DY = ma.masked_array(DY, mask=Mask[:,:,0])
        C  = ma.masked_array(arrowColors[:,:,0], mask=Mask[:,:,0])
        self.upArrows_fig.set_UVC(DY,DX,C)
        #Show Policy Down Arrows
        DX = -arrowSize[:,:,1]
        DY = zeros((self.ROWS,self.COLS))  
        DX = ma.masked_array(DX, mask=Mask[:,:,1])
        DY = ma.masked_array(DY, mask=Mask[:,:,1])
        C  = ma.masked_array(arrowColors[:,:,1], mask=Mask[:,:,1])
        self.downArrows_fig.set_UVC(DY,DX,C)
        #Show Policy Left Arrows
        DX = zeros((self.ROWS,self.COLS))  
        DY = -arrowSize[:,:,2]
        DX = ma.masked_array(DX, mask=Mask[:,:,2])
        DY = ma.masked_array(DY, mask=Mask[:,:,2])       
        C  = ma.masked_array(arrowColors[:,:,2], mask=Mask[:,:,2])
        self.leftArrows_fig.set_UVC(DY,DX,C)
        #Show Policy Right Arrows
        DX = zeros((self.ROWS,self.COLS))  
        DY = arrowSize[:,:,3]
        DX = ma.masked_array(DX, mask=Mask[:,:,3])
        DY = ma.masked_array(DY, mask=Mask[:,:,3])
        C  = ma.masked_array(arrowColors[:,:,3], mask=Mask[:,:,3])
        self.rightArrows_fig.set_UVC(DY,DX,C)
        pl.draw()   
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
        return self.start
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
    
    