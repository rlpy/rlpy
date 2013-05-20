#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.' 
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT
######################################################
# State is x,y, Actions are 4 way directional with fixed noise. \n
# Each grid cell is: \n
# 0: empty \n
# 1: blocked \n
# 2: start \n
# 3: goal \n
# 4: pit \n
# The task is to reach the goal from the start while avoiding the pits
######################################################
class GridWorld(Domain):
    map = start = goal              = None
	## Used for graphics to show the domain
    agent_fig = upArrows_fig = downArrows_fig = leftArrows_fig = rightArrows_fig = domain_fig = valueFunction_fig  = None     
	## Number of rows and columns of the map
    ROWS = COLS = 0                 
    #Rewards
    GOAL_REWARD = +1
    PIT_REWARD = -1
    STEP_REWARD = -.001
	## Set by the domain = min(100,rows*cols)
    episodeCap  = None             
	## Movement Noise
    NOISE = 0                      
	## Used for graphical normalization
    MAX_RETURN  = 1                 
	## Used for graphical normalization
    MIN_RETURN  = -1                
	## Used for graphical shifting of arrows
    SHIFT       = .1                

    actions_num        = 4
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT = arange(6)
	## Up, Down, Left, Right
    ACTIONS = array([[-1,0], [+1,0], [0,-1], [0,+1] ])
    def __init__(self,mapname='/GridWorldMaps/4x5.txt', noise = .1, episodeCap = None, logger = None):
        self.map                = loadtxt(mapname, dtype = uint8)
        if self.map.ndim == 1: self.map = self.map[newaxis,:]
        self.start              = argwhere(self.map==self.START)[0]
        self.ROWS,self.COLS     = shape(self.map)
        self.statespace_limits  = array([[0,self.ROWS-1],[0,self.COLS-1]])
        self.NOISE              = noise
        self.DimNames           = ['Row','Col']
        self.episodeCap         = 1000 #2*self.ROWS*self.COLS, small values can cause problem for some planning techniques 
        super(GridWorld,self).__init__(logger)
        if logger: 
            self.logger.log("Dims:\t\t%dx%d" %(self.ROWS,self.COLS))
            self.logger.log("Movement Noise:\t%0.0f%%" %(self.NOISE*100))
    def showDomain(self,s,a = 0):
       #Draw the environment
       if self.domain_fig is None:
           self.agent_fig = pl.subplot(1,2,1)
           self.domain_fig = pl.imshow(self.map, cmap='GridWorld',interpolation='nearest',vmin=0,vmax=5)
           pl.xticks(arange(self.COLS), fontsize= FONTSIZE)
           pl.yticks(arange(self.ROWS), fontsize= FONTSIZE)
           #pl.tight_layout()
           self.agent_fig = self.agent_fig.plot(s[1],s[0],'kd',markersize=20.0-self.COLS)
           pl.show()
       #mapcopy = copy(self.map) 
       #mapcopy[s[0],s[1]] = self.AGENT
       #self.domain_fig.set_data(mapcopy)
       self.agent_fig.pop(0).remove()
       self.agent_fig = pl.subplot(1,2,1).plot(s[1],s[0],'k>',markersize=20.0-self.COLS) # Instead of '>' you can use 'D', 'o'
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
            arrow_ratio = 0.4 # length of arrow/width of bax. Less then 0.5 because each arrow is offset, 0.4 looks nice but could be better/auto generated 
            Max_Ratio_ArrowHead_to_ArrowLength = 0.25
            ARROW_WIDTH = 0.5*Max_Ratio_ArrowHead_to_ArrowLength/5.0
            self.upArrows_fig = pl.quiver(Y,X,DY,DX,C, units='y', cmap='Actions', scale_units="height", scale=self.ROWS/arrow_ratio, width = -1*ARROW_WIDTH) 
            self.upArrows_fig.set_clim(vmin=0,vmax=1)
            X   = arange(self.ROWS)+self.SHIFT
            Y   = arange(self.COLS)
            X,Y = pl.meshgrid(X,Y) 
            self.downArrows_fig = pl.quiver(Y,X,DY,DX,C, units='y', cmap='Actions', scale_units="height", scale=self.ROWS/arrow_ratio, width = -1*ARROW_WIDTH)
            self.downArrows_fig.set_clim(vmin=0,vmax=1)
            X   = arange(self.ROWS)
            Y   = arange(self.COLS)-self.SHIFT
            X,Y = pl.meshgrid(X,Y) 
            self.leftArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions', scale_units="width", scale=self.COLS/arrow_ratio, width = ARROW_WIDTH)
            self.leftArrows_fig.set_clim(vmin=0,vmax=1)
            X   = arange(self.ROWS)
            Y   = arange(self.COLS)+self.SHIFT
            X,Y = pl.meshgrid(X,Y) 
            self.rightArrows_fig = pl.quiver(Y,X,DY,DX,C, units='x', cmap='Actions', scale_units="width", scale=self.COLS/arrow_ratio, width = ARROW_WIDTH)
            self.rightArrows_fig.set_clim(vmin=0,vmax=1)
            f = pl.gcf()
#            f.set_size_inches(10,20)
            pl.show()
            #pl.tight_layout()
        V            = zeros((self.ROWS,self.COLS))
        Mask         = ones((self.COLS,self.ROWS,self.actions_num), dtype='bool') #Boolean 3 dimensional array. The third array highlights the action. Thie mask is used to see in which cells what actions should exist
        arrowSize    = zeros((self.COLS,self.ROWS,self.actions_num), dtype ='float')
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
                    #print r,c, bestA
                    #print Qs
                    
                    for i in arange(len(As)):
                        a = As[i]
                        Q = Qs[i]
                        value = linearMap(Q,self.MIN_RETURN,self.MAX_RETURN,0,1)
                        arrowSize[c,r,a] = value
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
        r           = self.STEP_REWARD
        ns          = s.copy()
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
        if self.map[ns[0],ns[1]] == self.PIT:
                r = self.PIT_REWARD
        terminal = self.isTerminal(ns)
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
    def isTerminal(self,s):
        if self.map[s[0],s[1]] == self.GOAL:
                return self.NOMINAL_TERMINATION
        if self.map[s[0],s[1]] == self.PIT:
                return self.CRITICAL_TERMINATION
        return self.NOT_TERMINATED
    def expectedStep(self,s,a):
        #Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
#        print "State:", s
#        print "Action:", a
        actions = self.possibleActions(s)
        k       = len(actions)
        #Make Probabilities
        intended_action_index = findElemArray1D(a,actions) 
        p       = ones((k,1))*self.NOISE/(k*1.)
        p[intended_action_index,0] += 1-self.NOISE  
#       print "Probabilities:", p
        #Make next states
        ns      = tile(s,(k,1)).astype(int)
        actions = self.ACTIONS[actions]
        ns     += actions
#        print "next states:", ns
        #Make rewards
        r       = ones((k,1))*self.STEP_REWARD
        goal    = self.map[ns[:,0], ns[:,1]] == self.GOAL  
        pit     = self.map[ns[:,0], ns[:,1]] == self.PIT  
        r[goal] = self.GOAL_REWARD
        r[pit]  = self.PIT_REWARD
#        print"rewards", r
        #Make terminals
        t       = zeros((k,1),bool)
        t[goal] = True
        t[pit]  = True
#        print"terminals", t
#        raw_input()
        return p,r,ns,t
        
if __name__ == '__main__':
    p = GridWorld(mapname='GridWorldMaps/4x5.txt');
    p.test(1000)
    
    
