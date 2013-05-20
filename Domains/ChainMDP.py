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
# \author Developed by Alborz Geramiard Nov 20th 2012 at MIT #
######################################################
# A simple Chain MDP \n
# s0 <-> s1 <-> ... <-> sn \n  
# Actions are left [0] and right [1] \n
# The task is to reach sn from s0.
# Optimal policy is always to go right
######################################################
class ChainMDP(Domain):
    GOAL_REWARD = 0
    STEP_REWARD = -1
	## Set by the domain = min(100,rows*cols)
    episodeCap  = 0             
	## Used for graphical normalization
    MAX_RETURN  = 1             
	## Used for graphical normalization
    MIN_RETURN  = 0             
	## Used for graphical shifting of arrows
    SHIFT       = .3            
	## Used for graphical radius of states
    RADIUS      = .5            
	## Stores the graphical pathes for states so that we can later change their colors 
    circles     = None          
	## Number of states in the chain
    chainSize   = 0             
	## Y values used for drawing circles
    Y           = 1             
    actions_num = 2
    #Constants in the map
    def __init__(self, chainSize=2,logger = None):
        self.chainSize          = chainSize
        self.start              = 0
        self.goal               = chainSize - 1
        self.statespace_limits  = array([[0,chainSize-1]])
        self.episodeCap         = 2*chainSize
        super(ChainMDP,self).__init__(logger)
    def showDomain(self,s,a = 0):
        #Draw the environment
        s = s[0]
        if self.circles is None:
           fig = pl.figure(1, (self.chainSize*2, 2))
           ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
           ax.set_xlim(0, self.chainSize*2)
           ax.set_ylim(0, 2)
           ax.add_patch(mpatches.Circle((1+2*(self.chainSize-1), self.Y), self.RADIUS*1.1, fc="w")) #Make the last one double circle
           ax.xaxis.set_visible(False)
           ax.yaxis.set_visible(False)
           self.circles = [mpatches.Circle((1+2*i, self.Y), self.RADIUS, fc="w") for i in arange(self.chainSize)]
           for i in arange(self.chainSize):
               ax.add_patch(self.circles[i])
               if i != self.chainSize-1:
                    fromAtoB(1+2*i+self.SHIFT,self.Y+self.SHIFT,1+2*(i+1)-self.SHIFT, self.Y+self.SHIFT)
                    if i != self.chainSize-2: fromAtoB(1+2*(i+1)-self.SHIFT,self.Y-self.SHIFT,1+2*i+self.SHIFT, self.Y-self.SHIFT, 'r')
               fromAtoB(.75,self.Y-1.5*self.SHIFT,.75,self.Y+1.5*self.SHIFT,'r',connectionstyle='arc3,rad=-1.2')
               pl.show()
            
        [p.set_facecolor('w') for p in self.circles]
        self.circles[s].set_facecolor('k')
        pl.draw()
    def step(self,s,a):
        s = s[0]
        if a == 0: #left
            ns = max(0,s-1)
        if a == 1:
            ns = min(self.chainSize-1,s+1)
        ns = array([ns])
        terminal = self.isTerminal(ns)
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r,ns,terminal
    def s0(self):
        return array([0])
    def isTerminal(self,s):
        return (s[0] == self.chainSize - 1)

if __name__ == '__main__':
    #p = GridWorld('/Domains/GridWorldMaps/ACC2011.txt');
    p = ChainMDP(5);
    p.test(1000)
    
    