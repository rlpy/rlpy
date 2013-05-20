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
# \author Developed by Alborz Geramiard Dec 28th 2012 at MIT
######################################################
# A domain based on the last puzzle of D&R Game stage 5-3 \n
# The goal of the game is to get all elements of a 4x4 board
# to have value 1. \n \n
# The initial state is the following \n
# 1 0 0 0 \n
# 0 0 0 0 \n
# 0 1 0 0 \n
# 0 0 1 0 \n \n
# The action is to pick a spot on the board and all elements
# on the same row and column will flip their values.
######################################################
class FlipBoard(Domain):
    gamma = 1
    BOARD_SIZE  = 4 
    STEP_REWARD = -1
    episodeCap  = 100               # Set by the domain = min(100,rows*cols)
    actions_num = BOARD_SIZE**2
    statespace_limits = tile([0,1],(BOARD_SIZE**2,1))
    
    #Visual Stuff
    domain_fig = None
    move_fig = None
    def __init__(self,logger = None):
        super(FlipBoard,self).__init__(logger)
        if logger: 
            self.logger.log("Board Size:\t\t%dx%d" %(self.BOARD_SIZE,self.BOARD_SIZE))
    def showDomain(self,s,a = 0):
       #Draw the environment
       if self.domain_fig is None:
           self.move_fig  = pl.subplot(111)
           s = s.reshape((self.BOARD_SIZE,self.BOARD_SIZE))
           self.domain_fig = pl.imshow(s, cmap='FlipBoard',interpolation='nearest',vmin=0,vmax=1)
           pl.xticks(arange(self.BOARD_SIZE), fontsize= FONTSIZE)
           pl.yticks(arange(self.BOARD_SIZE), fontsize= FONTSIZE)
           #pl.tight_layout()
           a_row,a_col      = id2vec(a,[self.BOARD_SIZE, self.BOARD_SIZE])
           self.move_fig    = self.move_fig.plot(a_col,a_row,'kx',markersize=30.0)
           pl.show()
       a_row,a_col = id2vec(a,[self.BOARD_SIZE, self.BOARD_SIZE])
       self.move_fig.pop(0).remove()
       #print a_row,a_col
       self.move_fig = pl.plot(a_col,a_row,'kx',markersize=30.0) # Instead of '>' you can use 'D', 'o'
       s = s.reshape((self.BOARD_SIZE,self.BOARD_SIZE))
       self.domain_fig.set_data(s)
       pl.draw()   
       #raw_input()
    def step(self,s,a):
        ns          = s.copy()
        ns          = reshape(ns,(self.BOARD_SIZE,-1))
        a_row,a_col = id2vec(a,[self.BOARD_SIZE, self.BOARD_SIZE])
        #print a_row, a_col
        #print ns
        ns[a_row,:] =  logical_not(ns[a_row,:])
        ns[:,a_col] =  logical_not(ns[:,a_col])
        ns[a_row,a_col] = not ns[a_row,a_col] 
        if self.isTerminal(s):
            terminal = True
            r        = 0
        else:
            terminal    = False
            r           = self.STEP_REWARD
        #sleep(1)
        ns = ns.flatten()
        return r,ns,terminal
    def s0(self):
        s = array([ [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                      ], dtype='bool')
        return s.flatten()
    def isTerminal(self,s):
        return count_nonzero(s) == self.BOARD_SIZE**2 
if __name__ == '__main__':
    p = FlipBoard();
    p.test(1000)