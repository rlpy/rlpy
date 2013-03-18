import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
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