import sys, os
import copy

import csv
#



#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RL-Python/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RL-Python'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)


from Tools import *
from Domain import *

########################################################
# \author Trevor Campbell Mar 7 2013 at MIT
########################################################
# Multiagent Multitarget tracking mission. \n
# Goal is to try to capture each target
# (i.e. land on top or beside each target, with
# capture probability 0.8 and 0.2 respectively). \n
# Capturing a target provides a reward of 10. \n
# Once a target is captured, it is removed from
# the scenario (there is a binary vector denoting
# availability of targets in the state)

########################################################

# @author Trevor Campbell
class MultiTrack(Domain):

    episodeCap          = 1000 # 100 used in tutorial, 1000 in matlab
    gamma               = 0.9  # 0.9 used in tutorial and matlab
    NUM_AGENTS          = 1
    NUM_TARGETS         = 1
    GRID                = 6
    ASTEP               = 2
    TSTEP               = 1
    motionNoise         = 0

    ###
    def __init__(self, GRID = 6, ASTEP = 2, TSTEP = 1, NUM_AGENTS = 1, NUM_TARGETS = 1, motionNoise = 0, logger = None):
        #Domain-specific
        self.NUM_AGENTS             = NUM_AGENTS
        self.NUM_TARGETS            = NUM_TARGETS
        self.GRID                   = GRID
        self.ASTEP                  = ASTEP
        self.TSTEP                  = TSTEP
        self.motionNoise            = motionNoise

        #Inherited
        self.episodeCap = 1000
        self.states_num = (GRID*GRID)*(NUM_AGENTS+NUM_TARGETS)*2**(NUM_TARGETS)
        self.actions_num = ((ASTEP*2+1)*(ASTEP*2+1))**(NUM_AGENTS)
        locations_lim = array(tile([0,GRID*GRID-1],((NUM_AGENTS+NUM_TARGETS),1)))
        binvec_lim = array(tile([0,1], (NUM_TARGETS, 1)))
        self.statespace_limits = vstack([locations_lim, binvec_lim])
        self.ALIMITS = (self.ASTEP*2+1)*(self.ASTEP*2+1)*ones(NUM_AGENTS, dtype='int') # eg [3,3,3,3], number of possible actions
        #self.discrete_statespace_limits = None
        #self.state_space_dims = None
        #self.continuous_dims = []
        super(MultiTrack,self).__init__(logger)
        #if self.logger: self.logger.log("NUM_UAV:\t\t%d" % self.NUM_UAV)

    def showDomain(self,s,a = 0):
        pass

    def showLearning(self,representation):
        pass

    def step(self,s,a):
        #return reward, terminalness, etc
        ss = self.state2Struct(s)
        nss = ss


        #For each agent, move them deterministically with their preferred action
        actionVector = array(id2vec(a,self.ALIMITS)) # returns list of form [0,1,0,2] corresponding to action of each uav

        for i in range(0, self.NUM_AGENTS):
           ac = actionVector[i]
           dy = floor(ac/(self.ASTEP*2+1)) - self.ASTEP
           dx = ac%(self.ASTEP*2+1) - self.ASTEP
           nss.agentlocs[i] = LocStruct(ss.agentlocs[i].x + dx, ss.agentlocs[i].y + dy, -1, self.GRID)


        #For each target, move them stochastically/deterministically depending on motionNoise
        for i in range(0,self.NUM_TARGETS):
            dx = 0
            dy = 0
            #decide whether to be random or not
            if random.random() < self.motionNoise:
                #stochastic motion
                dx = random.randint(-self.TSTEP, self.TSTEP)
                dy = random.randint(-self.TSTEP, self.TSTEP)
            else:
                #nominal motion
                #dictionary for nominal dx/dy - targets move left,right,up,down
                nomdict = {0 : (1, 0), 1 : (-1, 0), 2 : (0, 1), 3 : (0, -1)}
                dx, dy = nomdict[i%4]
                if self.TSTEP > 1:
                    dx = dx*random.randint(1, self.TSTEP)
                    dy = dy*random.randint(1, self.TSTEP)

            if self.worldContains(ss.targetlocs[i].x + dx, ss.targetlocs[i].y+dy):
                nss.targetlocs[i] = LocStruct(ss.targetlocs[i].x + dx, ss.targetlocs[i].y + dy, -1, self.GRID)
            else:
                nss.targetlocs[i] = LocStruct(ss.targetlocs[i].x, ss.targetlocs[i].y, -1, self.GRID)
                nss.targetavail[i] = 0;

        rwd = 0
        #Calculate stochastic termination for each target
        for i in range(0, self.NUM_AGENTS):
            ax = nss.agentlocs[i].x
            ay = nss.agentlocs[i].y
            for j in range(0, self.NUM_TARGETS):
                tx = nss.targetlocs[j].x
                ty = nss.targetlocs[j].y
                if ss.targetavail[j] == 1 and (abs(ax-tx) == 1 or abs(ay-ty) == 1):
                    if random.random() < 0.2:
                        nss.targetavail[j] = 0
                        rwd = rwd + 10
                elif ss.targetavail[j] == 1 and abs(ax-tx) == 0 and abs(ay-ty) == 0:
                    if random.random() < 0.8:
                        nss.targetavail[j] = 0
                        rwd = rwd + 10
            #otherwise, targetavail is left alone (still equal to the previous state's target avail

        ns = self.struct2State(nss)
        return rwd,ns,self.isTerminal(ns)
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal

    def s0(self):
        alocs = []
        tlocs = []
        for i in range(0, self.NUM_AGENTS):
            loc = LocStruct(-1, -1, random.randint(0, self.GRID*self.GRID-1), self.GRID)
            alocs.append(loc)
        for i in range(0, self.NUM_TARGETS):
            loc = LocStruct(-1, -1, random.randint(0, self.GRID*self.GRID-1), self.GRID)
            tlocs.append(loc)
        binvec = ones(self.NUM_TARGETS, dtype='int')
        return self.struct2State(StateStruct(alocs, tlocs, binvec))

    # @return: the tuple (locations, fuel, actuator, sensor), each an array indexed by uav_id
    def state2Struct(self,s):
        alocs = [LocStruct(-1, -1, idx, self.GRID) for idx in s[0:self.NUM_AGENTS]]
        tlocs = [LocStruct(-1, -1, idx, self.GRID) for idx in s[self.NUM_AGENTS:self.NUM_AGENTS+self.NUM_TARGETS]]
        binvec = s[self.NUM_AGENTS+self.NUM_TARGETS:self.NUM_AGENTS+2*self.NUM_TARGETS]
        return StateStruct(alocs, tlocs, binvec)

    def struct2State(self,s):
        return hstack([ [aloc.idx for aloc in s.agentlocs], [tloc.idx for tloc in s.targetlocs], s.targetavail])

    def worldContains(self, x, y):
        return True if 0 <= x and x < self.GRID and 0 <= y and y < self.GRID  else False

    def possibleActions(self,s):
        # return the id of possible actions
        # find empty blocks (nothing on top)
        validActions = [] # Contains a list of uav_actions lists, e.g. [[0,1,2],[0,1],[1,2]] with the index corresponding to a uav.
        # First, enumerate the possible actions for each uav
        ss = self.state2Struct(s)
        for aid in range(0, self.NUM_AGENTS):
            acs = []
            ax = ss.agentlocs[aid].x
            ay = ss.agentlocs[aid].y
            for dx in range(-self.ASTEP, self.ASTEP+1):
                for dy in range (-self.ASTEP,self.ASTEP+1):
                    if self.worldContains(ax+dx, ay+dy):
                        dxshift = dx + self.ASTEP
                        dyshift = dy + self.ASTEP
                        acs.append(dyshift*(self.ASTEP*2+1)+dxshift) #converts the delta x, y into an action index

            validActions.append(acs)

        return array(self.vecList2id(validActions, (self.ASTEP*2+1)*(self.ASTEP*2+1)))


            # Given a list of lists 'validActions' of the form [[0,1,2],[0,1],[1,2],[0,1]]... return
        # unique id for each permutation between lists; eg above, would return 3*2*2*2 values
        # ranging from 0 to 3^4 -1 (3 is max value possible in each of the lists)
        # Takes parameters of totalActionSpaceSize, here 3^4-1, and number of actions available
        # to each agent, here 3

            ####### TODO place this in Tools
            # Given a list of lists of the form [[0,1,2],[0,1],[1,2],[0,1]]... return
            # unique id for each permutation between lists; eg above, would return 3*2*2*2 values
            # ranging from 0 to 3^4 -1 (3 is max value possible in each of the lists, maxValue)
    def vecList2id(self,x,maxValue):
    #returns a list of unique id's based on possible permutations of this list of lists
        _id = 0
        actionIDs = []
        curActionList = []
        lenX = len(x)
        limits = tile(maxValue, (1,lenX))[0] # eg [3,3,3,3] # TODO redundant computation
        self.vecList2idHelper(x,actionIDs,0, curActionList, maxValue,limits) # TODO remove self

        return actionIDs

    def vecList2idHelper(self,x,actionIDs,ind,curActionList, maxValue,limits):
    #returns a list of unique id's based on possible permutations of this list of lists.  See vecList2id
        for curAction in x[ind]: # x[ind] is one of the lists, e.g [0, 2] or [1,2]
            partialActionAssignment = curActionList[:]
            partialActionAssignment.append(curAction)
            if(ind == len(x) - 1): # We have reached the final list, assignment is complete
#                print partialActionAssignment,',,',limits
                actionIDs.append(vec2id(partialActionAssignment, limits)) # eg [0,1,0,2] and [3,3,3,3]
            else:
                self.vecList2idHelper(x,actionIDs,ind+1,partialActionAssignment, maxValue,limits) # TODO remove self
#        return actionIDs
    def isTerminal(self,s):
        ss = self.state2Struct(s)
        return all([True if x == 0 else False for x in ss.targetavail])

## \cond DEV
class LocStruct:
    def __init__(self, x, y, idx, GRID):
        if 0 <= idx and idx < GRID*GRID:
            self.GRID = GRID
            self.idx = idx
            self.y = floor(idx/GRID)
            self.x = idx%GRID
        elif 0 <= x and 0 <= y and x < GRID and y < GRID :
            self.GRID = GRID
            self.x = x
            self.y = y
            self.idx = self.y*self.GRID+self.x
        else:
            assert(False)


class StateStruct:
    def __init__(self, agentlocs, targetlocs, targetavail):
        self.agentlocs  = agentlocs
        self.targetlocs = targetlocs
        self.targetavail= targetavail
# \endcond

if __name__ == '__main__':
        random.seed(0)
        p = MultiTrack(GRID = 10, TSTEP = 1, ASTEP = 2, NUM_AGENTS = 1, NUM_TARGETS = 1)
        p.test(1000)

        # Code below was used to test output of this domain for various actions,
        # confirmed alignment with MATLAB version (see bobtest there)

#        allA = arange(27)
#
#        s = p.s0()
#        for i in arange(20): # Number of steps desired for test
#            a = 17 - i
#            print 'pythontest: original action ',a
##            print 'pythontest: vector action ', array(id2vec(a,p.ALIMITS))
#            (r, s, isT) = p.step(s,a)
#            print 'pythontest: new state, reward, and possible a', s, r, p.possibleActions(s)
#
#        actionVectors = [array(id2vec(a,p.ALIMITS)) for a in allA]
#
#        a_aVect_tups = zip(allA, actionVectors)
#        for a_aVect_tup in a_aVect_tups:
#            print a_aVect_tup



#        x = array([[1,2,0],[1,2],[0,1],[0]])
#        q = p.vecList2id(x, 3)
#        print x, q
#
#        x = array([[1,2,0],[1],[0,1],[0]])
#        q = p.vecList2id(x, 3)
#        print x, q
#
#        x = array([[1,2,0],[2],[0,1],[0]])
#        q = p.vecList2id(x, 3)
#        print x, q
#
#        x = array([[1,2,0],[2,1],[0,1],[0]])
#        q = p.vecList2id(x, 3)
#        print x, q
#






