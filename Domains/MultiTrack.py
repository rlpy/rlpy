"""Multitrack Mission"""
from Tools import *
from Domain import Domain


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Trevor Campbell"


class MultiTrack(Domain):
    """
    Multiagent Multitarget tracking mission.
    Goal is to try to capture each target
    (i.e. land on top or beside each target, with
    capture probability 0.8 and 0.2 respectively).
    Capturing a target provides a reward of 10.
    Once a target is captured, it is removed from
    the scenario (there is a binary vector denoting
    availability of targets in the state)

    """

    episodeCap          = 1000 # 100 used in tutorial, 1000 in matlab
    gamma               = 0.9  # 0.9 used in tutorial and matlab
    NUM_AGENTS          = 1
    NUM_TARGETS         = 1
    GRID                = 6
    ASTEP               = 2
    TSTEP               = 1
    motionNoise         = 0

    ###
    def __init__(self, GRID = 6, ASTEP = 2, TSTEP = 1, NUM_AGENTS = 1,
                 NUM_TARGETS = 1, motionNoise = 0, logger = None):
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

    def step(self, a):
        s = self.state
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
            if self.random_state.random_sample() < self.motionNoise:
                #stochastic motion
                dx = self.random_state.randint(-self.TSTEP, self.TSTEP)
                dy = self.random_state.randint(-self.TSTEP, self.TSTEP)
            else:
                #nominal motion
                #dictionary for nominal dx/dy - targets move left,right,up,down
                nomdict = {0 : (1, 0), 1 : (-1, 0), 2 : (0, 1), 3 : (0, -1)}
                dx, dy = nomdict[i%4]
                if self.TSTEP > 1:
                    dx = dx*self.random_state.randint(1, self.TSTEP)
                    dy = dy*self.random_state.randint(1, self.TSTEP)

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
                    if self.random_state.random_sample() < 0.2:
                        nss.targetavail[j] = 0
                        rwd = rwd + 10
                elif ss.targetavail[j] == 1 and abs(ax-tx) == 0 and abs(ay-ty) == 0:
                    if self.random_state.random_sample() < 0.8:
                        nss.targetavail[j] = 0
                        rwd = rwd + 10
            #otherwise, targetavail is left alone (still equal to the previous state's target avail

        ns = self.struct2State(nss)
        self.state = ns.copy()
        return rwd,ns,self.isTerminal(), self.possibleActions()
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal

    def s0(self):
        alocs = []
        tlocs = []
        for i in range(0, self.NUM_AGENTS):
            loc = LocStruct(-1, -1, self.random_state.randint(0, self.GRID*self.GRID-1), self.GRID)
            alocs.append(loc)
        for i in range(0, self.NUM_TARGETS):
            loc = LocStruct(-1, -1, self.random_state.randint(0, self.GRID*self.GRID-1), self.GRID)
            tlocs.append(loc)
        binvec = ones(self.NUM_TARGETS, dtype='int')
        self.state = self.struct2State(StateStruct(alocs, tlocs, binvec))
        return self.state.copy(), False, self.possibleActions()

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

    def possibleActions(self):
        s = self.state
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
        """
        returns a list of unique id's based on possible permutations of this list of lists
        """
        _id = 0
        actionIDs = []
        curActionList = []
        lenX = len(x)
        limits = tile(maxValue, (1,lenX))[0] # eg [3,3,3,3] # TODO redundant computation
        self.vecList2idHelper(x,actionIDs,0, curActionList, maxValue,limits) # TODO remove self

        return actionIDs

    def vecList2idHelper(self,x,actionIDs,ind,curActionList, maxValue,limits):
        """
        returns a list of unique id's based on possible permutations of this list of lists.  See vecList2id
        """
        for curAction in x[ind]: # x[ind] is one of the lists, e.g [0, 2] or [1,2]
            partialActionAssignment = curActionList[:]
            partialActionAssignment.append(curAction)
            if(ind == len(x) - 1): # We have reached the final list, assignment is complete
                actionIDs.append(vec2id(partialActionAssignment, limits)) # eg [0,1,0,2] and [3,3,3,3]
            else:
                self.vecList2idHelper(x,actionIDs,ind+1,partialActionAssignment, maxValue,limits) # TODO remove self

    def isTerminal(self):
        ss = self.state2Struct(self.state)
        return all([True if x == 0 else False for x in ss.targetavail])


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
