"""Intruder monitoring task."""

from Tools import *
from Domain import Domain
import time
import os
from Tools import __rlpy_location__

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
            "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "N. Kemal Ure"


class IntruderMonitoring(Domain):
    """
    Formulated as an MDP, the intruder monitoring task is to guard danger zones using cameras
    so that if an intruder moves to a danger zone, at least one camera is pointing at that location.

    The episode is finished after 1000 steps.

    State Features (n is number of agents, m is number of intruders):
        Location of Agent_1 x ... x Location of Agent n...
        Location of Intruder 1 x ...x Location of Intruder_m

    Location is 2D position on a grid

    Actions:
        [Up, Down, Left, Right, StayStill]^n (oone action for each agent)

    Dynamics:
        Each agent can move in 4 directions + stay still, there is no noise
        Each intruder moves with a fixed policy (specified by the user)
        By Default, intruder policy is uniform random

    Map of the world contains fixed number of danger zones. Maps are simple text files
    contained in the Domains/IntruderMonitoringMaps/ directory.

    The team receives a penalty whenever there is an intruder on a danger zone in the
    absence of an agent. The task is to allocate agents on the map so that intruders
    do not enter the danger zones without attendance of an agent.

    Reward:
        -1 for every visit of an intruder to a danger zone with no camera present

    """

    map = None
	#: Number of rows and columns of the map
    ROWS = COLS = 0
	#: Number of Cooperating agents
    NUMBER_OF_AGENTS = 0
	#: Number of Intruders
    NUMBER_OF_INTRUDERS = 0
    NUMBER_OF_DANGER_ZONES = 0
    gamma = .8
    #: Rewards
    INTRUSION_PENALTY = -1.0
    episodeCap  = 100              # Episode Cap

    #Constants in the map
    EMPTY, INTRUDER, AGENT, DANGER = arange(4)
	#: Actions: Up, Down, Left, Right, Null
    ACTIONS_PER_AGENT = array([[-1,0], [+1,0], [0,-1], [0,+1], [0,0], ])

    #Visual Variables
    domain_fig      = None
    ally_fig        = None
    intruder_fig    = None

    #: directory with maps shipped with rlpy
    default_map_dir = os.path.join(__rlpy_location__, "Domains", "IntruderMonitoringMaps")

    def __init__(self, mapname=os.path.join(default_map_dir, "4x4_2A_3I.txt"), logger = None):

        self.setupMap(mapname)
        self.state_space_dims                   = 2*(self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS)

        _statespace_limits                      = vstack([[0,self.ROWS-1],[0,self.COLS-1]])
        self.statespace_limits                  = tile(_statespace_limits,((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS),1))
        #self.statespace_limits_non_extended     = tile(_statespace_limits,((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS),1))

        self.actions_num        = 5**self.NUMBER_OF_AGENTS
        self.ACTION_LIMITS      = [5]*self.NUMBER_OF_AGENTS
        self.DimNames           = []

        super(IntruderMonitoring,self).__init__(logger)
        if self.logger:
            _,_,shortmapname = mapname.rpartition('/')
            self.logger.log("Map Name:\t%s" % shortmapname)

    def setupMap(self,mapname):
        #Load the map as an array
        self.map = loadtxt(mapname, dtype = uint8)
        if self.map.ndim == 1: self.map = self.map[newaxis,:]
        self.ROWS,self.COLS = shape(self.map)

        R,C = (self.map == self.AGENT).nonzero()
        self.agents_initial_locations = vstack([R,C]).T
        self.NUMBER_OF_AGENTS = len(self.agents_initial_locations)
        R,C = (self.map == self.INTRUDER).nonzero()
        self.intruders_initial_locations = vstack([R,C]).T
        self.NUMBER_OF_INTRUDERS = len(self.intruders_initial_locations)
        R,C = (self.map == self.DANGER).nonzero()
        self.danger_zone_locations = vstack([R,C]).T
        self.NUMBER_OF_DANGER_ZONES = len(self.danger_zone_locations)

    def step(self,a):
        #Move all agents according to the action
        #Move all intruders randomly
        #Calculate the reward: Number of danger zones being violated by intruders while no agents being present
        s = self.state

        #Move all agents based on the taken action
        agents      = array(s[:self.NUMBER_OF_AGENTS*2].reshape(-1,2))
        actions     = id2vec(a,self.ACTION_LIMITS)
        actions     = self.ACTIONS_PER_AGENT[actions]
        agents      += actions

        # Generate actions for each intruder based on the function IntruderPolicy
        intruders       = array(s[self.NUMBER_OF_AGENTS*2:].reshape(-1,2))
        actions         = [self.IntruderPolicy(intruders[i]) for i in arange(self.NUMBER_OF_INTRUDERS)]
        actions         = self.ACTIONS_PER_AGENT[actions]
        intruders       += actions

        # Put all info in one big vector
        ns              = hstack((agents.ravel(),intruders.ravel()))
        #Saturate states so that if actions forced agents to move out of the grid world they bound back
        ns              =bound_vec(ns,self.discrete_statespace_limits)
        # Find agents and intruders after saturation
        agents          = ns[:self.NUMBER_OF_AGENTS*2].reshape(-1,2)
        intruders       = ns[self.NUMBER_OF_AGENTS*2:].reshape(-1,2)

        # Reward Calculation
        map = zeros((self.ROWS,self.COLS), 'bool')
        map[intruders[:,0],intruders[:,1]] = True
        map[agents[:,0],agents[:,1]] = False
        intrusion_counter = count_nonzero(map[self.danger_zone_locations[:,0],self.danger_zone_locations[:,1]])
        r = intrusion_counter*self.INTRUSION_PENALTY
        ns              =bound_vec(ns,self.discrete_statespace_limits)
        #print s, id2vec(a,self.ACTION_LIMITS), ns
        self.state = ns.copy()
        return r,ns,False, self.possibleActions()

    def s0(self):
        self.state = hstack([self.agents_initial_locations.ravel(), self.intruders_initial_locations.ravel()])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def possibleActionsPerAgent(self,s):
        # 1. tile the [R,C] for all actions
        # 2. add all actions to the results
        # 3. Find feasible rows and add them as possible actions
        tile_s              = tile(s,[len(self.ACTIONS_PER_AGENT),1])
        next_states         = tile_s + self.ACTIONS_PER_AGENT
        next_states_rows    = next_states[:,0]
        next_states_cols    = next_states[:,1]
        possibleActions1    = logical_and(0 <= next_states_rows,next_states_rows < self.ROWS)
        possibleActions2    = logical_and(0 <= next_states_cols,next_states_cols < self.COLS)
        possibleActions,_   = logical_and(possibleActions1,possibleActions2).reshape(-1,1).nonzero()
        return possibleActions
    def printDomain(self,s,a):
        print '--------------'

        for i in arange(0,self.NUMBER_OF_AGENTS):
             s_a = s[i*2:i*2+2]
             aa = id2vec(a,self.ACTION_LIMITS)
             #print 'Agent {} X: {} Y: {}'.format(i,s_a[0],s_a[1])
             print 'Agent {} Location: {} Action {}'.format(i,s_a,aa)
        offset = 2*self.NUMBER_OF_AGENTS
        for i in arange(0,self.NUMBER_OF_INTRUDERS):
            s_i = s[offset+ i*2:offset + i*2+2]
            #print 'Intruder {} X: {} Y: {}'.format(i,s_i[0],s_i[1])
            print 'Intruder',s_i
        r,ns,terminal = self.step(s, a)

        print 'Reward ',r
    def IntruderPolicy(self,s_i):
         return self.random_state.choice(self.possibleActionsPerAgent(s_i))

    def showDomain(self, a):
       s = self.state
       #Draw the environment
       if self.domain_fig is None:
           self.domain_fig  = pl.imshow(self.map, cmap='IntruderMonitoring',interpolation='nearest',vmin=0,vmax=3)
           pl.xticks(arange(self.COLS), fontsize= FONTSIZE)
           pl.yticks(arange(self.ROWS), fontsize= FONTSIZE)
           pl.show()
       if self.ally_fig != None:
           self.ally_fig.pop(0).remove()
           self.intruder_fig.pop(0).remove()

       s_ally               = s[0:self.NUMBER_OF_AGENTS*2].reshape((-1,2))
       s_intruder           = s[self.NUMBER_OF_AGENTS*2:].reshape((-1,2))
       self.ally_fig        = pl.plot(s_ally[:,1],s_ally[:,0],'bo',markersize=30.0,alpha = .7,markeredgecolor = 'k',markeredgewidth=2)
       self.intruder_fig    = pl.plot(s_intruder[:,1],s_intruder[:,0],'g>',color='gray',markersize=30.0,alpha = .7,markeredgecolor = 'k',markeredgewidth=2)
       pl.draw()
