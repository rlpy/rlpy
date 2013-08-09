#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys, os

#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from numpy.ma.core import logical_or
from Tools import *
from Domain import *
######################################################
# \author Developed by N. Kemal Ure Dec 3rd 2012 at MIT
# \author Edited by Alborz Geramifard on Feb 20 2013
######################################################
# State is : Location of Agent_1 x ... x Location of Agent n... \n
# Location of Intruder 1 x ...x Location of Intruder_m \n
# n is number of agents, m is number of intruders \n
# Location is 2D position on a grid \n
# Each agent can move in 4 directions + stay still, there is no noise \n
# Each intruder moves with a fixed policy (specified by the user) \n
# By Default, intruder policy is uniform random \n
# Map of the world contains fixed number of danger zones,
# Team receives a penalty whenever there is an intruder
# on a danger zone in the absence of an agent  \n
# Task is to allocate agents on the map
# so that intruders do not enter the danger zones without
# attendance of an agent
######################################################
class IntruderMonitoring(Domain):
    map = None
	## Number of rows and columns of the map
    ROWS = COLS = 0
	## Number of Cooperating agents
    NUMBER_OF_AGENTS = 0
	## Number of Intruders
    NUMBER_OF_INTRUDERS = 0
    NUMBER_OF_DANGER_ZONES = 0
    gamma = .8
    ## Rewards
    INTRUSION_PENALTY = -1.0
    episodeCap  = 1000              # Episode Cap

    #Constants in the map
    EMPTY, INTRUDER, AGENT, DANGER = arange(4)
	## Actions: Up, Down, Left, Right, Null
    ACTIONS_PER_AGENT = array([[-1,0], [+1,0], [0,-1], [0,+1], [0,0], ])

    #Visual Variables
    domain_fig      = None
    ally_fig        = None
    intruder_fig    = None

    def __init__(self, mapname = './Domains/IntruderMonitoringMaps/4x4_2A_3I.txt', logger = None):

        self.setupMap(mapname)
        self.state_space_dims                   = 2*(self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS)

        _statespace_limits                      = vstack([[0,self.ROWS-1],[0,self.COLS-1]])
        self.statespace_limits                  = tile(_statespace_limits,((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS),1))
        #self.statespace_limits_non_extended     = tile(_statespace_limits,((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS),1))

        self.actions_num        = 5**self.NUMBER_OF_AGENTS
        self.ACTION_LIMITS      = [5]*self.NUMBER_OF_AGENTS
        self.DimNames           = []

        '''
        print 'Initialization Finished'
        print 'Number of Agents', self.NUMBER_OF_AGENTS
        print 'Number of Intruders', self.NUMBER_OF_INTRUDERS
        print 'Initial State',self.s0()
        print 'Possible Actions',self.possibleActions(self.s0())
        print 'limits', self.statespace_limits
        '''

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
        ns              = self.saturateState(ns)
        # Find agents and intruders after saturation
        agents          = ns[:self.NUMBER_OF_AGENTS*2].reshape(-1,2)
        intruders       = ns[self.NUMBER_OF_AGENTS*2:].reshape(-1,2)

        # Reward Calculation
        map = zeros((self.ROWS,self.COLS), 'bool')
        map[intruders[:,0],intruders[:,1]] = True
        map[agents[:,0],agents[:,1]] = False
        intrusion_counter = count_nonzero(map[self.danger_zone_locations[:,0],self.danger_zone_locations[:,1]])
        r = intrusion_counter*self.INTRUSION_PENALTY
        self.saturateState(ns)
        #print s, id2vec(a,self.ACTION_LIMITS), ns
        self.state = ns.copy()
        return r,ns,False

    def s0(self):
        self.state = hstack([self.agents_initial_locations.ravel(), self.intruders_initial_locations.ravel()])
        return self.state.copy()

#    def possibleActions(self,s):
#
#       possibleA = array([],uint8)
#
#       for a in arange(self.actions_num):
#               possibleA = append(possibleA,[a])
#
#
#       return possibleA
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
         return randSet(self.possibleActionsPerAgent(s_i))
    def showDomain(self,s,a):
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
       #pl.gcf().savefig('domain.pdf', transparent=True, pad_inches=.1)
if __name__ == '__main__':

    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/1x3_1A_1I.txt')
    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/2x3_2A_1I.txt')
    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/4x4_1A_2I.txt')
    random.seed(99)
    p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/4x4_2A_3I.txt')
    p.test(1000)


