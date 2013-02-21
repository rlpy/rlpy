import sys, os
from numpy.ma.core import logical_or
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Developed by N. Kemal Ure Dec 3rd 2012 at MIT #
# Edited by Alborz Geramifard on Feb 20 2013 #
######################################################
# State is : Location of Agent_1 x ... x Location of Agent n...
# Location of Intruder 1 x ...x Location of Intruder_m
# n is number of agents, m is number of intruders
# Location is 2D position on a grid
# Each agent can move in 4 directions + stay still, there is no noise
# Each intruder moves with a fixed policy (specified by the user)
# By Default, intruder policy is uniform random
# Map of the world contains fixed number of danger zones,
# Team receives a penalty whenever there is an intruder
# on a danger zone in the absence of an agent  
# Task is to allocate agents on the map
# so that intruders do not enter the danger zones without 
# attendance of an agent
######################################################
class IntruderMonitoring(Domain):
    map = None
    ROWS = COLS = 0                 # Number of rows and columns of the map
    NUMBER_OF_AGENTS = 0            # Number of Cooperating agents
    NUMBER_OF_INTRUDERS = 0         # Number of Intruders
    NUMBER_OF_DANGER_ZONES = 0
    gamma = .95
    #Rewards
    INTRUSION_PENALTY = -1.0
    episodeCap  = 1000              # Episode Cap
    
    #Constants in the map
    EMPTY, INTRUDER, AGENT, DANGER = arange(4)
    ACTIONS_PER_AGENT = array([[-1,0], #Up
               [+1,0], #Down
               [0,-1], #left
               [0,+1], #Right
               [0,0], # Null
               ])

    #Visual Variables
    domain_fig      = None    
    ally_fig        = None
    intruder_fig    = None
    
    def __init__(self, mapname, logger = None):
                 
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
        if self.logger: self.logger.log("Dims:\t\t%dx%d" %(self.ROWS,self.COLS))
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
    def step(self,s,a):
        #Move all agents according to the action
        #Move all intruders randomly
        #Calculate the reward: Number of danger zones being violated by intruders while no agents being present
        
        #Move all agents based on the taken action
        agents      = array(s[:self.NUMBER_OF_AGENTS*2].reshape(-1,2))
        actions     = id2vec(a,self.ACTION_LIMITS)
        actions     = self.ACTIONS_PER_AGENT[actions]
        agents      += actions
        
        # Generate uniform random actions for intruders and move them
        intruders       = array(s[self.NUMBER_OF_AGENTS*2:].reshape(-1,2))
        actions         = random.randint(len(self.ACTIONS_PER_AGENT), size=self.NUMBER_OF_INTRUDERS)
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
        return r,ns,False
    def s0(self):
        return hstack([self.agents_initial_locations.ravel(), self.intruders_initial_locations.ravel()])
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
           self.domain_fig  = pl.imshow(self.map, cmap='IntruderMonitorying',interpolation='nearest',vmin=0,vmax=3)
           pl.xticks(arange(self.COLS), fontsize= FONTSIZE)
           pl.yticks(arange(self.ROWS), fontsize= FONTSIZE)
           pl.show()
       if self.ally_fig != None:
           self.ally_fig.pop(0).remove()
           self.intruder_fig.pop(0).remove()

       s_ally               = s[0:self.NUMBER_OF_AGENTS*2].reshape((-1,2))
       s_intruder           = s[self.NUMBER_OF_AGENTS*2:].reshape((-1,2)) 
       self.ally_fig        = pl.plot(s_ally[:,1],s_ally[:,0],'b>',markersize=30.0,alpha = .7,markeredgecolor = 'k',markeredgewidth=2)
       self.intruder_fig    = pl.plot(s_intruder[:,1],s_intruder[:,0],'go',color='gray',markersize=30.0,alpha = .7,markeredgecolor = 'k',markeredgewidth=2)
       pl.draw()   
       sleep(0)
if __name__ == '__main__':
   
    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/1x3_1A_1I.txt')
    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/2x3_2A_1I.txt')
    #p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/4x4_1A_2I.txt')
    random.seed(99)
    p = IntruderMonitoring(mapname = 'IntruderMonitoringMaps/4x4_2A_3I.txt')
    p.test(1000)
    
    