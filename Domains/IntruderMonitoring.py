import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
######################################################
# Developed by N. Kemal Ure Dec 3rd 2012 at MIT #
######################################################
# State is : Location of Agent_1 x ... x Location of Agent n...
# Location of Intruder 1 x ...x Location of Intruder_m
# n is number of agents, m is number of intruders
# Location is 2D position on a grid
# Each agent can move in 4 directions, there is no noise
# Each intruder moves with a fixed policy (specified by the user)
# By Default, intruder policy is uniform random
# Map of the world contains fixed number of danger zones,
# Team receives a penalty whenever there is an intruder
# on a danger zone in the absence of an agent  
# Task is to collobratively allocate agents on the map
# so that intruders do not enter the danger zones without 
# attendance of an agent
######################################################
class IntruderMonitoring(Domain):
    map = None
    ROWS = COLS = 0                 # Number of rows and columns of the map
    NUMBER_OF_AGENTS = 0            # Number of Cooperating agents
    NUMBER_OF_INTRUDERS = 0         # Number of Intruders
    NUMBER_OF_DANGER_ZONES = 0
    #Rewards
    INTRUSION_PENALTY = -1.0
    episodeCap  = 0                 # Set by the domain = min(100,rows*cols)
        
    
    #Constants in the map
    EMPTY, INTRUDER, AGENT, DANGER = range(4)
    ACTIONS_PER_AGENT = array([[-1,0], #Up
               [+1,0], #Down
               [0,-1], #left
               [0,+1] #Right
               ])
      
       
    def __init__(self,logger, mapname, episodeCap = None):
                 
        path                    = os.getcwd() + mapname
        self.map                = loadtxt(path, dtype = uint8)
        if self.map.ndim == 1: self.map = self.map[newaxis,:]
        
        self.ROWS,self.COLS     = shape(self.map)
        self.GetAgentAndIntruderNumbers()
        
        _statespace_limits = vstack([[0,self.ROWS-1],[0,self.COLS-1]])
        self.statespace_limits      = tile(_statespace_limits,((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS),1))     
                
        self.states_num = ((self.ROWS-1)*(self.COLS-1))^((self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS))    
        self.state_space_dims = 2*(self.NUMBER_OF_AGENTS + self.NUMBER_OF_INTRUDERS)             
        self.actions_num        = 4*self.NUMBER_OF_AGENTS
        self.ACTION_LIMITS = [4]*self.NUMBER_OF_AGENTS
                                      
        if episodeCap is None:
            self.episodeCap         = 2*self.ROWS*self.COLS
        else:
            self.episodeCap         = episodeCap
        
        '''
        print 'Initialization Finished'    
        print 'Number of Agents', self.NUMBER_OF_AGENTS
        print 'Number of Intruders', self.NUMBER_OF_INTRUDERS
        print 'Initial State',self.s0()
        print 'Possible Actions',self.possibleActions(self.s0())
        print 'limits', self.statespace_limits
        '''
            
        super(IntruderMonitoring,self).__init__(logger)
        self.logger.log("Dims:\t\t%dx%d" %(self.ROWS,self.COLS))
   
    def step(self,s,a):
                    
        action_vector = id2vec(a,self.ACTION_LIMITS)
        r = 0
                                          
        # Agent Step
                      
        ns = []            
                        
        for i in range(0,self.NUMBER_OF_AGENTS):
            s_a = s[i*2:i*2+2]
                                              
            action_a = action_vector[i]
            ns_a = list(s_a)#s_a.copy()                                   
            #ns_a = ns_a + self.ACTIONS_PER_AGENT[action_a]
            self.move(ns_a, action_a)
                                
            #if(ns_a[0] < 0 or ns_a[0] == self.ROWS-1 or ns_a[1] < 0 or ns_a[1] == self.COLS-1):
            #    ns_a = s_a
            
                       
            # Merge the state
           
            ns +=  ns_a
            
        # Intruder Step
        
        intrusion_counter = 0
        offset = 2*self.NUMBER_OF_AGENTS                
        for i in range(0,self.NUMBER_OF_INTRUDERS):
            s_i = s[offset+ i*2:offset+i*2+2]
            ns_i = list(s_i)#s_i.copy()    
            action_i =   randSet(self.possibleActionsPerAgent(s_i))
                                    
            #ns_i = ns_i + self.ACTIONS_PER_AGENT[action_i]
            self.move(ns_i, action_i)
                    
           # if(ns_i[0] < 0 or ns_i[0] == self.ROWS-1 or ns_i[1] < 0 or ns_i[1] == self.COLS-1):
           #     ns_i = s_i  
                
            # Check if there is an intrusion
            IntruderMonitoring
            if self.map[ns_i[0],ns_i[1]] == self.DANGER: # Intruder is in a danger zone !!
                
                for j in range(0,self.NUMBER_OF_AGENTS):
                    ns_a=  ns[j*2:j*2+2]
                    if (ns_a != ns_i): # Intrusion occured !
                       # print 'Intrusion !!'
                        intrusion_counter += 1
                         
             # Merge the state
            ns += ns_i                         
                
        # Reward Calculation           
       
        self.saturateState(ns)
        r = intrusion_counter*self.INTRUSION_PENALTY
                
        return r,ns,False
    
    def move(self,s,a):
        
        if (a == 0):
            s[0] -= 1
        if (a == 1):
            s[0] += 1
        if (a == 2):
            s[1] -= 1
        if (a == 3):
            s[1] += 1      
        
        
    
    def s0(self):
        
        s_init = []
        ns_a = []
        ns_i = []
        
        for r in arange(self.ROWS):
            for c in arange(self.COLS):
                if self.map[r,c] == self.AGENT: ns_a += [r,c]
                if self.map[r,c] == self.INTRUDER: ns_i += [r,c]
                                      
        #s_init.append(ns_a)
        #s_init.append(ns_i)
        s_init = ns_a + ns_i
        
        return s_init
    
        
    def possibleActions(self,s):
              
       possibleA = array([],uint8)
       
       for a in arange(self.actions_num):
               possibleA = append(possibleA,[a])
           
       
       return possibleA
    
    
    def isTerminal(self,s):
       
        return False
    
    def possibleActionsPerAgent(self,s):
              
        possibleA = array([],uint8)
        for a in arange(self.actions_num):
            ns = s + self.ACTIONS_PER_AGENT[a]
            if (
                ns[0] < 0 or ns[0] == self.ROWS or
                ns[1] < 0 or ns[1] == self.COLS):
                continue
            possibleA = append(possibleA,[a])
        return possibleA
    
    def GetAgentAndIntruderNumbers(self):
         
        
        for r in arange(self.ROWS):
              for c in arange(self.COLS):
                if self.map[r,c] == self.AGENT: self.NUMBER_OF_AGENTS +=1
                if self.map[r,c] == self.INTRUDER: self.NUMBER_OF_INTRUDERS +=1
                                
    def showDomain(self,s,a):
        print '--------------'
       
        for i in range(0,self.NUMBER_OF_AGENTS):
             s_a = s[i*2:i*2+2]
             aa = id2vec(a,self.ACTION_LIMITS)
             #print 'Agent {} X: {} Y: {}'.format(i,s_a[0],s_a[1])
             print 'Agent {} Location: {} Action {}'.format(i,s_a,aa)
        offset = 2*self.NUMBER_OF_AGENTS
        for i in range(0,self.NUMBER_OF_INTRUDERS):
            s_i = s[offset+ i*2:offset + i*2+2]
            #print 'Intruder {} X: {} Y: {}'.format(i,s_i[0],s_i[1])
            print 'Intruder',s_i    
        r,ns,terminal = self.step(s, a)
        
        print 'Reward ',r               
                              
                           
if __name__ == '__main__':
   
    p = IntruderMonitoring(logger = None, mapname = '/IntruderMonitoringMaps/4x4_1A_1I.txt')
    p.test(100)
    
    