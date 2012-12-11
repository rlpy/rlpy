######################################################
# Developed by N. Kemal Ure Dec 10th 2012 at MIT #
######################################################
# Classical Value Iteration
# Performs full Bellman Backup on a given s,a pair  
from Agent import *
class ValueIteration(Agent):
    mc_ns_samples = 0         # Number of next state samples used to approximate BE expectation          
        
    def __init__(self,representation,policy,domain,logger, mc_ns_samples= 10):
        self.phi_sa_size        = domain.actions_num * representation.features_num
        self.mc_ns_samples = mc_ns_samples
        super(ValueIteration, self).__init__(representation, policy, domain,logger)
        
    def learn(self,s,a,r = None,ns = None,na = None ,terminal = None):
        #print ' Theta, ', self.representation.theta
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        
        # Sample Next States and rewards
        
        next_states,rewards = self.domain.MCExpectedStep(s,a,self.mc_ns_samples)
        
        prob = 1/float(self.mc_ns_samples)
        #print 'prob',prob
        # Loop over next States
        
        update = 0
        
        for i in range(0,self.mc_ns_samples):
                        
            ns =   array(next_states[i,:],int)
            #print 'next state', ns            
            r = rewards[i]
            #print 'reward', r
            Qns = []                          
            # Get the best action for the next state
            
            next_actions = self.domain.possibleActions(ns)
            
            for j in next_actions:
                
                phi_ns = self.representation.phi(ns)
                
                phi_n = self.representation.phi_sa(ns,j,phi_ns)
                
                #print 'phi_n', phi_n 
                              
                Qns.append(dot(phi_n,theta)) 
            
            #print 'Qns', Qns 
                           
            val = max(Qns)    
            #print 'val', val
            
            update += prob*(r + gamma*val)
            
            #if(len(self.domain.possibleActions(ns)) != 0):
               
                
                
             #   update += prob*(r + gamma*val)
                #print 'update', update
                
            #print 'update', update
        
        # Update the representation
        
        s_index = vec2id( self.representation.binState(s),self.representation.bins_per_dim)
        #print 's_index',s_index
        
        theta_index = self.domain.states_num*a + s_index
        #print 'theta_index', theta_index
        self.representation.theta[theta_index]  =  update
                
        
        
        
        
        
      