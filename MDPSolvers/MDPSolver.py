######################################################
# Developed by N. Kemal Ure Dec 24th 2012 at MIT #
######################################################
from Tools import *
from Representations import *
class MDPSolver(object):
    representation      = None # Link to the representation object 
    domain              = None         # Link to the domain object
    logger              = None          #A simple objects that record the prints in a file
    def __init__(self,representation,domain,logger):
        self.representation = representation
        
        self.domain = domain
        self.logger = logger
        if self.logger:
            self.logger.line()
            self.logger.log("Solver:\t\t"+str(className(self)))
            #self.logger.log("Policy:\t\t"+str(className(self.policy)))    
    def solve(self):
        abstract
    def printAll(self):
        printClass(self)  
    def MCBellmanBackup(self,s,a,mc_ns_samples):
        
        # Perform Monte-Carlo Bellman Backup on a given s,a pair
        
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        
        # Sample Next States and rewards
        
        next_states,rewards = self.domain.MCExpectedStep(s,a,mc_ns_samples)
        
        prob = 1/float(mc_ns_samples)
      
        # Loop over next States
        
        update = 0
        
        for i in arange(0,mc_ns_samples):
            
            #print 'next states', next_states            
            ns =   array(next_states[i,:],int)
              
            r = rewards[i]
            #print 'reward', r
            
            val = 0
            
            phi_ns = self.representation.phi(ns)
            if(len(self.domain.possibleActions(ns))>0):   
                val = self.representation.V(ns,phi_ns)
            
            update += prob*(r + gamma*val)
            
        # Update the representation
        
        s_index = vec2id( self.representation.binState(s),self.representation.bins_per_dim)
              
        theta_index = self.domain.states_num*a + s_index
      
        self.representation.theta[theta_index]  =  update
        
    def performanceRun(self,total_steps):
        # Set Exploration to zero and sample one episode from the domain
        eps_length  = 0
        eps_return  = 0
        eps_term    = 0
        
        s           = self.domain.s0()
        terminal    = False
        #if self.show_performance: 
        #    self.domain.showLearning(self.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.representation.bestAction(s)
            #if self.show_performance: 
                #self.domain.showDomain(s,a)
             #   pl.title('After '+str(total_steps)+' Steps')

            r,ns,eps_term    = self.domain.step(s, a)
            #self.logger.log("TEST"+str(eps_length)+"."+str(s)+"("+str(a)+")"+"=>"+str(ns))
            s               = ns
            eps_return     += r
            eps_length     += 1
        #if self.show_performance: self.domain.showDomain(s,a)
        #self.agent.policy.turnOnExploration()
        return eps_return, eps_length, eps_term   
                    