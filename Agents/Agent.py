##\file
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT
######################################################

from Tools import *
from Representations import *

## The \c %Agent class controls the actions of an RL agent 
# 
# \c %Agent provides the basic framework for RL agents to interact with the Domain,
# Representation, Policy, and Domain classes. All new agent implimentations should inherit
# from \c %Agent.
#
# Describe what role agents do exactly here.

class Agent(object):
	## Link to the representation object 
    representation      = None 
	## Link to the domain object
    domain              = None         
	## Link to the policy object
    policy              = None         
	## Initial learning rate
    initial_alpha       = 0             
	## Learning rate to be used
    alpha               = 0             
	## Candid Learning Rate Calculated by [Dabney W 2012]
    candid_alpha        = 0             
	## In case lambda parameter in SARSA definition is used
    eligibility_trace   = []            
	## A simple objects that record the prints in a file
    logger              = None          
	## Used by some alpha_decay modes
    episode_count       = 0             
	## Decay mode of learning rate; 'boyan' or 'dabney (automatic)'
    alpha_decay_mode    = None          
    
	## decay modes with an implementation [use all lowercase]
    valid_decay_modes   = ['dabney','boyan'] 
    # Automatic learning rate: [Dabney W. 2012]
	##	N0 parameter for boyan learning rate decay
    boyan_N0            = 0             
    # 
    
	
	## Initializes the \ %agent object.
    def __init__(self,representation,policy,domain,logger,initial_alpha = 0.1,alpha_decay_mode = 'dabney', boyan_N0 = 1000):
		## @var initial_alpha 
		# default is 0.1
		# @var alpha_decay_mode
		# default is 'dabney'
		# @var boyan_N0 
		# default is 1000
        self.representation = representation
        self.policy     = policy
        self.domain     = domain
        self.logger     = logger
        self.initial_alpha = initial_alpha
        self.alpha      = initial_alpha
        self.alpha_decay_mode = alpha_decay_mode.lower()
        self.boyan_N0   = boyan_N0  
        if self.logger:
            self.logger.line()
            self.logger.log("Agent:\t\t"+str(className(self)))
            self.logger.log("Policy:\t\t"+str(className(self.policy)))
            if self.alpha_decay_mode == 'boyan': self.logger.log("boyan_N0:\t%.1f"%self.boyan_N0)
        # Check to make sure selected alpha_decay mode is valid
        if not self.alpha_decay_mode in self.valid_decay_modes:
            errMsg = "Invalid decay mode selected:"+self.alpha_decay -_mode+".\nValid choices are: "+str(self.valid_decay_modes)
            if self.logger:
                self.logger.log(errMsg)
            else: shout(errMsg)
            sys.exit(1)
		# Note that initial_alpha should be set to 1 for automatic learning rate; otherwise,
		# initial_alpha will act as a permanent upper-bound on alpha.
        if self.alpha_decay_mode == 'dabney':
            self.initial_alpha = 1.0
            self.alpha = 1.0
            
    ## \b ABSTRACT \b METHOD: Defined by the domain            
    def learn(self,s,a,r,ns,na,terminal):
        if terminal: self.episode_count += 1
        # ABSTRACT
        
    ## Computes a new alpha for this agent based on self.alpha_decay_mode.
    # Note that we divide by number of active features in SARShe A
    # We pass the phi corresponding to the STATE, *NOT* the copy/pasted
    # phi_s_a.
	# @param nnz 
	# The number of nonzero features
	# @param terminal
	# Boolean that determines if the step is terminal or not
    def updateAlpha(self,phi_s, phi_prime_s, eligibility_trace_s, gamma, nnz, terminal):
        if self.alpha_decay_mode == 'dabney':
        # We only update alpha if this step is non-terminal; else phi_prime becomes
        # zero and the dot product below becomes very large, creating a very small alpha
            if not terminal:
                #Automatic learning rate: [Dabney W. 2012]
                self.candid_alpha    = abs(dot(gamma*phi_prime_s-phi_s,eligibility_trace_s)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.p
                self.candid_alpha    = 1/(self.candid_alpha*1.) if self.candid_alpha != 0 else inf
                self.alpha          = min(self.alpha,self.candid_alpha)
            # else we take no action
        elif self.alpha_decay_mode == 'boyan':
            self.alpha = self.initial_alpha * (self.boyan_N0 + 1.) / (self.boyan_N0 + self.episode_count ** 1.1)
            self.alpha /= nnz # divide by number of nonzero features; note that this method is only called if nnz > 0
        else:
            shout("Unrecognized decay mode")
            self.logger.log("Unrecognized decay mode ")    
        
    def printAll(self):
        printClass(self)
		
	## @cond DEV
    def checkPerformance(self):
        # This function should not be here. This is just for debugging and getting insight into the performance evolution
        # Set Exploration to zero and sample one episode from the domain
        eps_length  = 0
        eps_return  = 0
        eps_term    = 0
        self.policy.turnOffExploration()
        s           = self.domain.s0()
        terminal    = False

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.policy.pi(s)
            r,ns,eps_term   = self.domain.step(s, a)
            s               = ns
            eps_return     += r
            eps_length     += 1
        self.policy.turnOnExploration()
        return eps_return, eps_length, eps_term
    
	## @endcond