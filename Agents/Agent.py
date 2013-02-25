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
    def MC_episode(self,s=None,a=None):
        # Run a single monte-carlo simulation episode from state s with action a following the current policy of the agent and return:
        # eps_return, eps_length, eps_term, eps_discounted_return
        eps_length              = 0
        eps_return              = 0
        eps_discounted_return   = 0
        eps_term                = 0
        if s is None: s = self.domain.s0()
        if a is None: a = self.policy.pi(s)
        terminal    = False
        while not eps_term and eps_length < self.domain.episodeCap:
            r,ns,eps_term       = self.domain.step(s, a)
            s                   = ns
            eps_return          += r
            eps_discounted_return += self.representation.domain.gamma**eps_length * r
            eps_length          += 1
            a                   = self.policy.pi(s)
        return eps_return, eps_length, eps_term, eps_discounted_return
        
    def Q_MC(self,s,a,MC_samples = 1000):
        # Use Monte-Carlo samples with the fixed policy to evaluate the Q(s,a)
        Q_avg = 0
        for i in arange(MC_samples):
            #print "MC Sample:", i
            _,_,_,Q = self.MC_episode(s,a)
            Q_avg = incrementalAverageUpdate(Q_avg,Q,i+1)
        return Q_avg
    def evaulate(self,samples, MC_samples, output_file):
        # Evaluate the current policy for fixed number of samples and store them in samples-by-|S|+2 (2 corresponds to action and Q(s,a))
        # inputs: 
        # samples: number of samples (s,a)
        # MC_samples: Number of MC simulations used to estimate Q(s,a)
        # output_file: The DATA is stored in this file
        
        cols            = self.domain.state_space_dims + 2
        DATA            = empty((samples,cols))
        terminal        = True
        steps           = 0
        while steps < samples:
            s = self.domain.s0() if terminal or steps % self.domain.episodeCap == 0 else s 
            a = self.policy.pi(s)

            #Store the corresponding Q
            Q = self.Q_MC(s,a,MC_samples)
            DATA[steps,:] = hstack((s,[a, Q]))
            r,s,terminal = self.domain.step(s, a)
            steps += 1
            
            self.logger.log("Sample "+ str(steps)+":"+ str(s)+" "+str(a)+" "+str(Q)) 

        save(output_file, DATA)
        return DATA
    
	## @endcond
