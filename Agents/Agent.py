######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from Representations import *

class Agent(object):
    representation      = None # Link to the representation object 
    domain              = None         # Link to the domain object
    policy              = None         # Link to the policy object
    initial_alpha       = 0             #initial learning rate
    alpha               = 0             #Learning rate to be used
    candid_alpha        = 0             #Candid Learning Rate Calculated by [Dabney W. 2012]
    eligibility_trace   = []            #In case lambda parameter in SARSA definition is used
    logger              = None          #A simple objects that record the prints in a file
    episode_count       = 0             # Used by some alpha_decay modes
    alpha_decay_mode    = None          # Decay mode of learning rate; 'boyan' or 'dabney (automatic)'
    
    valid_decay_modes   = ['dabney','boyan'] # decay modes with an implementation [use all lowercase]
    # Automatic learning rate: [Dabney W. 2012]
    boyan_N0            = 0             # N0 parameter for boyan learning rate decay
    # 
    
    def __init__(self,representation,policy,domain,logger,initial_alpha = 0.1,alpha_decay_mode = 'dabney', boyan_N0 = 1000):
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
            
    # Defined by the domain            
    def learn(self,s,a,r,ns,na,terminal):
        if terminal: self.episode_count += 1
        # ABSTRACT
        
    ## Computes a new alpha for this agent based on @var self.alpha_decay_mode.
    # Note that we divide by number of active features in SARShe A
    # We pass the phi corresponding to the STATE, *NOT* the copy/pasted
    # phi_s_a.
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
    
