######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Agent import *
class SARSA(Agent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace   = []
    eligibility_trace_s = [] # eligibility trace using state only (no copy-paste), necessary for dabney decay mode
    def __init__(self, representation, policy, domain,logger, initial_alpha =.1, lambda_ = 0, alpha_decay_mode = 'dabney', boyan_N0 = 1000):
        self.eligibility_trace  = zeros(representation.features_num*domain.actions_num)
        self.eligibility_trace_s= zeros(representation.features_num) # use a state-only version of eligibility trace for dabney decay mode
        self.lambda_            = lambda_
        super(SARSA,self).__init__(representation,policy,domain,logger,initial_alpha,alpha_decay_mode, boyan_N0)
        self.logger.log("Alpha_0:\t\t%0.2f" % initial_alpha)
        self.logger.log("Decay mode:\t\t"+str(alpha_decay_mode))
        
        if lambda_: self.logger.log("lambda:\t%0.2f" % lambda_)
    def learn(self,s,a,r,ns,na,terminal):
        super(SARSA, self).learn(s,a,r,ns,na,terminal) # increment episode count
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        phi_s               = self.representation.phi(s)
        phi_prime_s         = self.representation.phi(ns)
        phi                 = self.representation.phi_sa(s,a,phi_s)
        phi_prime           = self.representation.phi_sa(ns,na,phi_prime_s)
        
        nnz                 = count_nonzero(phi_s)    #Number of non-zero elements
        if nnz == 0: # Phi has some nonzero elements, proceed with update
            return
        
        #Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace   *= gamma*self.lambda_
            self.eligibility_trace   += phi
            
            self.eligibility_trace_s  *= gamma*self.lambda_
            self.eligibility_trace_s += phi_s
            
            #Set max to 1
            self.eligibility_trace[self.eligibility_trace>1] = 1
            self.eligibility_trace_s[self.eligibility_trace_s>1] = 1
        else:
            self.eligibility_trace    = phi
            self.eligibility_trace_s  = phi_s
        
        td_error            = r + dot(gamma*phi_prime - phi, theta)
        
        self.updateAlpha(phi_s,phi_prime_s,self.eligibility_trace_s, gamma, nnz, terminal)
        theta               += self.alpha * td_error * self.eligibility_trace

        #Discover features if the representation has the discover method
        discover_func = getattr(self.representation,'discover',None) # None is the default value if the discover is not an attribute
        if discover_func and callable(discover_func):
            self.representation.discover(phi_s,td_error)
		
        # Set eligibility Traces to zero if it is end of the episode
        if self.lambda_: self.eligibility_trace = zeros(self.representation.features_num*self.domain.actions_num) 
        # Else there are no nonzero elements, halt update.
