######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Agent import *
class SARSA(Agent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace = []  #
    def __init__(self, representation, policy, domain,logger, initial_alpha =.1, lambda_ = 0):
        self.eligibility_trace  = zeros(representation.features_num*domain.actions_num)
        self.lambda_            = lambda_
        self.alpha              = initial_alpha 
        super(SARSA,self).__init__(representation,policy,domain,logger)
        self.logger.log("Alpha_0:\t\t%0.2f" % initial_alpha)
        if lambda_: self.logger.log("lambda:\t%0.2f" % lambda_)
    def learn(self,s,a,r,ns,na,terminal):
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        phi_s               = self.representation.phi(s)
        phi                 = self.representation.phi_sa(s,a,phi_s)
        phi_prime           = self.representation.phi_sa(ns,na)
        
        #Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace   *= gamma*self.lambda_
            self.eligibility_trace   += phi
            #Set max to 1
            self.eligibility_trace[self.eligibility_trace>1] = 1
        else:
            self.eligibility_trace    = phi
        
        td_error            = r + dot(gamma*phi_prime - phi, theta)

        #Automatic learning rate: [Dabney W. 2012]
        self.candid_alpha    = abs(dot(phi-gamma*phi_prime,self.eligibility_trace)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        self.candid_alpha    = 1/(self.candid_alpha*1.) if self.candid_alpha != 0 else inf
        self.alpha      = min(self.alpha,self.candid_alpha)
        #shout(self,self.alpha)
        theta               += self.alpha * td_error * self.eligibility_trace
        #print candid_alpha
        
        #use this if you want to divide by the number of active features  [do this for PST]
#        nnz                 = count_nonzero(phi)    #Number of non-zero elements
#        theta               += self.alpha * td_error * phi / (1.*nnz)  
        
        #Discover features using online iFDD
        if isinstance(self.representation,iFDD):
            self.representation.discover(phi_s,td_error)
        
        # Set eligibility Traces to zero if it is end of the episode
        if self.lambda_: self.eligibility_trace = zeros(self.representation.features_num*self.domain.actions_num) 
