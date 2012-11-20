######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from OnlineAgent import *
class SARSA(OnlineAgent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace = []  #
    def __init__(self,representation,policy,domain,initial_alpha =.1, lambda_ = 0):
        self.eligibility_trace  = zeros(representation.features_num*domain.actions_num)
        self.lambda_            = lambda_
        self.alpha              = initial_alpha 
        super(SARSA,self).__init__(representation,policy,domain)
    def learn(self,s,a,r,ns,na,terminal):
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        phi                 = self.representation.phi_sa(s,a)
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
        candid_alpha    = abs(dot(phi-gamma*phi_prime,self.eligibility_trace)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        candid_alpha    = 1/(self.candid_alpha*1.) if self.candid_alpha != 0 else inf 
        self.alpha      = min(self.alpha,candid_alpha)
        
        x = abs(dot(phi-gamma*phi_prime,self.eligibility_trace))
        #shout(self,self.alpha)
        theta               += self.alpha * td_error * self.eligibility_trace
        
        #use this if you want to divide by the number of active features 
        #nnz                 = count_nonzero(phi)    #Number of non-zero elements
        #theta               += alpha * td_error * phi / (1.*nnz)  
        super(SARSA,self).generalUpdates(s,td_error,terminal)
