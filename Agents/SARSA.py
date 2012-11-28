######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from OnlineAgent import *
class SARSA(OnlineAgent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace = []  #
    def __init__(self, representation, policy, domain, initial_alpha =.1, lambda_ = 0):
        self.eligibility_trace  = sp_matrix(representation.features_num*domain.actions_num)
        self.lambda_            = lambda_
        self.alpha              = initial_alpha 
        super(SARSA,self).__init__(representation,policy,domain)
        super(SARSA,self).printInfo()
        print "Alpha_0:\t\t", initial_alpha
        if lambda_: print "lambda:\t", lambda_
    def learn(self,s,a,r,ns,na,terminal):
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        phi_s               = self.representation.phi(s)
        phi                 = self.representation.phi_sa_from_phi_s(phi_s,a)
        phi_prime           = self.representation.phi_sa(ns,na)
        
        #Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace   = self.eligibility_trace * gamma*self.lambda_
            self.eligibility_trace   = self.eligibility_trace + phi
            #Set max to 1
            self.eligibility_trace[self.eligibility_trace>1] = 1
        else:
            self.eligibility_trace    = phi.astype(float)
        print type(phi)
        delta_phi           = (gamma*phi_prime.astype(float) - phi).todok()
        td_error            = r + sp_dot_array(delta_phi, theta)

        #Automatic learning rate: [Dabney W. 2012]
        candid_alpha    = abs(sp_dot_sp(-delta_phi,self.eligibility_trace)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
        candid_alpha    = 1/(self.candid_alpha*1.) if self.candid_alpha != 0 else inf 
        self.alpha      = min(self.alpha,candid_alpha)
        #shout(self,self.alpha)
        theta               += self.alpha * td_error * self.eligibility_trace
        
        #use this if you want to divide by the number of active features 
        #nnz                 = count_nonzero(phi)    #Number of non-zero elements
        #theta               += alpha * td_error * phi / (1.*nnz)  
        
        #Discover features using online iFDD
        if isinstance(self.representation,iFDD):
            self.representation.discover(phi_s,td_error)
        
        # Set eligibility Traces to zero if it is end of the episode
        if self.lambda_: self.eligibility_trace = zeros(self.representation.features_num*self.domain.actions_num) 
