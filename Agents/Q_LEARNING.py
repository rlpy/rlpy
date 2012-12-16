######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
# Extended to Q-learning by Girish Chowdhary Nov 20th 2012 at MIT #
######################################################
from Agent import *
class Q_LEARNING(Agent):
    lambda_ = 0        #lambda Parameter in [Sutton Book 1998]
    eligibility_trace = []  #
    def __init__(self, representation, policy, domain, logger = None, initial_alpha =.1, lambda_ = 0):
        self.eligibility_trace  = zeros(representation.features_num*domain.actions_num)
        self.lambda_            = lambda_
        self.alpha              = initial_alpha 
        super(Q_LEARNING,self).__init__(representation,policy,domain,logger)
        super(Q_LEARNING,self).printInfo()
        self.logger.log("Alpha_0:\t\t%0.2f" % initial_alpha)
        if lambda_: self.logger.log("lambda:\t%0.2f" % lambda_)
    def learn(self,s,a,r,ns,na,terminal):
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        phi_s               = self.representation.phi(s)
        phi                 = self.representation.phi_sa(s,a,phi_s)
        
        #Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace   *= gamma*self.lambda_
            self.eligibility_trace   += phi
            #Set max to 1
            self.eligibility_trace[self.eligibility_trace>1] = 1
        else:
            self.eligibility_trace    = phi
        
        ## find the best guess of Q^*: \hat{Q}=r+gamma*max_a(theta^T phi(s,a)), see e.g. Busoniu et al.'s book 2010
        na                  = self.representation.bestAction(ns)
        phi_prime           = self.representation.phi_sa(ns,na)
        # Girish I think this is your bug - passing state-action phi(s,a) using copy-paste method instead of phi(s)
        #Q_max               = self.representation.Q(ns,na,phi_prime)
        
        Q_max               = dot(phi_prime, theta)
        td_error            = r + gamma*Q_max-dot(phi, theta)

        #Automatic learning rate: [Dabney W. 2012]
        candid_alpha    = abs(dot(phi-gamma*phi_prime,self.eligibility_trace)) #http://people.cs.umass.edu/~wdabney/papers/alphaBounds.pdf
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
