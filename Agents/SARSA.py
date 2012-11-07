######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from OnlineAgent import *
class SARSA(OnlineAgent):
    lambda_param = 0    #lambda Parameter in SARSA [Sutton Book 1998]
    def __init__(self,representation,policy,domain,initial_alpha =.1):
        super(SARSA,self).__init__(representation,policy,domain,initial_alpha)
    def learn(self,s,a,r,ns,na):
        phi                 = self.representation.phi_sa(s,a)
        phi_prime           = self.representation.phi_sa(ns,na)
        gamma               = self.representation.domain.gamma
        theta               = self.representation.theta
        alpha               = self.alpha
        td_error            = r + dot(gamma*phi_prime - phi, theta)
        self.candidAlpha    = 1/dot(abs(phi-gamma*phi_prime),phi) #[Dabney W. 2012]
        theta               += alpha * td_error * phi
        #use this if you want to divide by the number of active features 
        #nnz                 = count_nonzero(phi)    #Number of non-zero elements
        #theta               += alpha * td_error * phi / nnz  
          
