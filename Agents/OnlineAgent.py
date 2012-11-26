######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Agent import *
class OnlineAgent(Agent):
    initial_alpha       = 0             #initial learning rate
    alpha               = 0             #Learning rate to be used
    candid_alpha        = 0             #Candid Learning Rate Calculated by [Dabney W. 2012]
    eligibility_trace   = []            #In case lambda parameter in SARSA definition is used
    def __init__(self,representation,policy,domain):
        super(OnlineAgent,self).__init__(representation,policy,domain)
    def generalUpdates(self,phi_s,td_error,terminal):
        # Call all necessary functions such as iFDD update
        # This function is often called at the end of <child>.learn()
        
        #Discover features using online iFDD
        if isinstance(self.representation,iFDD):
            self.representation.discover(phi_s,td_error)

        #Discover features using batch iFDD
        if isinstance(self.representation,BatchiFDD):
            self.representation.discover(s,td_error)

        #Set eligibility traces to zero if episode finishes
        if terminal:
           self.eligibility_trace = zeros(self.representation.features_num*self.domain.actions_num) 
