######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Agent import *
class OnlineAgent(Agent):
    initial_alpha       = 0             #initial learning rate
    alpha               = 0             #Learning rate to be used
    candid_alpha        = 0             #Candid Learning Rate Calculated by [Dabney W. 2012]
    def __init__(self,representation,policy,domain,initial_alpha):
        self.initial_alpha      = initial_alpha
        self.alpha              = initial_alpha 
        super(OnlineAgent,self).__init__(representation,policy,domain)
    def adjustAlpha(selfs):
        #Adjust Alpha based on a rule
        self.alpha = min(self.alpha,self.candid_alpha)
        #TODO: Add Boyan     
        