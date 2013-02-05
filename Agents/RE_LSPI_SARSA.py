#EXPERIMENTAL
######################################################
# Developed by Alborz Geramiard Jan 9th 2013 at MIT #
######################################################
# This agent uses SARSA for online learning and calls RE_LSPI on sample_window
from RE_LSPI import *
from SARSA import *
class RE_LSPI_SARSA(Agent):
    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3, re_iterations = 100,initial_alpha =.1, lambda_=0,alpha_decay_mode ='dabney', boyan_N0 = 1000):
        self.SARSA = SARSA(representation, policy, domain,logger, initial_alpha, lambda_,alpha_decay_mode, boyan_N0)
        self.RE_LSPI = RE_LSPI(representation,policy,domain,logger, lspi_iterations, sample_window, epsilon, re_iterations)
        super(RE_LSPI_SARSA,self).__init__(representation,policy,domain,logger)
    def learn(self,s,a,r,ns,na,terminal):
        self.RE_LSPI.storeData(s,a,r,ns,na)        
        if self.RE_LSPI.samples_count == self.RE_LSPI.sample_window:
            self.RE_LSPI.representationExpansionLSPI()
        else:
            self.SARSA.learn(s,a,r,ns,na,terminal)