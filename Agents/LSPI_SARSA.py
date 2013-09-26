"""Least-Squares Policy Iteration but with SARSA"""
from Agent import Agent
from Tools import *
from LSPI import LSPI
from TDControlAgent import SARSA

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

#EXPERIMENTAL

class LSPI_SARSA(SARSA):
    """This agent uses SARSA for online learning and calls LSPI on sample_window"""

    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3, re_iterations = 100,initial_alpha =.1, lambda_=0,alpha_decay_mode ='dabney', boyan_N0 = 1000):
        super(LSPI_SARSA,self).__init__(representation, policy, domain,logger, initial_alpha, lambda_,alpha_decay_mode, boyan_N0)
        self.LSPI = LSPI(representation,policy,domain,logger, lspi_iterations, sample_window, epsilon, re_iterations)
        
    def learn(self,s,p_actions,a,r,ns,np_actions,na,terminal):
        """Iterative learning method for the agent. 

        Args:
            s (ndarray):    The current state features
            p_actions (ndarray):    The actions available in state s
            a (int):    The action taken by the agent in state s
            r (float):  The reward received by the agent for taking action a in state s
            ns (ndarray):   The next state features
            np_actions (ndarray): The actions available in state ns
            na (int):   The action taken by the agent in state ns
            terminal (bool): Whether or not ns is a terminal state
        """
        self.LSPI.process(s,a,r,ns,na,terminal)
        if self.LSPI.samples_count+1 % self.LSPI.steps_between_LSPI == 0:
            self.LSPI.representationExpansionLSPI()
            if terminal:
                self.episodeTerminated()
        else:
            super(LSPI_SARSA, self).learn(s,p_actions,a,r,ns,np_actions,na,terminal)
