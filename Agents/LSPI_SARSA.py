#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#EXPERIMENTAL
######################################################
# Developed by Alborz Geramiard Jan 9th 2013 at MIT #
######################################################
# This agent uses SARSA for online learning and calls LSPI on sample_window
from LSPI import *
from TDControlAgent import SARSA
class LSPI_SARSA(Agent):
    def __init__(self,representation,policy,domain,logger, lspi_iterations = 5, sample_window = 100, epsilon = 1e-3, re_iterations = 100,initial_alpha =.1, lambda_=0,alpha_decay_mode ='dabney', boyan_N0 = 1000):
        self.SARSA = SARSA(representation, policy, domain,logger, initial_alpha, lambda_,alpha_decay_mode, boyan_N0)
        self.LSPI = LSPI(representation,policy,domain,logger, lspi_iterations, sample_window, epsilon, re_iterations)
        super(LSPI_SARSA,self).__init__(representation,policy,domain,logger)
    def learn(self,s,a,r,ns,na,terminal):
        self.LSPI.process(s,a,r,ns,na,terminal)
        if self.LSPI.samples_count+1 % self.LSPI.steps_between_LSPI == 0:
            self.LSPI.representationExpansionLSPI()
            if terminal:
                self.episodeTerminated()
        else:
            self.SARSA.learn(s,a,r,ns,na,terminal)
