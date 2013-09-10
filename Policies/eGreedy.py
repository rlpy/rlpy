#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
from Policy import *
class eGreedy(Policy):
    epsilon         = None
    old_epsilon     = None
    forcedDeterministicAmongBestActions = None # This boolean variable is used to avoid random selection among actions with the same values
    def __init__(self,representation,logger,epsilon = .1,
                 forcedDeterministicAmongBestActions = False):
        self.epsilon = epsilon
        self.forcedDeterministicAmongBestActions = forcedDeterministicAmongBestActions
        super(eGreedy,self).__init__(representation,logger)
        if self.logger:
            self.logger.log("=" * 60)
            self.logger.log("Policy: eGreedy")
            self.logger.log("Epsilon\t\t{0}".format(self.epsilon))

    def pi(self,s, terminal, p_actions):
        coin = random.rand()
        #print "coin=",coin
        if coin < self.epsilon:
            return randSet(p_actions)
        else:
            b_actions = self.representation.bestActions(s, terminal, p_actions)
            if self.forcedDeterministicAmongBestActions:
                return b_actions[0]
            else:
                return randSet(b_actions)
    def turnOffExploration(self):
        self.old_epsilon = self.epsilon
        self.epsilon = 0
    def turnOnExploration(self):
        self.epsilon = self.old_epsilon




