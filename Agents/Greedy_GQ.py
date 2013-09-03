#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
# Following Maei et al. 2009 Greedy GQ Algorithm
# Use it with lambda = 0 as Eligibility Trace mode has not been tested!
######################################################
from Agent import *
class Greedy_GQ(Agent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace   = []
    eligibility_trace_s = [] # eligibility trace using state only (no copy-paste), necessary for dabney decay mode
    def __init__(self, representation, policy, domain,logger, initial_alpha =.1, lambda_ = 0, alpha_decay_mode = 'dabney', boyan_N0 = 1000, BetaCoef = 1e-3):
        self.eligibility_trace  = zeros(representation.features_num*domain.actions_num)
        self.eligibility_trace_s= zeros(representation.features_num) # use a state-only version of eligibility trace for dabney decay mode
        self.lambda_            = lambda_
        super(Greedy_GQ,self).__init__(representation,policy,domain,logger,initial_alpha,alpha_decay_mode, boyan_N0)
        self.GQWeight = copy(self.representation.theta)
        self.secondLearningRateCoef = BetaCoef  # The beta in the GQ algorithm is assumed to be alpha * THIS CONSTANT
        self.logger.log("Alpha_0:\t\t%0.2f" % initial_alpha)
        self.logger.log("Decay mode:\t\t"+str(alpha_decay_mode))
        self.logger.log("Beta:\t\t"+str(BetaCoef))
        if lambda_: self.logger.log("lambda:\t%0.2f" % lambda_)

    def learn(self,s,p_actions, a,r,ns, np_actions, na,terminal):
        gamma           = self.representation.domain.gamma
        theta           = self.representation.theta
        phi_s           = self.representation.phi(s, False)
        phi             = self.representation.phi_sa(s, False, a, phi_s)
        phi_prime_s     = self.representation.phi(ns, terminal)
        na              = self.representation.bestAction(ns,terminal, np_actions, phi_prime_s) #Switch na to the best possible action
        phi_prime       = self.representation.phi_sa(ns, terminal, na, phi_prime_s)
        nnz             = count_nonzero(phi_s)    # Number of non-zero elements


        #Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace   *= gamma*self.lambda_
            self.eligibility_trace   += phi

            self.eligibility_trace_s  *= gamma*self.lambda_
            self.eligibility_trace_s += phi_s

            #Set max to 1
            self.eligibility_trace[self.eligibility_trace>1] = 1
            self.eligibility_trace_s[self.eligibility_trace_s>1] = 1
        else:
            self.eligibility_trace    = phi
            self.eligibility_trace_s  = phi_s

        td_error                     = r + dot(gamma*phi_prime - phi, theta)
        self.updateAlpha(phi_s,phi_prime_s,self.eligibility_trace_s, gamma, nnz, terminal)

        if nnz > 0: # Phi has some nonzero elements, proceed with update
            td_error_estimate_now       = dot(phi,self.GQWeight)
            Delta_theta                 = td_error*self.eligibility_trace - gamma*td_error_estimate_now*phi_prime
            theta                       += self.alpha*Delta_theta
            Delta_GQWeight              = (td_error-td_error_estimate_now)*phi
            self.GQWeight               += self.alpha*self.secondLearningRateCoef*Delta_GQWeight

        #Discover features if the representation has the discover method
        discover_func = getattr(self.representation,'discover',None) # None is the default value if the discover is not an attribute
        if discover_func and callable(discover_func):
            expanded = self.representation.discover(s, False, a, td_error, phi_s)

            #Assuming one expansion for one interaction.
            if expanded:
                new_elem = zeros((self.domain.actions_num, expanded))
                # Correct the size of self.GQWeight
                self.GQWeight = addNewElementForAllActions(self.GQWeight,self.domain.actions_num, new_elem)
                if self.lambda_:
                    # Correct the size of eligibility traces (pad with zeros for new features)
                    self.eligibility_trace  = addNewElementForAllActions(self.eligibility_trace,self.domain.actions_num, new_elem)
                    self.eligibility_trace_s = addNewElementForAllActions(self.eligibility_trace_s,1, zeros((1, expanded)))

        if terminal:
            self.episodeTerminated()
