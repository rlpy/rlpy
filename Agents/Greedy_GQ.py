"""Greedy-GQ(lambda) learning agent"""
from Agent import Agent, DescentAlgorithm
from Tools import addNewElementForAllActions, count_nonzero
import numpy as np
from copy import copy

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Greedy_GQ(DescentAlgorithm, Agent):
    lambda_ = 0        #lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace   = []
    eligibility_trace_s = [] # eligibility trace using state only (no copy-paste), necessary for dabney decay mode
    def __init__(self, representation, policy, domain,logger, lambda_ = 0,
                 BetaCoef = 1e-6, **kwargs):
        self.eligibility_trace  = np.zeros(representation.features_num*domain.actions_num)
        self.eligibility_trace_s= np.zeros(representation.features_num) # use a state-only version of eligibility trace for dabney decay mode
        self.lambda_            = lambda_
        super(Greedy_GQ,self).__init__(representation=representation,
                                       policy=policy,
                                       domain=domain,
                                       logger=logger, **kwargs)
        self.GQWeight = copy(self.representation.theta)
        self.secondLearningRateCoef = BetaCoef  # The beta in the GQ algorithm is assumed to be alpha * THIS CONSTANT

    def learn(self,s,p_actions, a,r,ns, np_actions, na,terminal):
        self.representation.pre_discover(s, False, a, ns, terminal)
        gamma           = self.representation.domain.gamma
        theta           = self.representation.theta
        phi_s           = self.representation.phi(s, False)
        phi             = self.representation.phi_sa(s, False, a, phi_s)
        phi_prime_s     = self.representation.phi(ns, terminal)
        na              = self.representation.bestAction(ns,terminal, np_actions, phi_prime_s) #Switch na to the best possible action
        phi_prime       = self.representation.phi_sa(ns, terminal, na, phi_prime_s)
        nnz             = count_nonzero(phi_s)    # Number of non-zero elements

        expanded = (- len(self.GQWeight) + len(phi)) / self.domain.actions_num
        if expanded:
            self._expand_vectors(expanded)
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

        td_error                     = r + np.dot(gamma*phi_prime - phi, theta)
        self.updateAlpha(phi_s,phi_prime_s,self.eligibility_trace_s, gamma, nnz, terminal)

        if nnz > 0: # Phi has some nonzero elements, proceed with update
            td_error_estimate_now       = np.dot(phi,self.GQWeight)
            Delta_theta                 = td_error*self.eligibility_trace - gamma*td_error_estimate_now*phi_prime
            theta                       += self.alpha*Delta_theta
            Delta_GQWeight              = (td_error-td_error_estimate_now)*phi
            self.GQWeight               += self.alpha*self.secondLearningRateCoef*Delta_GQWeight


        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)
        if expanded:
            self._expand_vectors(expanded)
        if terminal:
            self.episodeTerminated()

    def episodeTerminated(self):
        if self.lambda_:
            self.eligibility_trace  = np.zeros(self.representation.features_num*self.domain.actions_num)
            self.eligibility_trace_s = np.zeros(self.representation.features_num)
        super(Greedy_GQ, self).episodeTerminated()

    def _expand_vectors(self, num_expansions):
        """
        correct size of GQ weight and e-traces when new features were expanded
        """
        new_elem = np.zeros((self.domain.actions_num, num_expansions))
        self.GQWeight = addNewElementForAllActions(self.GQWeight,self.domain.actions_num, new_elem)
        if self.lambda_:
            # Correct the size of eligibility traces (pad with zeros for new features)
            self.eligibility_trace  = addNewElementForAllActions(self.eligibility_trace,self.domain.actions_num, new_elem)
            self.eligibility_trace_s = addNewElementForAllActions(self.eligibility_trace_s,1, np.zeros((1, num_expansions)))
