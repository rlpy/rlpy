"""Greedy-GQ(lambda) learning agent"""
from .Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
import numpy as np
from copy import copy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Greedy_GQ(DescentAlgorithm, Agent):
    """
    A variant of Q-learning for use with linear function approximation.
    Convergence guarantees are available even without an exact value 
    function representation.
    See Maei et al., 2010 (http://www.icml2010.org/papers/627.pdf)

    """

    lambda_ = 0  # lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace = []
    # eligibility trace using state only (no copy-paste), necessary for dabney
    # decay mode
    eligibility_trace_s = []

    def __init__(self, policy, representation,
            lambda_=0, BetaCoef=1e-6, **kwargs):
        self.eligibility_trace = np.zeros(
            representation.features_num *
            representation.actions_num)
        # use a state-only version of eligibility trace for dabney decay mode
        self.eligibility_trace_s = np.zeros(representation.features_num)
        self.lambda_ = lambda_
        super(
            Greedy_GQ,
            self).__init__(
            policy=policy,
            representation=representation,
            **kwargs)
        self.GQWeight = copy(self.representation.weight_vec)
        # The beta in the GQ algorithm is assumed to be learn_rate * THIS CONSTANT
        self.secondLearningRateCoef = BetaCoef

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
        self.representation.pre_discover(s, False, a, ns, terminal)
        discount_factor = self.discount_factor
        weight_vec = self.representation.weight_vec
        phi_s = self.representation.phi(s, False)
        phi = self.representation.phi_sa(s, False, a, phi_s)
        phi_prime_s = self.representation.phi(ns, terminal)
        na = self.representation.bestAction(
            ns,
            terminal,
            np_actions,
            phi_prime_s)  # Switch na to the best possible action
        phi_prime = self.representation.phi_sa(
            ns,
            terminal,
            na,
            phi_prime_s)
        nnz = count_nonzero(phi_s)    # Number of non-zero elements

        expanded = (- len(self.GQWeight) + len(phi)) / self.representation.actions_num
        if expanded:
            self._expand_vectors(expanded)
        # Set eligibility traces:
        if self.lambda_:
            self.eligibility_trace *= discount_factor * self.lambda_
            self.eligibility_trace += phi

            self.eligibility_trace_s *= discount_factor * self.lambda_
            self.eligibility_trace_s += phi_s

            # Set max to 1
            self.eligibility_trace[self.eligibility_trace > 1] = 1
            self.eligibility_trace_s[self.eligibility_trace_s > 1] = 1
        else:
            self.eligibility_trace = phi
            self.eligibility_trace_s = phi_s

        td_error                     = r + \
            np.dot(discount_factor * phi_prime - phi, weight_vec)
        self.updateLearnRate(
            phi_s,
            phi_prime_s,
            self.eligibility_trace_s,
            discount_factor,
            nnz,
            terminal)

        if nnz > 0:  # Phi has some nonzero elements, proceed with update
            td_error_estimate_now = np.dot(phi, self.GQWeight)
            Delta_weight_vec                 = td_error * self.eligibility_trace - \
                discount_factor * td_error_estimate_now * phi_prime
            weight_vec += self.learn_rate * Delta_weight_vec
            Delta_GQWeight = (
                td_error - td_error_estimate_now) * phi
            self.GQWeight               += self.learn_rate * \
                self.secondLearningRateCoef * Delta_GQWeight

        expanded = self.representation.post_discover(
            s,
            False,
            a,
            td_error,
            phi_s)
        if expanded:
            self._expand_vectors(expanded)
        if terminal:
            self.episodeTerminated()

    def _expand_vectors(self, num_expansions):
        """
        correct size of GQ weight and e-traces when new features were expanded
        """
        new_elem = np.zeros((self.representation.actions_num, num_expansions))
        self.GQWeight = addNewElementForAllActions(
            self.GQWeight,
            self.representation.actions_num,
            new_elem)
        if self.lambda_:
            # Correct the size of eligibility traces (pad with zeros for new
            # features)
            self.eligibility_trace = addNewElementForAllActions(
                self.eligibility_trace,
                self.representation.actions_num,
                new_elem)
            self.eligibility_trace_s = addNewElementForAllActions(
                self.eligibility_trace_s, 1, np.zeros((1, num_expansions)))
