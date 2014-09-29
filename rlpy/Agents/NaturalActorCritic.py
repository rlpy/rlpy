"""
Experimental Implementation of Natural Actor Critic
"""
import numpy as np
from .Agent import Agent
from rlpy.Tools import solveLinear, regularize

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class NaturalActorCritic(Agent):

    """
    the step-based Natural Actor Critic algorithm
    as described in algorithm 1 of
        Peters, J. & Schaal, S. Natural Actor-Critic.
        Neurocomputing 71, 1180-1190 (2008).

    """

    # minimum for the cosine of the current and last gradient
    min_cos = np.cos(np.pi / 180.)

    def __init__(self, policy, representation, discount_factor, forgetting_rate,
                 min_steps_between_updates, max_steps_between_updates, lambda_,
                 learn_rate):
        """
        @param representation: function approximation used to approximate the
                               value function
        @param policy:  parametrized stochastic policy that is an instance of
                                DifferentiablePolicy
        @param forgetting_rate: specifies the decay of previous statistics
                                after a policy update; 1 = forget all
                                0 = forget none
        @param min_steps_between_updates: minimum number of steps between
                                two policy updates
        @param max_steps_between_updates
        @param lambda_:    e-trace parameter lambda
        @param learn_rate:  learning rate

        """

        self.samples_count = 0
        self.forgetting_rate = forgetting_rate
        self.n = representation.features_num + len(policy.theta)
        self.representation = representation
        self.min_steps_between_updates = min_steps_between_updates
        self.max_steps_between_updates = max_steps_between_updates
        self.lambda_ = lambda_
        self.learn_rate = learn_rate

        self.steps_between_updates = 0
        self.b = np.zeros((self.n))
        self.A = np.zeros((self.n, self.n))
        self.buf_ = np.zeros((self.n, self.n))
        self.z = np.zeros((self.n))

        super(NaturalActorCritic, self).__init__(policy,
                                                 representation, discount_factor)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        # compute basis functions
        phi_s = np.zeros((self.n))
        phi_ns = np.zeros((self.n))
        k = self.representation.features_num
        phi_s[:k] = self.representation.phi(s, False)
        phi_s[k:] = self.policy.dlogpi(s, a)
        phi_ns[:k] = self.representation.phi(ns, terminal)

        # update statistics
        self.z *= self.lambda_
        self.z += phi_s

        self.A += np.einsum("i,j", self.z, phi_s - self.discount_factor * phi_ns,
                            out=self.buf_)
        self.b += self.z * r
        if terminal:
            self.z[:] = 0.
        self.steps_between_updates += 1
        self.logger.debug("Statistics updated")

        if self.steps_between_updates > self.min_steps_between_updates:
            A = regularize(self.A)
            param, time = solveLinear(A, self.b)
            #  v = param[:k]  # parameters of the value function representation
            w = param[k:]  # natural gradient estimate

            if self._gradient_sane(w) or self.steps_between_updates > self.max_steps_between_updates:
                # update policy
                self.policy.theta = self.policy.theta + self.learn_rate * w
                self.last_w = w
                self.logger.debug(
                    "Policy updated, norm of gradient {}".format(np.linalg.norm(w)))
                # forget statistics
                self.z *= 1. - self.forgetting_rate
                self.A *= 1. - self.forgetting_rate
                self.b *= 1. - self.forgetting_rate
                self.steps_between_updates = 0

        if terminal:
            self.episodeTerminated()

    def _gradient_sane(self, w):
        """
        checks the natural gradient estimate w for sanity
        """
        if hasattr(self, "last_w"):
            cos = np.dot(w, self.last_w) / np.linalg.norm(w) \
                / np.linalg.norm(self.last_w)
            return cos < self.min_cos
        else:
            return False
