"""Control Agents based on TD Learning, i.e., Q-Learning and SARSA"""
from .Agent import Agent, DescentAlgorithm
from rlpy.Tools import addNewElementForAllActions, count_nonzero
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class TDControlAgent(DescentAlgorithm, Agent):

    """
    abstract class for the control variants of the classical linear TD-Learning.
    It is the parent of SARSA and Q-Learning

    All children must implement the _future_action function.
    """

    lambda_ = 0        #: lambda Parameter in SARSA [Sutton Book 1998]
    eligibility_trace = []  #: eligibility trace
    #: eligibility trace using state only (no copy-paste), necessary for dabney decay mode
    eligibility_trace_s = []

    def __init__(self, representation, policy, domain, lambda_=0, **kwargs):
        self.eligibility_trace = np.zeros(
            representation.features_num *
            domain.actions_num)
        # use a state-only version of eligibility trace for dabney decay mode
        self.eligibility_trace_s = np.zeros(representation.features_num)
        self.lambda_ = lambda_
        super(TDControlAgent, self).__init__(representation=representation, policy=policy,
            domain=domain, **kwargs)

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """needs to be implemented by children"""
        pass

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):

        self.representation.pre_discover(s, False, a, ns, terminal)
        gamma = self.representation.domain.gamma
        theta = self.representation.theta
        phi_s = self.representation.phi(s, False)
        phi = self.representation.phi_sa(s, False, a, phi_s)
        phi_prime_s = self.representation.phi(ns, terminal)
        na = self._future_action(
            ns,
            terminal,
            np_actions,
            phi_prime_s,
            na)  # here comes the difference between SARSA and Q-Learning
        phi_prime = self.representation.phi_sa(
            ns,
            terminal,
            na,
            phi_prime_s)
        nnz = count_nonzero(phi_s)    # Number of non-zero elements

        # Set eligibility traces:
        if self.lambda_:
            # make sure that
            expanded = (- len(self.eligibility_trace) + len(phi)) / \
                self.domain.actions_num
            if expanded > 0:
                # Correct the size of eligibility traces (pad with zeros for
                # new features)
                self.eligibility_trace = addNewElementForAllActions(
                    self.eligibility_trace,
                    self.domain.actions_num,
                    np.zeros((self.domain.actions_num,
                              expanded)))
                self.eligibility_trace_s = addNewElementForAllActions(
                    self.eligibility_trace_s, 1, np.zeros((1, expanded)))

            self.eligibility_trace *= gamma * self.lambda_
            self.eligibility_trace += phi

            self.eligibility_trace_s *= gamma * self.lambda_
            self.eligibility_trace_s += phi_s

            # Set max to 1
            self.eligibility_trace[self.eligibility_trace > 1] = 1
            self.eligibility_trace_s[self.eligibility_trace_s > 1] = 1
        else:
            self.eligibility_trace = phi
            self.eligibility_trace_s = phi_s

        td_error = r + np.dot(gamma * phi_prime - phi, theta)
        if nnz > 0:
            self.updateAlpha(
                phi_s,
                phi_prime_s,
                self.eligibility_trace_s,
                gamma,
                nnz,
                terminal)
            theta_old = theta.copy()
            theta               += self.alpha * \
                td_error * self.eligibility_trace
            if not np.all(np.isfinite(theta)):
                theta = theta_old
                print "WARNING: TD-Learning diverged, theta reached infinity!"
        # Discover features if the representation has the discover method
        expanded = self.representation.post_discover(
            s,
            False,
            a,
            td_error,
            phi_s)

        if terminal:
            self.episodeTerminated()


class Q_Learning(TDControlAgent):

    """
    The off-policy variant known as Q-Learning
    """

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """Q Learning choses the optimal action"""
        return self.representation.bestAction(ns, terminal, np_actions, ns_phi)


class SARSA(TDControlAgent):

    """
    The on-policy variant known as SARSA.
    """

    def _future_action(self, ns, terminal, np_actions, ns_phi, na):
        """SARS-->A<--, so SARSA simply choses the action the agent will follow"""
        return na
