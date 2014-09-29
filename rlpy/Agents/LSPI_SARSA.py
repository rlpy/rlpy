"""Least-Squares Policy Iteration but with SARSA"""
from .LSPI import LSPI
from .TDControlAgent import SARSA

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

# EXPERIMENTAL


class LSPI_SARSA(SARSA):

    """This agent uses SARSA for online learning and calls LSPI on sample_window"""

    def __init__(
            self, policy, representation, discount_factor, lspi_iterations=5,
            steps_between_LSPI=100,sample_window=100,
            tol_epsilon=1e-3, re_iterations=100, initial_learn_rate=.1,
            lambda_=0, learn_rate_decay_mode='dabney', boyan_N0=1000):
        super(
            LSPI_SARSA,
            self).__init__(
            policy,
            representation,
            discount_factor=discount_factor,
            lambda_=lambda_,
            initial_learn_rate=initial_learn_rate,
            learn_rate_decay_mode=learn_rate_decay_mode,
            boyan_N0=boyan_N0)
        self.LSPI = LSPI(
            policy,
            representation,
            discount_factor,
            sample_window,
            steps_between_LSPI,
            lspi_iterations,
            tol_epsilon,
            re_iterations)

    def learn(self, s, p_actions, a, r, ns, np_actions, na, terminal):
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
        self.LSPI.process(s, a, r, ns, na, terminal)
        if self.LSPI.samples_count + 1 % self.LSPI.steps_between_LSPI == 0:
            self.LSPI.representationExpansionLSPI()
            if terminal:
                self.episodeTerminated()
        else:
            super(
                LSPI_SARSA,
                self).learn(s,
                            p_actions,
                            a,
                            r,
                            ns,
                            np_actions,
                            na,
                            terminal)
