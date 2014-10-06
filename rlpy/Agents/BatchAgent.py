"""An abstract class for Batch Learning Agents"""
from .Agent import Agent
import rlpy.Tools as Tools
import numpy as np
from scipy import sparse as sp

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class BatchAgent(Agent):

    """An abstract class for batch agents
    """

    max_window = 0
    samples_count = 0  # Number of samples gathered so far

    def __init__(self, policy, representation, discount_factor, max_window):

        super(BatchAgent, self).__init__(policy, representation, discount_factor=discount_factor)

        self.max_window = max_window
        self.samples_count = 0

        # Take memory for stored values
        self.data_s = np.zeros((max_window, self. representation.state_space_dims))
        self.data_ns = np.zeros((max_window, self.representation.state_space_dims))
        self.data_a = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_na = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_r = np.zeros((max_window, 1))

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
        self.process(s, a, r, ns, na, terminal)
        if (self.samples_count) % self.steps_between_LSPI == 0:
            self.representationExpansionLSPI()
        if terminal:
            self.episodeTerminated()

    def process(self, s, a, r, ns, na, terminal):
        """Process one transition instance."""
        # Save samples
        self.data_s[self.samples_count, :] = s
        self.data_a[self.samples_count] = a
        self.data_r[self.samples_count] = r
        self.data_ns[self.samples_count, :] = ns
        self.data_na[self.samples_count] = na

        self.samples_count += 1

    def calculateTDErrors(self):
        """Calculate TD errors over the transition instances stored.
        Returns the TD errors for all transitions with the current parameters.
        """
        # Calculates the TD-Errors in a matrix format for a set of samples = R
        # + (discount_factor*F2 - F1) * Theta
        discount_factor = self.discount_factor
        R = self.data_r[:self.samples_count, :]
        if self.use_sparse:
            F1 = sp.csr_matrix(self.all_phi_s_a[:self.samples_count, :])
            F2 = sp.csr_matrix(self.all_phi_ns_na[:self.samples_count, :])
            answer = (
                R + (discount_factor * F2 - F1) * self.representation.weight_vec.reshape(-1, 1))
            return np.squeeze(np.asarray(answer))
        else:
            F1 = self.all_phi_s_a[:self.samples_count, :]
            F2 = self.all_phi_ns_na[:self.samples_count, :]
            return R.ravel() + np.dot(discount_factor * F2 - F1, self.representation.weight_vec)

    def representationExpansionLSPI(self):
        re_iteration = 0
        added_feature = True

        if self.representation.features_num == 0:
            print "No features, hence no LSPI is necessary!"
            return

        self.logger.info(
            "============================\nRunning LSPI with %d Samples\n============================" %
            self.samples_count)
        while added_feature and re_iteration <= self.re_iterations:
            re_iteration += 1
            # Some Prints
            if Tools.hasFunction(self.representation, 'batchDiscover'):
                self.logger.info(
                    '-----------------\nRepresentation Expansion iteration #%d\n-----------------' %
                    re_iteration)
            # Run LSTD for first solution
            self.LSTD()
            # Run Policy Iteration to change a_prime and recalculate weight_vec in a
            # loop
            td_errors = self.policyIteration()
            # Add new Features
            if Tools.hasFunction(self.representation, 'batchDiscover'):
                added_feature = self.representation.batchDiscover(td_errors, self.all_phi_s[:self.samples_count, :], self.data_s[:self.samples_count, :])
            else:
                added_feature = False
            # print 'L_inf distance to V*= ',
        if added_feature:
            # Run LSPI one last time with the new features
            self.LSTD()
            self.policyIteration()

    def episodeTerminated(self):
        """This function adjusts all necessary elements of the agent at the end of
        the episodes.
        .. note::
            Every agent must call this function at the end of the learning if the
            transition led to terminal state.
        """
        # Increase the number of episodes
        self.episode_count += 1
