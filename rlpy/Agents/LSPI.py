"""Least-Squares Policy Iteration [Lagoudakis and Parr 2003]."""
from .Agent import Agent
import rlpy.Tools as Tools
import numpy as np
from scipy import sparse as sp

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class LSPI(Agent):

    """Least Squares Policy Iteration reinforcement learning agent.

    Args:
        representation (Representation):    Representation over the state features used by the agent.

        policy (Policy):    Policy used by the agent.

        domain (Domain):    Domain the agent will be acting within. This is used to get the state
                            information and discount rate.

        max_window (int):   Maximum number of steps the agent will be run for,
                            which acts as the number of transitions to store.

        steps_between_LSPI (int):   Number of steps between runs of the LSPI algorithm.

        lspi_iterations (int):  Maximum number of iterations to go through for each update with LSPI.

        epsilon (float):    Stopping criteria, threshold, for when LSPI is considered to have converged.

        re_iterations (int):    Number of iterations of representation expansion to run.

        use_sparse (bool):  Use sparse operators for building the matrix of transitions?

    """

    use_sparse = 0         # Use sparse operators for building A?
    lspi_iterations = 0         # Number of LSPI iterations
    # Number of samples to be used to calculate the A and b matrices
    max_window = 0
    steps_between_LSPI = 0         # Number of samples between each LSPI run.
    samples_count = 0         # Number of samples gathered so far
    # Minimum l_2 change required to continue iterations in LSPI
    epsilon = 0

    # Store Data in separate matrixes
    data_s = []        #
    data_a = []        #
    data_r = []        #
    data_ns = []        #
    data_na = []        #

    # Reprsentation Expansion
    # Maximum number of iterations over LSPI and Representation expansion
    re_iterations = 0

    def __init__(
            self, representation, policy, domain, max_window, steps_between_LSPI,
            lspi_iterations=5, epsilon=1e-3, re_iterations=100, use_sparse=False):
        self.samples_count = 0
        self.max_window = max_window
        self.steps_between_LSPI = steps_between_LSPI
        self.epsilon = epsilon
        self.lspi_iterations = lspi_iterations
        self.re_iterations = re_iterations
        self.use_sparse = use_sparse

        # Take memory for stored values
        self.data_s = np.zeros((max_window, domain.state_space_dims))
        self.data_ns = np.zeros((max_window, domain.state_space_dims))
        self.data_a = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_na = np.zeros((max_window, 1), dtype=np.uint32)
        self.data_r = np.zeros((max_window, 1))

        # Make A and r incrementally if the representation can not expand
        self.fixedRep = not representation.isDynamic
        if self.fixedRep:
            f_size = representation.features_num * domain.actions_num
            self.b = np.zeros((f_size, 1))
            self.A = np.zeros((f_size, f_size))

            # Cache calculated phi vectors
            if self.use_sparse:
                self.all_phi_s = sp.lil_matrix(
                    (max_window, representation.features_num))
                self.all_phi_ns = sp.lil_matrix(
                    (max_window, representation.features_num))
                self.all_phi_s_a = sp.lil_matrix((max_window, f_size))
                self.all_phi_ns_na = sp.lil_matrix((max_window, f_size))
            else:
                self.all_phi_s = np.zeros(
                    (max_window, representation.features_num))
                self.all_phi_ns = np.zeros(
                    (max_window, representation.features_num))
                self.all_phi_s_a = np.zeros((max_window, f_size))
                self.all_phi_ns_na = np.zeros((max_window, f_size))

        super(LSPI, self).__init__(representation, policy, domain)

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

    def policyIteration(self):
        """Update the policy by recalculating A based on new na.

        Returns the TD error for each sample based on the latest weights and next actions.
        """
        start_time = Tools.clock()
        weight_diff = self.epsilon + 1  # So that the loop starts
        lspi_iteration = 0
        self.best_performance = -np.inf
        self.logger.info('Running Policy Iteration:')

        # We save action_mask on the first iteration (used for batchBestAction) to reuse it and boost the speed
        # action_mask is a matrix that shows which actions are available for
        # each state
        action_mask = None
        gamma = self.domain.gamma
        F1 = sp.csr_matrix(self.all_phi_s_a[:self.samples_count, :]) if self.use_sparse else self.all_phi_s_a[:self.samples_count,:]
        while lspi_iteration < self.lspi_iterations and weight_diff > self.epsilon:

            # Find the best action for each state given the current value function
            # Notice if actions have the same value the first action is
            # selected in the batch mode
            iteration_start_time = Tools.clock()
            bestAction, self.all_phi_ns_new_na, action_mask = self.representation.batchBestAction(self.data_ns[:self.samples_count, :], self.all_phi_ns, action_mask, self.use_sparse)

            # Recalculate A matrix (b remains the same)
            # Solve for the new theta
            if self.use_sparse:
                F2  = sp.csr_matrix(self.all_phi_ns_new_na[:self.samples_count, :])
                A = F1.T * (F1 - gamma * F2)
            else:
                F2  = self.all_phi_ns_new_na[:self.samples_count, :]
                A = np.dot(F1.T, F1 - gamma * F2)

            A = Tools.regularize(A)
            new_theta, solve_time = Tools.solveLinear(A, self.b)

            # Calculate TD_Errors
            ####################
            td_errors = self.calculateTDErrors()

            # Calculate the weight difference. If it is big enough update the
            # theta
            weight_diff = np.linalg.norm(self.representation.theta - new_theta)
            if weight_diff > self.epsilon:
                self.representation.theta = new_theta

            self.logger.info(
                "%d: %0.0f(s), ||w1-w2|| = %0.4f, Sparsity=%0.1f%%, %d Features" % (lspi_iteration + 1,
                                                                                    Tools.deltaT(
                                                                                        iteration_start_time),
                                                                                    weight_diff,
                                                                                    Tools.sparsity(
                                                                                        A),
                                                                                    self.representation.features_num))
            lspi_iteration += 1

        self.logger.info(
            'Total Policy Iteration Time = %0.0f(s)' %
            Tools.deltaT(start_time))
        return td_errors

    def LSTD(self):
        """Run the LSTD algorithm on the collected data, and update the
        policy parameters.
        """
        start_time = Tools.clock()

        if not self.fixedRep:
            # build phi_s and phi_ns for all samples
            p = self.samples_count
            n = self.representation.features_num
            self.all_phi_s = np.empty(
                (p, n), dtype=self.representation.featureType())
            self.all_phi_ns = np.empty(
                (p, n), dtype=self.representation.featureType())

            for i in np.arange(self.samples_count):
                self.all_phi_s[i, :]  = self.representation.phi(self.data_s[i])
                self.all_phi_ns[i, :] = self.representation.phi(self.data_ns[i])

            # build phi_s_a and phi_ns_na for all samples given phi_s and
            # phi_ns
            self.all_phi_s_a     = self.representation.batchPhi_s_a(self.all_phi_s[:self.samples_count, :], self.data_a[:self.samples_count,:], use_sparse=self.use_sparse)
            self.all_phi_ns_na   = self.representation.batchPhi_s_a(self.all_phi_ns[:self.samples_count, :], self.data_na[:self.samples_count,:], use_sparse=self.use_sparse)

            # calculate A and b for LSTD
            F1              = self.all_phi_s_a[:self.samples_count, :]
            F2              = self.all_phi_ns_na[:self.samples_count, :]
            R               = self.data_r[:self.samples_count, :]
            gamma = self.domain.gamma

            if self.use_sparse:
                self.b = (F1.T * R).reshape(-1, 1)
                self.A = F1.T * (F1 - gamma * F2)
            else:
                self.b = np.dot(F1.T, R).reshape(-1, 1)
                self.A = np.dot(F1.T, F1 - gamma * F2)

        A = Tools.regularize(self.A)

        # Calculate theta
        self.representation.theta, solve_time = Tools.solveLinear(A, self.b)

        # log solve time only if takes more than 1 second
        if solve_time > 1:
            self.logger.info(
                'Total LSTD Time = %0.0f(s), Solve Time = %0.0f(s)' %
                (Tools.deltaT(start_time), solve_time))
        else:
            self.logger.info(
                'Total LSTD Time = %0.0f(s)' %
                (Tools.deltaT(start_time)))

    def process(self, s, a, r, ns, na, terminal):
        """Process one transition instance."""
        # Save samples
        self.data_s[self.samples_count, :]   = s
        self.data_a[self.samples_count] = a
        self.data_r[self.samples_count] = r
        self.data_ns[self.samples_count, :]  = ns
        self.data_na[self.samples_count] = na

        # Update A and b if representation is going to be fix together with all
        # features
        if self.fixedRep:
            if terminal:
                phi_s = self.representation.phi(s, False)
                phi_s_a = self.representation.phi_sa(
                    s,
                    False,
                    a,
                    phi_s=phi_s)
            else:
                # This is because the current s,a will be the previous ns, na
                if self.use_sparse:
                    phi_s = self.all_phi_ns[self.samples_count - 1, :].todense()
                    phi_s_a = self.all_phi_ns_na[self.samples_count - 1, :].todense()
                else:
                    phi_s = self.all_phi_ns[self.samples_count - 1, :]
                    phi_s_a = self.all_phi_ns_na[self.samples_count - 1, :]

            phi_ns = self.representation.phi(ns, terminal)
            phi_ns_na = self.representation.phi_sa(
                ns,
                terminal,
                na,
                phi_s=phi_ns)

            self.all_phi_s[self.samples_count, :] = phi_s
            self.all_phi_ns[self.samples_count, :] = phi_ns
            self.all_phi_s_a[self.samples_count, :] = phi_s_a
            self.all_phi_ns_na[self.samples_count, :] = phi_ns_na

            gamma = self.domain.gamma
            self.b += phi_s_a.reshape((-1, 1)) * r
            d = phi_s_a - gamma * phi_ns_na
            self.A += np.outer(phi_s_a, d)

        self.samples_count += 1

    def calculateTDErrors(self):
        """Calculate TD errors over the transition instances stored.
        Returns the TD errors for all transitions with the current parameters.
        """
        # Calculates the TD-Errors in a matrix format for a set of samples = R
        # + (gamma*F2 - F1) * Theta
        gamma = self.representation.domain.gamma
        R       = self.data_r[:self.samples_count, :]
        if self.use_sparse:
            F1      = sp.csr_matrix(self.all_phi_s_a[:self.samples_count, :])
            F2      = sp.csr_matrix(self.all_phi_ns_na[:self.samples_count, :])
            answer = (
                R + (gamma * F2 - F1) * self.representation.theta.reshape(-1, 1))
            return np.squeeze(np.asarray(answer))
        else:
            F1 = self.all_phi_s_a[:self.samples_count, :]
            F2 = self.all_phi_ns_na[:self.samples_count, :]
            return R.ravel() + np.dot(gamma * F2 - F1, self.representation.theta)

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
            # Run Policy Iteration to change a_prime and recalculate theta in a
            # loop
            td_errors = self.policyIteration()
            # Add new Features
            if Tools.hasFunction(self.representation, 'batchDiscover'):
                added_feature = self.representation.batchDiscover(td_errors, self.all_phi_s[:self.samples_count, :], self.data_s[:self.samples_count,:])
            else:
                added_feature = False
            # print 'L_inf distance to V*= ',
            # self.domain.L_inf_distance_to_V_star(self.representation)
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
