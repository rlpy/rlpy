"""Trajectory Based Policy Iteration:
    Loop until the weight change to the value function is small for some number of trajectories
    (cant check policy because we dont store anything in the size of the state-space)
    1. Update the evaluation of the policy till the change is small.
    2. Update the policy

    * There is solveInMatrixFormat function which does policy evaluation in one shot using samples collected in the matrix format.
      Since the algorithm toss out the samples, convergence is hardly reached because the policy may alternate.
"""

from .MDPSolver import MDPSolver
from rlpy.Tools import className, hhmmss, deltaT, randSet, hasFunction, solveLinear, regularize, clock, padZeros, l_norm
from rlpy.Policies import eGreedy
import numpy as np
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class TrajectoryBasedPolicyIteration(MDPSolver):

    """Trajectory Based Policy Iteration MDP Solver.

    Args:
        job_id (int):   Job ID number used for running multiple jobs on a cluster.

        representation (Representation):    Representation used for the value function.

        domain (Domain):    Domain (MDP) to solve.

        planning_time (int):    Maximum amount of time in seconds allowed for planning. Defaults to inf (unlimited).

        convergence_threshold (float):  Threshold for determining if the value function has converged.

        ns_samples (int):   How many samples of the successor states to take.

        project_path (str): Output path for saving the results of running the MDPSolver on a domain.

        log_interval (int): Minimum number of seconds between displaying logged information.

        show (bool):    Enable visualization?

        epsilon (float):    Probability of taking a random action during each decision making.

        max_PE_iterations (int):    Maximum number of Policy evaluation iterations to run.

    """

    # Probability of taking a random action during each decision making
    epsilon = None

    # step size parameter to adjust the weights. If the representation is
    # tabular you can set this to 1.
    alpha = .1

    # Minimum number of trajectories required for convergence in which the max
    # bellman error was below the threshold
    MIN_CONVERGED_TRAJECTORIES = 5

    def __init__(
            self, job_id, representation, domain, planning_time=np.inf, convergence_threshold=.005,
            ns_samples=100, project_path='.', log_interval=500, show=False, epsilon=.1, max_PE_iterations=10):
        super(
            TrajectoryBasedPolicyIteration,
            self).__init__(job_id,
                           representation,
                           domain,
                           planning_time,
                           convergence_threshold,
                           ns_samples,
                           project_path,
                           log_interval,
                           show)

        self.epsilon = epsilon
        self.max_PE_iterations = max_PE_iterations
        if self.IsTabularRepresentation(): self.alpha = 1

    def sample_ns_na(self, policy, action=None, start_trajectory=False):
        ''' Given a policy sample the next state and next action along the trajectory followed by the policy
        * Noise is added in selecting action:
        with probability 1-e, follow the policy
        with probability self.epsilon pick a uniform random action from possible actions
        * if start_trajectory = True the initial state is sampled from s0() function of the domain otherwise
        take the action given in the current state
        '''
        if start_trajectory:
            ns, terminal, possible_actions = self.domain.s0()
        else:
            _, ns, terminal, possible_actions = self.domain.step(action)

        if np.random.rand() > self.epsilon:
            na = policy.pi(ns, terminal, possible_actions)
        else:
            na = randSet(possible_actions)

        return ns, na, terminal

    def trajectoryBasedPolicyEvaluation(self, policy):
        ''' evaluate the current policy by simulating trajectories and update the value function along the
        visited states
        '''
        PE_iteration = 0
        evaluation_is_accurate = False
        converged_trajectories = 0
        while not evaluation_is_accurate and self.hasTime() and PE_iteration < self.max_PE_iterations:

            # Generate a new episode e-greedy with the current values
            max_Bellman_Error = 0
            step = 0

            s, a, terminal = self.sample_ns_na(policy, start_trajectory=True)

            while not terminal and step < self.domain.episodeCap and self.hasTime():
                new_Q = self.representation.Q_oneStepLookAhead(s, a, self.ns_samples, policy)
                phi_s = self.representation.phi(s, terminal)
                phi_s_a = self.representation.phi_sa(s, terminal, a, phi_s=phi_s)
                old_Q = np.dot(phi_s_a, self.representation.weight_vec)
                bellman_error = new_Q - old_Q

                # Update the value function using approximate bellman backup
                self.representation.weight_vec += (self.alpha * bellman_error * phi_s_a)
                self.bellmanUpdates += 1
                step += 1
                max_Bellman_Error = max(max_Bellman_Error, abs(bellman_error))

                # Discover features if the representation has the discover method
                discover_func = getattr(self.representation, 'discover', None)  # None is the default value if the discover is not an attribute
                if discover_func and callable(discover_func):
                    self.representation.post_discover(phi_s, bellman_error)
                    # if discovered:
                    # print "Features = %d" % self.representation.features_num

                s, a, terminal = self.sample_ns_na(policy, a)

            # check for convergence of policy evaluation
            PE_iteration += 1
            if max_Bellman_Error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            evaluation_is_accurate = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES
            self.logger.info(
                'PE #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Features=%d' % (PE_iteration,
                                                                                          hhmmss(
                                                                                              deltaT(
                                                                                                  self.start_time)),
                                                                                          self. bellmanUpdates,
                                                                                          max_Bellman_Error,
                                                                                          self.representation.features_num))

    def solve(self):
        """Solve the domain MDP."""

        self.start_time = clock()  # Used to track the total time for solving
        self.bellmanUpdates = 0
        converged = False
        PI_iteration = 0

        # The policy is maintained as separate copy of the representation.
        # This way as the representation is updated the policy remains intact
        policy = eGreedy(
            deepcopy(self.representation),
            epsilon=0,
            forcedDeterministicAmongBestActions=True)

        while self.hasTime() and not converged:

            self.trajectoryBasedPolicyEvaluation(policy)

            # Policy Improvement (Updating the representation of the value
            # function will automatically improve the policy
            PI_iteration += 1

            # Theta can increase in size if the representation is expanded hence padding the weight vector with zeros
            paddedTheta = padZeros(policy.representation.weight_vec, len(self.representation.weight_vec))

            # Calculate the change in the weight_vec as L2-norm
            delta_weight_vec = np.linalg.norm(paddedTheta - self.representation.weight_vec)
            converged = delta_weight_vec < self.convergence_threshold

            # Update the underlying value function of the policy
            policy.representation = deepcopy(self.representation)  # self.representation

            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun()
            self.logger.info(
                'PI #%d [%s]: BellmanUpdates=%d, ||delta-weight_vec||=%0.4f, Return=%0.3f, steps=%d, features=%d' % (PI_iteration,
                                                                                                                hhmmss(
                                                                                                                    deltaT(
                                                                                                                        self.start_time)),
                                                                                                                self.bellmanUpdates,
                                                                                                                delta_weight_vec,
                                                                                                                performance_return,
                                                                                                                performance_steps,
                                                                                                                self.representation.features_num))
            if self.show:
                self.domain.show(a, representation=self.representation, s=s)

            # store stats
            self.result["bellman_updates"].append(self.bellmanUpdates)
            self.result["return"].append(performance_return)
            self.result["planning_time"].append(deltaT(self.start_time))
            self.result["num_features"].append(self.representation.features_num)
            self.result["steps"].append(performance_steps)
            self.result["terminated"].append(performance_term)
            self.result["discounted_return"].append(performance_discounted_return)
            self.result["policy_improvemnt_iteration"].append(PI_iteration)

        if converged:
            self.logger.info('Converged!')
        super(TrajectoryBasedPolicyIteration, self).solve()

    def solveInMatrixFormat(self):
        # while delta_weight_vec > threshold
        #  1. Gather data following an e-greedy policy
        #  2. Calculate A and b estimates
        #  3. calculate new_weight_vec, and delta_weight_vec
        # return policy greedy w.r.t last weight_vec
        self.policy = eGreedy(
            self.representation,
            epsilon=self.epsilon)

        # Number of samples to be used for each policy evaluation phase. L1 in
        # the Geramifard et. al. FTML 2012 paper
        self.samples_num = 1000

        self.start_time = clock()  # Used to track the total time for solving
        samples = 0
        converged = False
        iteration = 0
        while self.hasTime() and not converged:

            #  1. Gather samples following an e-greedy policy
            S, Actions, NS, R, T = self.collectSamples(self.samples_num)
            samples += self.samples_num

            #  2. Calculate A and b estimates
            a_num = self.domain.actions_num
            n = self.representation.features_num
            discount_factor = self.domain.discount_factor

            self.A = np.zeros((n * a_num, n * a_num))
            self.b = np.zeros((n * a_num, 1))
            for i in xrange(self.samples_num):
                phi_s_a = self.representation.phi_sa(
                    S[i], T[i], Actions[i, 0]).reshape((-1, 1))
                E_phi_ns_na = self.calculate_expected_phi_ns_na(
                    S[i], Actions[i, 0], self.ns_samples).reshape((-1, 1))
                d = phi_s_a - discount_factor * E_phi_ns_na
                self.A += np.outer(phi_s_a, d.T)
                self.b += phi_s_a * R[i, 0]

            #  3. calculate new_weight_vec, and delta_weight_vec
            new_weight_vec, solve_time = solveLinear(regularize(self.A), self.b)
            iteration += 1
            if solve_time > 1:
                self.logger.info(
                    '#%d: Finished Policy Evaluation. Solve Time = %0.2f(s)' %
                    (iteration, solve_time))
            delta_weight_vec = l_norm(new_weight_vec - self.representation.weight_vec, np.inf)
            converged = delta_weight_vec < self.convergence_threshold
            self.representation.weight_vec = new_weight_vec
            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun()
            self.logger.info(
                '#%d [%s]: Samples=%d, ||weight-Change||=%0.4f, Return = %0.4f' %
                (iteration, hhmmss(deltaT(self.start_time)), samples, delta_weight_vec, performance_return))
            if self.show:
                self.domain.show(S[-1], Actions[-1], self.representation)

            # store stats
            self.result["samples"].append(samples)
            self.result["return"].append(performance_return)
            self.result["planning_time"].append(deltaT(self.start_time))
            self.result["num_features"].append(self.representation.features_num)
            self.result["steps"].append(performance_steps)
            self.result["terminated"].append(performance_term)
            self.result["discounted_return"].append(performance_discounted_return)
            self.result["iteration"].append(iteration)

        if converged:
            self.logger.info('Converged!')

        super(TrajectoryBasedPolicyIteration, self).solve()

    def calculate_expected_phi_ns_na(self, s, a, ns_samples):
        # calculate the expected next feature vector (phi(ns,pi(ns)) given s
        # and a. Eqns 2.20 and 2.25 in [Geramifard et. al. 2012 FTML Paper]
        if hasFunction(self.domain, 'expectedStep'):
            p, r, ns, t, pa = self.domain.expectedStep(s, a)
            phi_ns_na = np.zeros(
                self.representation.features_num *
                self.domain.actions_num)
            for j in xrange(len(p)):
                na = self.policy.pi(ns[j], t[j], pa[j])
                phi_ns_na += p[j] * self.representation.phi_sa(ns[j], t[j], na)
        else:
            next_states, rewards = self.domain.sampleStep(s, a, ns_samples)
            phi_ns_na = np.mean(
                [self.representation.phisa(next_states[i],
                                           self.policy.pi(next_states[i])) for i in xrange(ns_samples)])
        return phi_ns_na
