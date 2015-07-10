"""Classical Policy Iteration.
Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the
state space. Once the errors are bounded, the policy is changed.
"""

from .MDPSolver import MDPSolver
from rlpy.Tools import className, deltaT, hhmmss, clock, l_norm
from copy import deepcopy
from rlpy.Policies import eGreedy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class PolicyIteration(MDPSolver):

    """
    Policy Iteration MDP Solver.

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

        max_PE_iterations (int):    Maximum number of Policy evaluation iterations to run.

    """

    def __init__(
            self, job_id, representation, domain, planning_time=np.inf, convergence_threshold=.005,
            ns_samples=100, project_path='.', log_interval=5000, show=False, max_PE_iterations=10):
        super(PolicyIteration, self).__init__(job_id,
                                             representation,
                                             domain,
                                             planning_time,
                                             convergence_threshold,
                                             ns_samples,
                                             project_path,
                                             log_interval,
                                             show)
        self.max_PE_iterations = max_PE_iterations
        self.bellmanUpdates = 0
        self.logger.info('Max PE Iterations:\t%d' % self.max_PE_iterations)

    def policyEvaluation(self, policy):

        '''
        Evaluate a given policy: this is done by applying the Bellman backup over all states until the change is less than
        a given threshold.

        Returns: convergence status as a boolean
        '''
        converged = False
        policy_evaluation_iteration = 0
        while (not converged and
                self.hasTime() and
                policy_evaluation_iteration < self.max_PE_iterations
                ):
            policy_evaluation_iteration += 1

            # Sweep The State Space
            for i in xrange(0, self.representation.agg_states_num):

                # Check for solver time
                if not self.hasTime(): break

                # Map an state ID to state
                s = self.representation.stateID2state(i)

                # Skip terminal states and states with no possible action
                possible_actions = self.domain.possibleActions(s=s)
                if (self.domain.isTerminal(s) or
                    len(possible_actions) == 0):
                    continue

                # Apply Bellman Backup
                self.BellmanBackup(
                    s,
                    policy.pi(s, False, possible_actions),
                    self.ns_samples,
                    policy)

                # Update number of backups
                self.bellmanUpdates += 1

                # Check for the performance
                if self.bellmanUpdates % self.log_interval == 0:
                    performance_return = self.performanceRun()[0]
                    self.logger.info(
                        '[%s]: BellmanUpdates=%d, Return=%0.4f' %
                        (hhmmss(deltaT(self.start_time)), self.bellmanUpdates, performance_return))

            # check for convergence: L_infinity norm of the difference between the to the weight vector of representation
            weight_vec_change = l_norm(policy.representation.weight_vec - self.representation.weight_vec, np.inf)
            converged = weight_vec_change < self.convergence_threshold

            # Log Status
            self.logger.info(
                'PE #%d [%s]: BellmanUpdates=%d, ||delta-weight_vec||=%0.4f' %
                (policy_evaluation_iteration, hhmmss(deltaT(self.start_time)), self.bellmanUpdates, weight_vec_change))

            # Show Plots
            if self.show:
                self.domain.show(
                                 policy.pi(s, False, possible_actions),
                                 self.representation,
                                 s=s)
        return converged

    def policyImprovement(self, policy):
        ''' Given a policy improve it by taking the greedy action in each state based on the value function
            Returns the new policy
        '''
        policyChanges = 0
        i = 0
        while i < self.representation.agg_states_num and self.hasTime():
            s = self.representation.stateID2state(i)
            if not self.domain.isTerminal(s) and len(self.domain.possibleActions(s)):
                for a in self.domain.possibleActions(s):
                    if not self.hasTime():
                        break
                    self.BellmanBackup(s, a, self.ns_samples, policy)
                if policy.pi(s, False, self.domain.possibleActions(s=s)) != self.representation.bestAction(s, False, self.domain.possibleActions(s=s)):
                    policyChanges += 1
            i += 1
        # This will cause the policy to be copied over
        policy.representation.weight_vec = self.representation.weight_vec.copy()
        performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun(
        )
        self.logger.info(
            'PI #%d [%s]: BellmanUpdates=%d, Policy Change=%d, Return=%0.4f, Steps=%d' % (
                self.policy_improvement_iteration,
                hhmmss(
                    deltaT(
                        self.start_time)),
                self.bellmanUpdates,
                policyChanges,
                performance_return,
                performance_steps))

        # store stats
        self.result["bellman_updates"].append(self.bellmanUpdates)
        self.result["return"].append(performance_return)
        self.result["planning_time"].append(deltaT(self.start_time))
        self.result["num_features"].append(self.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["discounted_return"].append(performance_discounted_return)
        self.result["policy_improvement_iteration"].append(self.policy_improvement_iteration)
        return policy, policyChanges

    def solve(self):
        """Solve the domain MDP."""
        self.bellmanUpdates = 0
        self.policy_improvement_iteration = 0
        self.start_time = clock()

        # Check for Tabular Representation
        if not self.IsTabularRepresentation():
            self.logger.error("Policy Iteration works only with a tabular representation.")
            return 0

        # Initialize the policy
        policy = eGreedy(
            deepcopy(self.representation),
            epsilon=0,
            forcedDeterministicAmongBestActions=True)  # Copy the representation so that the weight change during the evaluation does not change the policy

        # Setup the number of policy changes to 1 so the while loop starts
        policyChanges = True

        while policyChanges and deltaT(self.start_time) < self.planning_time:

            # Evaluate the policy
            converged = self.policyEvaluation(policy)

            # Improve the policy
            self.policy_improvement_iteration += 1
            policy, policyChanges = self.policyImprovement(policy)

        super(PolicyIteration, self).solve()
