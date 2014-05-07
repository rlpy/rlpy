"""Classical Policy Iteration.
Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the
state space. Once the errors are bounded, the policy is changed.
"""

from .MDPSolver import MDPSolver
from rlpy.Tools import className, deltaT, hhmmss, clock
from copy import deepcopy
from rlpy.Policies import eGreedy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
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
        super(
            PolicyIteration,
            self).__init__(job_id,
                           representation,
                           domain,
                           planning_time,
                           convergence_threshold,
                           ns_samples,
                           project_path,
                           log_interval,
                           show)
        self.max_PE_iterations = max_PE_iterations
        self.logger.info('Max PE Iterations:\t%d' % self.max_PE_iterations)

    def solve(self):
        """Solve the domain MDP."""
        self.result = []
        # Used to show the total time took the process
        self.start_time = clock()

        # Check for Tabular Representation
        rep = self.representation
        if className(rep) != 'Tabular':
            self.logger.error(
                "Value Iteration works only with the tabular representation.")
            return 0

        no_of_states = self.representation.agg_states_num
        bellmanUpdates = 0

        # Initialize the policy
        policy = eGreedy(
            deepcopy(self.representation),
            epsilon=0,
            forcedDeterministicAmongBestActions=True)  # Copy the representation so that the weight change during the evaluation does not change the policy
        policyChanged = True

        policy_improvement_iteration = 0
        while policyChanged and deltaT(self.start_time) < self.planning_time:

            # Policy Evaluation
            converged = False
            policy_evaluation_iteration = 0
            while not converged and self.hasTime() and policy_evaluation_iteration < self.max_PE_iterations:
                policy_evaluation_iteration += 1
                # Sweep The State Space
                for i in xrange(0, no_of_states):
                    if not self.hasTime():
                        break
                    s = self.representation.stateID2state(i)
                    if not self.domain.isTerminal(s) and len(self.domain.possibleActions(s=s)):
                        self.BellmanBackup(
                            s,
                            policy.pi(s,
                                      False,
                                      self.domain.possibleActions(s=s)),
                            self.ns_samples,
                            policy)
                        bellmanUpdates += 1

                        if bellmanUpdates % self.log_interval == 0:
                            performance_return, _, _, _ = self.performanceRun(
                            )
                            self.logger.info(
                                '[%s]: BellmanUpdates=%d, Return=%0.4f' %
                                (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))

                # check for convergence
                theta_change = np.linalg.norm(
                    policy.representation.theta -
                    self.representation.theta,
                    np.inf)
                converged = theta_change < self.convergence_threshold
                self.logger.info(
                    'PE #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f' %
                    (policy_evaluation_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change))
                if self.show:
                    self.domain.show(
                        policy.pi(s,
                                  False,
                                  self.domain.possibleActions(s=s)),
                        self.representation,
                        s=s)

            # Policy Improvement:
            policy_improvement_iteration += 1
            policyChanged = 0
            i = 0
            while i < no_of_states and self.hasTime():
                s = self.representation.stateID2state(i)
                if not self.domain.isTerminal(s) and len(self.domain.possibleActions(s)):
                    for a in self.domain.possibleActions(s):
                        if not self.hasTime():
                            break
                        self.BellmanBackup(s, a, self.ns_samples, policy)
                    if policy.pi(s, False, self.domain.possibleActions(s=s)) != self.representation.bestAction(s, False, self.domain.possibleActions(s=s)):
                        policyChanged += 1
                i += 1
            # This will cause the policy to be copied over
            policy.representation.theta = self.representation.theta.copy()
            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun(
            )
            self.logger.info(
                'PI #%d [%s]: BellmanUpdates=%d, Policy Change=%d, Return=%0.4f, Steps=%d' % (
                    policy_improvement_iteration,
                    hhmmss(
                        deltaT(
                            self.start_time)),
                    bellmanUpdates,
                    policyChanged,
                    performance_return,
                    performance_steps))

            # store stats
            self.result.append([bellmanUpdates,  # index = 0
                               performance_return,  # index = 1
                               deltaT(self.start_time),  # index = 2
                               self.representation.features_num,  # index = 3
                               performance_steps,  # index = 4
                               performance_term,  # index = 5
                               performance_discounted_return,  # index = 6
                               policy_improvement_iteration  # index = 7
                                ])

        if converged:
            self.logger.info('Converged!')
        super(PolicyIteration, self).solve()
