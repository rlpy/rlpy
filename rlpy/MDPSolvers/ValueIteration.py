"""Classical Value Iteration
Performs full Bellman Backup on a given s,a pair by sweeping through the state space
"""

from .MDPSolver import MDPSolver
from rlpy.Tools import hhmmss, deltaT, className, clock
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class ValueIteration(MDPSolver):

    """Value Iteration MDP Solver.

    Args:
        job_id (int):   Job ID number used for running multiple jobs on a cluster.

        representation (Representation):    Representation used for the value function.

        domain (Domain):    Domain (MDP) to solve.

        logger (Logger):    Logger object to log information and debugging.

        planning_time (int):    Maximum amount of time in seconds allowed for planning. Defaults to inf (unlimited).

        convergence_threshold (float):  Threshold for determining if the value function has converged.

        ns_samples (int):   How many samples of the successor states to take.

        project_path (str): Output path for saving the results of running the MDPSolver on a domain.

        log_interval (int): Minimum number of seconds between displaying logged information.

        show (bool):    Enable visualization?

    .. warning::

        THE CURRENT IMPLEMENTATION ASSUMES *DETERMINISTIC* TRANSITIONS:
        In other words, in each iteration, from each state, we only sample
        each possible action **once**. \n
        For stochastic domains, it is necessary to sample multiple times and
        use the average.
    """

    def solve(self):
        """Solve the domain MDP."""

        self.result = []
        # Used to show the total time took the process
        self.start_time = clock()
        # Check for Tabular Representation
        rep = self.representation
        if className(rep) != 'Tabular':
            self.logger.log(
                "Value Iteration works only with the tabular representation.")
            return 0

        no_of_states = self.representation.agg_states_num
        # used to track the performance improvement.
        bellmanUpdates = 0
        converged = False
        iteration = 0
        while self.hasTime() and not converged:
            prev_theta = self.representation.theta.copy()
            # Sweep The State Space
            for i in xrange(0, no_of_states):
                if not self.hasTime():
                    break
                s = self.representation.stateID2state(i)
                actions = self.domain.possibleActions(s)
                # Sweep The Actions
                for a in actions:
                    if not self.hasTime():
                        break
                    self.BellmanBackup(s, a, ns_samples=self.ns_samples)
                    bellmanUpdates += 1

                    if bellmanUpdates % self.log_interval == 0:
                        performance_return, _, _, _ = self.performanceRun()
                        self.logger.log(
                            '[%s]: BellmanUpdates=%d, Return=%0.4f' %
                            (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))

            # check for convergence
            iteration += 1
            theta_change = np.linalg.norm(
                prev_theta -
                self.representation.theta,
                np.inf)
            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun(
            )
            converged = theta_change < self.convergence_threshold
            self.logger.log(
                'PI #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f, Return=%0.4f, Steps=%d' % (iteration,
                                                                                                   hhmmss(
                                                                                                       deltaT(
                                                                                                           self.start_time)),
                                                                                                   bellmanUpdates,
                                                                                                   theta_change,
                                                                                                   performance_return,
                                                                                                   performance_steps))
            if self.show:
                self.domain.show(a, s=s, representation=self.representation)

            # store stats
            self.result.append([bellmanUpdates,  # index = 0
                               performance_return,  # index = 1
                               deltaT(self.start_time),  # index = 2
                               self.representation.features_num,  # index = 3
                               performance_steps,  # index = 4
                               performance_term,  # index = 5
                               performance_discounted_return,  # index = 6
                               iteration  # index = 7
                                ])

        if converged:
            self.logger.log('Converged!')
        super(ValueIteration, self).solve()
