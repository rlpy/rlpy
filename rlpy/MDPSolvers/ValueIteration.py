"""Classical Value Iteration
Performs full Bellman Backup on a given s,a pair by sweeping through the state space
"""

from .MDPSolver import MDPSolver
from rlpy.Tools import hhmmss, deltaT, className, clock, l_norm
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
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

        self.start_time = clock()  # Used to show the total time took the process
        bellmanUpdates = 0  # used to track the performance improvement.
        converged = False
        iteration = 0

        # Check for Tabular Representation
        if not self.IsTabularRepresentation():
            self.logger.error("Value Iteration works only with a tabular representation.")
            return 0

        no_of_states = self.representation.agg_states_num

        while self.hasTime() and not converged:

            iteration += 1

            # Store the weight vector for comparison
            prev_weight_vec = self.representation.weight_vec.copy()

            # Sweep The State Space
            for i in xrange(no_of_states):

                s = self.representation.stateID2state(i)

                # Sweep through possible actions
                for a in self.domain.possibleActions(s):

                    # Check for available planning time
                    if not self.hasTime(): break

                    self.BellmanBackup(s, a, ns_samples=self.ns_samples)
                    bellmanUpdates += 1

                    # Create Log
                    if bellmanUpdates % self.log_interval == 0:
                        performance_return, _, _, _ = self.performanceRun()
                        self.logger.info(
                            '[%s]: BellmanUpdates=%d, Return=%0.4f' %
                            (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))

            # check for convergence
            weight_vec_change = l_norm(prev_weight_vec - self.representation.weight_vec, np.inf)
            converged = weight_vec_change < self.convergence_threshold

            # log the stats
            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun()
            self.logger.info(
                'PI #%d [%s]: BellmanUpdates=%d, ||delta-weight_vec||=%0.4f, Return=%0.4f, Steps=%d' % (iteration,
                 hhmmss(deltaT(self.start_time)),
                 bellmanUpdates,
                 weight_vec_change,
                 performance_return,
                 performance_steps))

            # Show the domain and value function
            if self.show:
                self.domain.show(a, s=s, representation=self.representation)

            # store stats
            self.result["bellman_updates"].append(bellmanUpdates)
            self.result["return"].append(performance_return)
            self.result["planning_time"].append(deltaT(self.start_time))
            self.result["num_features"].append(self.representation.features_num)
            self.result["steps"].append(performance_steps)
            self.result["terminated"].append(performance_term)
            self.result["discounted_return"].append(performance_discounted_return)
            self.result["iteration"].append(iteration)

        if converged: self.logger.info('Converged!')
        super(ValueIteration, self).solve()
