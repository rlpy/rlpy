"""MDP Solver base class."""

from abc import ABCMeta, abstractmethod
import numpy as np
import logging
from copy import deepcopy
from rlpy.Tools import className, deltaT, hhmmss, clock, l_norm, vec2id, checkNCreateDirectory
from collections import defaultdict
import os
import json

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "N. Kemal Ure"


class MDPSolver(object):

    """MDPSolver is the base class for model based reinforcement learning agents and
    planners.

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

    """

    __metaclass__ = ABCMeta

    representation = None  # Link to the representation object
    domain = None  # Link to the domain object
    # A simple objects that record the prints in a file
    logger = None
    # Amount of time in seconds provided for the solver. After this it returns
    # its performance.
    planning_time = None
    # The job id of this run of the algorithm
    exp_id = None
    # To make sure all same job ids see the same random sequence
    mainSeed = 999999999
    # Maximum number of runs of an algorithm for averaging
    maxRuns = 100
    # Threshold to determine the convergence of the planner
    convergence_threshold = None
    # Number of samples to be used to generate estimated bellman backup if the
    # domain does not provide explicit probabilities though expectedStep
    # function.
    ns_samples = None
    # Number of bellman backups before reporting the performance. (Not all
    # planners may use this)
    log_interval = None
    show = None  # Show the learning if possible?

    def __init__(
            self, job_id, representation, domain, planning_time=np.inf,
            convergence_threshold=.005, ns_samples=100, project_path='.', log_interval=5000, show=False):
        self.exp_id = job_id
        self.representation = representation
        self.domain = domain
        self.logger = logging.getLogger("rlpy.MDPSolvers." + self.__class__.__name__)
        self.ns_samples = ns_samples
        self.planning_time = planning_time
        self.project_path = project_path
        self.log_interval = log_interval
        self.show = show
        self.convergence_threshold = convergence_threshold

        # Set random seed for this job id
        np.random.seed(self.mainSeed)
        self.randomSeeds = np.random.randint(1, self.mainSeed, (self.maxRuns, 1))
        np.random.seed(self.randomSeeds[self.exp_id - 1, 0])

        # TODO setup logging to file in experiment

        # create a dictionary of results
        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        self.output_filename = '{:0>3}-results.json'.format(self.exp_id)

    @abstractmethod
    def solve(self):
        """Solve the domain MDP."""
        # Abstract
        self.logger.info(
            'Value of S0 is = %0.5f' %
            self.representation.V(*self.domain.s0()))
        self.saveStats()

    def printAll(self):
        printClass(self)

    def BellmanBackup(self, s, a, ns_samples, policy=None):
        """Applied Bellman Backup to state-action pair s,a
        i.e. Q(s,a) = E[r + discount_factor * V(s')]
        If policy is given then Q(s,a) =  E[r + discount_factor * Q(s',pi(s')]

        Args:
            s (ndarray):        The current state
            a (int):            The action taken in state s
            ns_samples(int):    Number of next state samples to use.
            policy (Policy):    Policy object to use for sampling actions.
        """
        Q = self.representation.Q_oneStepLookAhead(
            s,
            a,
            ns_samples,
            policy)
        s_index = vec2id(
            self.representation.binState(s),
            self.representation.bins_per_dim)
        weight_vec_index = int(self.representation.agg_states_num * a + s_index)
        self.representation.weight_vec[weight_vec_index] = Q

    def performanceRun(self):
        """Set Exploration to zero and sample one episode from the domain."""

        eps_length = 0
        eps_return = 0
        eps_term = False
        eps_discounted_return = 0

        s, eps_term, p_actions = self.domain.s0()
        # if self.visualize_performance:
        #    self.domain.showLearning(self.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a = self.representation.bestAction(
                s,
                eps_term,
                p_actions)
            r, ns, eps_term, p_actions = self.domain.step(a)
            s = ns
            eps_discounted_return += self.domain.discount_factor ** eps_length * r
            eps_return += r
            eps_length += 1
        return eps_return, eps_length, eps_term, eps_discounted_return

    def saveStats(self):
        fullpath_output = os.path.join(self.project_path, self.output_filename)
        print ">>> ", fullpath_output
        checkNCreateDirectory(self.project_path + '/')
        with open(fullpath_output, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

    def hasTime(self):
        """Return a boolean stating if there is time left for planning."""
        return deltaT(self.start_time) < self.planning_time

    def IsTabularRepresentation(self):
        '''
        Check to see if the representation is Tabular as Policy Iteration and Value Iteration only work with
        Tabular representation
        '''
        return className(self.representation) == 'Tabular'

        return True

    def collectSamples(self, samples):
        """
        Return matrices of S,A,NS,R,T where each row of each numpy 2d-array
        is a sample by following the current policy.

        - S: (#samples) x (# state space dimensions)
        - A: (#samples) x (1) int [we are storing actionIDs here, integers]
        - NS:(#samples) x (# state space dimensions)
        - R: (#samples) x (1) float
        - T: (#samples) x (1) bool

        See :py:meth:`~rlpy.Agents.Agent.Agent.Q_MC` and :py:meth:`~rlpy.Agents.Agent.Agent.MC_episode`
        """
        domain = self.representation.domain
        S = np.empty(
            (samples,
             self.representation.domain.state_space_dims),
            dtype=type(domain.s0()))
        A = np.empty((samples, 1), dtype='uint16')
        NS = S.copy()
        T = A.copy()
        R = np.empty((samples, 1))

        sample = 0
        eps_length = 0
        # So the first sample forces initialization of s and a
        terminal = True
        while sample < samples:
            if terminal or eps_length > self.representation.domain.episodeCap:
                s, terminal, possible_actions = domain.s0()
                a = self.policy.pi(s, terminal, possible_actions)

            # Transition
            r, ns, terminal, possible_actions = domain.step(a)
            # Collect Samples
            S[sample] = s
            A[sample] = a
            NS[sample] = ns
            T[sample] = terminal
            R[sample] = r

            sample += 1
            eps_length += 1
            s = ns
            a = self.policy.pi(s, terminal, possible_actions)

        return S, A, NS, R, T
