"""Standard Experiment for Learning Control in RL."""

import logging
from rlpy.Tools import plt
import numpy as np
from copy import deepcopy
import re
import argparse
from rlpy.Tools import deltaT, clock, hhmmss
from rlpy.Tools import className, checkNCreateDirectory
from rlpy.Tools import printClass
import rlpy.Tools.results
from rlpy.Tools import lower
import os
import rlpy.Tools.ipshell
import json
from collections import defaultdict

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Experiment(object):

    """
    The Experiment controls the training, testing, and evaluation of the
    agent. Reinforcement learning is based around
    the concept of training an :py:class:`~Agents.Agent.Agent` to solve a task,
    and later testing its ability to do so based on what it has learned.
    This cycle forms a loop that the experiment defines and controls. First
    the agent is repeatedly tasked with solving a problem determined by the
    :py:class:`~Domains.Domain.Domain`, restarting after some termination
    condition is reached.
    (The sequence of steps between terminations is known as an *episode*.)

    Each time the Agent attempts to solve the task, it learns more about how
    to accomplish its goal. The experiment controls this loop of "training
    sessions", iterating over each step in which the Agent and Domain interact.
    After a set number of training sessions defined by the experiment, the
    agent's current policy is tested for its performance on the task.
    The experiment collects data on the agent's performance and then puts the
    agent through more training sessions. After a set number of loops, training
    sessions followed by an evaluation, the experiment is complete and the
    gathered data is printed and saved. For each section, training and
    evaluation, the experiment determines whether or not the visualization
    of the step should generated.

    The Experiment class is a base class that provides
    the basic framework for all RL experiments. It provides the methods and
    attributes that allow child classes to interact with the Agent
    and Domain classes within the RLPy library.

    .. note::
        All experiment implementations should inherit from this class.
    """

    #: The Main Random Seed used to generate other random seeds (we use a different seed for each experiment id)
    mainSeed = 999999999
    #: Maximum number of runs used for averaging, specified so that enough
    #: random seeds are generated
    maxRuns = 1000
    # Array of random seeds. This is used to make sure all jobs start with
    # the same random seed
    randomSeeds = np.random.RandomState(mainSeed).randint(1, mainSeed, maxRuns)

    #: ID of the current experiment (main seed used for calls to np.rand)
    exp_id = 1

    # The domain to be tested on
    domain = None
    # The agent to be tested
    agent = None

    #: A 2-d numpy array that stores all generated results.The purpose of a run
    #: is to fill this array. Size is stats_num x num_policy_checks.
    result = None
    #: The name of the file used to store the data
    output_filename = ''
    # A simple object that records the prints in a file
    logger = None

    max_steps = 0  # Total number of interactions
    # Number of Performance Checks uniformly scattered along timesteps of the
    # experiment
    num_policy_checks = 0
    log_interval = 0  # Number of seconds between log prints to console

    log_template = '{total_steps: >6}: E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'
    performance_log_template = '{total_steps: >6}: >>> E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'

    def __init__(self, agent, domain, exp_id=1, max_steps=max_steps,
                 config_logging=True, num_policy_checks=10, log_interval=1,
                 path='Results/Temp',
                 checks_per_policy=1, stat_bins_per_state_dim=0, **kwargs):
        """
        :param agent: the :py:class:`~Agents.Agent.Agent` to use for learning the task.
        :param domain: the problem :py:class:`~Domains.Domain.Domain` to learn
        :param exp_id: ID of this experiment (main seed used for calls to np.rand)
        :param max_steps: Total number of interactions (steps) before experiment termination.

        .. note::
            ``max_steps`` is distinct from ``episodeCap``; ``episodeCap`` defines the
            the largest number of interactions which can occur in a single
            episode / trajectory, while ``max_steps`` limits the sum of all
            interactions over all episodes which can occur in an experiment.

        :param num_policy_checks: Number of Performance Checks uniformly
            scattered along timesteps of the experiment
        :param log_interval: Number of seconds between log prints to console
        :param path: Path to the directory to be used for results storage
            (Results are stored in ``path/output_filename``)
        :param checks_per_policy: defines how many episodes should be run to
            estimate the performance of a single policy

        """
        self.exp_id = exp_id
        assert exp_id > 0
        self.agent = agent
        self.checks_per_policy = checks_per_policy
        self.domain = domain
        self.max_steps = max_steps
        self.num_policy_checks = num_policy_checks
        self.logger = logging.getLogger("rlpy.Experiments.Experiment")
        self.log_interval = log_interval
        self.config_logging = config_logging
        self.path = path
        if stat_bins_per_state_dim > 0:
            self.state_counts_learn = np.zeros(
                (domain.statespace_limits.shape[0],
                 stat_bins_per_state_dim), dtype=np.long)
            self.state_counts_perf = np.zeros(
                (domain.statespace_limits.shape[0],
                 stat_bins_per_state_dim), dtype=np.long)

    def _update_path(self, path):

        # compile and create output path
        self.full_path = self.compile_path(path)
        checkNCreateDirectory(self.full_path + '/')
        self.logger.info(
            "Output:\t\t\t%s/%s" %
            (self.full_path, self.output_filename))
        #TODO set up logging to file for rlpy loggers

        self.log_filename = '{:0>3}.log'.format(self.exp_id)
        if self.config_logging:
            rlpy_logger = logging.getLogger("rlpy")
            for h in rlpy_logger.handlers:
                rlpy_logger.removeHandler(h)
            rlpy_logger.addHandler(logging.StreamHandler())
            rlpy_logger.addHandler(logging.FileHandler(os.path.join(self.full_path, self.log_filename)))
            rlpy_logger.setLevel(logging.INFO)

    def seed_components(self):
        """
        set the initial seeds for all random number generators used during
        the experiment run based on the currently set ``exp_id``.
        """
        self._update_path(self.path)
        self.output_filename = '{:0>3}-results.json'.format(self.exp_id)
        np.random.seed(self.randomSeeds[self.exp_id - 1])
        self.domain.random_state = np.random.RandomState(
            self.randomSeeds[self.exp_id - 1])
        self.domain.init_randomization()
        # make sure the performance_domain has a different seed
        self.performance_domain.random_state = np.random.RandomState(
            self.randomSeeds[self.exp_id + 20])

        # Its ok if use same seed as domain, random calls completely different
        self.agent.random_state = np.random.RandomState(
            self.randomSeeds[self.exp_id - 1])
        self.agent.init_randomization()
        self.agent.representation.random_state = np.random.RandomState(
            self.randomSeeds[self.exp_id - 1])
        self.agent.representation.init_randomization()
        self.agent.policy.random_state = np.random.RandomState(
            self.randomSeeds[self.exp_id - 1])
        self.agent.policy.init_randomization()

        self.log_filename = '{:0>3}.log'.format(self.exp_id)
        if self.config_logging:
            rlpy_logger = logging.getLogger("rlpy")
            for h in rlpy_logger.handlers:
                if isinstance(h, logging.FileHandler):
                    rlpy_logger.removeHandler(h)
            rlpy_logger.addHandler(logging.FileHandler(os.path.join(self.full_path, self.log_filename)))

    def performanceRun(self, total_steps, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        :param total_steps: int
            maximum number of steps of the episode to peform
        :param visualize: boolean, optional
            defines whether to show each step or not (if implemented by the domain)
        """

        # Set Exploration to zero and sample one episode from the domain
        eps_length = 0
        eps_return = 0
        eps_discount_return = 0
        eps_term = 0

        self.agent.policy.turnOffExploration()

        s, eps_term, p_actions = self.performance_domain.s0()

        while not eps_term and eps_length < self.domain.episodeCap:
            a = self.agent.policy.pi(s, eps_term, p_actions)
            if visualize:
                self.performance_domain.showDomain(a)

            r, ns, eps_term, p_actions = self.performance_domain.step(a)
            self._gather_transition_statistics(s, a, ns, r, learning=False)
            s = ns
            eps_return += r
            eps_discount_return += self.performance_domain.discount_factor ** eps_length * \
                r
            eps_length += 1
        if visualize:
            self.performance_domain.showDomain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying
        # to accomodate them as an MDP

        return eps_return, eps_length, eps_term, eps_discount_return

    def printAll(self):
        """
        prints all information about the experiment
        """
        printClass(self)

    def _gather_transition_statistics(self, s, a, sn, r, learning=False):
        """
        This function can be used in subclasses to collect statistics about the transitions
        """
        if hasattr(self, "state_counts_learn") and learning:
            counts = self.state_counts_learn
        elif hasattr(self, "state_counts_perf") and not learning:
            counts = self.state_counts_perf
        else:
            return
        rng = self.domain.statespace_limits[:, 1] - \
            self.domain.statespace_limits[:, 0]
        d = counts.shape[-1] - 2
        s_norm = s - self.domain.statespace_limits[:, 0]
        idx = np.floor(s_norm / rng * d).astype("int")
        idx += 1
        idx[idx < 0] = 0
        idx[idx >= d + 2] = d + 1
        #import ipdb; ipdb.set_trace()
        counts[range(counts.shape[0]), idx] += 1

    def run_from_commandline(self):
        """
        wrapper around run method which automatically reads run parameters from command line arguments
        """

        parser = argparse.ArgumentParser("Run rlpy experiments")
        parser.add_argument(
            "-v", "--show-plot", default=False, action="store_true",
            help="show a plot at the end with the results of the assessment runs")
        parser.add_argument("-w", "--save", default=False, action="store_true",
                            help="save results to disk")
        parser.add_argument(
            "-p", "--visualize-performance", default=False, action="store_true",
            help="visualize the interaction with the domain during assessment trials")
        parser.add_argument(
            "-s", "--visualize-steps", default=False, action="store_true",
            help="visualize steps during learning")
        parser.add_argument(
            "-l", "--visualize-learning", default=False, action="store_true",
            help="visualize learning progress")
        parser.add_argument(
            "-d", "--debug", default=False, action="store_true",
            help="enter pdb debugger when receiving SIGURG signal")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--path", type=str)
        args = parser.parse_args()
        if args.seed is not None and args.seed > 0:
            self.exp_id = args.seed
        if args.path is not None:
            self._update_path(args.path)
        self.run(visualize_performance=args.visualize_performance,
                 visualize_learning=args.visualize_learning,
                 visualize_steps=args.visualize_steps,
                 debug_on_sigurg=args.debug)
        if args.save:
            self.save()
        if args.show_plot:
            self.plot()

    def run(self, visualize_performance=0, visualize_learning=False,
            visualize_steps=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

        :param visualize_performance: (int)
            determines whether a visualization of the steps taken in
            performance runs are shown. 0 means no visualization is shown.
            A value n > 0 means that only the first n performance runs for a
            specific policy are shown (i.e., for n < checks_per_policy, not all
            performance runs are shown)
        :param visualize_learning: (boolean)
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        :param visualize_steps: (boolean)
            visualize all steps taken during learning
        :param debug_on_sigurg: (boolean)
            if true, the ipdb debugger is opened when the python process
            receives a SIGURG signal. This allows to enter a debugger at any
            time, e.g. to view data interactively or actual debugging.
            The feature works only in Unix systems. The signal can be sent
            with the kill command:

                kill -URG pid

            where pid is the process id of the python interpreter running this
            function.

        """

        if debug_on_sigurg:
            rlpy.Tools.ipshell.ipdb_on_SIGURG()
        self.performance_domain = deepcopy(self.domain)
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.exp_id
        total_steps = 0
        eps_steps = 0
        eps_return = 0
        episode_number = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.agent.representation)

        # Used to bound the number of logs in the file
        start_log_time = clock()
        # Used to show the total time took the process
        self.start_time = clock()
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        self.evaluate(total_steps, episode_number, visualize_performance)
        self.total_eval_time = 0.
        terminal = True
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                s, terminal, p_actions = self.domain.s0()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been
                # passed
                eps_return = 0
                eps_steps = 0
                episode_number += 1
            # Act,Step
            r, ns, terminal, np_actions = self.domain.step(a)

            self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps += 1
            eps_steps += 1
            eps_return += r

            # Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap) and deltaT(start_log_time) > self.log_interval:
                start_log_time = clock()
                elapsedTime = deltaT(self.start_time)
                self.logger.info(
                    self.log_template.format(total_steps=total_steps,
                                             elapsed=hhmmss(
                                                 elapsedTime),
                                             remaining=hhmmss(
                                                 elapsedTime * (
                                                     self.max_steps - total_steps) / total_steps),
                                             totreturn=eps_return,
                                             steps=eps_steps,
                                             num_feat=self.agent.representation.features_num))

            # learning
            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)

            # Check Performance
            if total_steps % (self.max_steps / self.num_policy_checks) == 0:
                self.elapsed_time = deltaT(
                    self.start_time) - self.total_eval_time

                # show policy or value function
                if visualize_learning:
                    self.domain.showLearning(self.agent.representation)

                self.evaluate(
                    total_steps,
                    episode_number,
                    visualize_performance)
                self.total_eval_time += deltaT(self.start_time) - \
                    self.elapsed_time - \
                    self.total_eval_time
                start_log_time = clock()

        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.info("Total Experiment Duration %s" % (hhmmss(deltaT(self.start_time))))

    def evaluate(self, total_steps, episode_number, visualize=0):
        """
        Evaluate the current agent within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """
        # TODO resolve this hack
        if className(self.agent) == 'PolicyEvaluation':
            # Policy Evaluation Case
            self.result = self.agent.STATS
            return

        random_state = np.random.get_state()
        #random_state_domain = copy(self.domain.random_state)
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        performance_discounted_return = 0.
        for j in xrange(self.checks_per_policy):
            p_ret, p_step, p_term, p_dret = self.performanceRun(
                total_steps, visualize=visualize > j)
            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
            performance_discounted_return += p_dret
        performance_return /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        performance_discounted_return /= self.checks_per_policy
        self.result["learning_steps"].append(total_steps)
        self.result["return"].append(performance_return)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.agent.representation.features_num)
        self.result["steps"].append(performance_steps)
        self.result["terminated"].append(performance_term)
        self.result["learning_episode"].append(episode_number)
        self.result["discounted_return"].append(performance_discounted_return)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        if total_steps > 0:
            remaining = hhmmss(
                elapsedTime * (self.max_steps - total_steps) / total_steps)
        else:
            remaining = "?"
        self.logger.info(
            self.performance_log_template.format(total_steps=total_steps,
                                                 elapsed=hhmmss(
                                                     elapsedTime),
                                                 remaining=remaining,
                                                 totreturn=performance_return,
                                                 steps=performance_steps,
                                                 num_feat=self.agent.representation.features_num))

        np.random.set_state(random_state)
        #self.domain.rand_state = random_state_domain

    def save(self):
        """Saves the experimental results to the ``results.json`` file
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)

    def load(self):
        """loads the experimental results from the ``results.txt`` file
        If the results could not be found, the function returns ``None``
        and the results array otherwise.
        """
        results_fn = os.path.join(self.full_path, self.output_filename)
        self.results = rlpy.Tools.results.load_single(results_fn)
        return self.results

    def plot(self, y="return", x="learning_steps", save=False):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider
        :py:class:`Tools.Merger.Merger`.
        """
        labels = rlpy.Tools.results.default_labels
        performance_fig = plt.figure("Performance")
        res = self.result
        plt.plot(res[x], res[y], '-bo', lw=3, markersize=10)
        plt.xlim(0, res[x][-1] * 1.01)
        y_arr = np.array(res[y])
        m = y_arr.min()
        M = y_arr.max()
        delta = M - m
        if delta > 0:
            plt.ylim(m - .1 * delta - .1, M + .1 * delta + .1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(
                self.full_path,
                "{:3}-performance.pdf".format(self.exp_id))
            performance_fig.savefig(path, transparent=True, pad_inches=.1)
        plt.ioff()
        plt.show()

    def compile_path(self, path):
        """
        An experiment path can be specified with placeholders. For
        example, ``Results/Temp/{domain}/{agent}/{representation}``.
        This functions replaces the placeholders with actual values.
        """
        variables = re.findall("{([^}]*)}", path)
        replacements = {}
        for v in variables:
            if lower(v).startswith('representation') or lower(v).startswith('policy'):
                obj = 'self.agent.' + v
            else:
                obj = 'self.' + v

            if len([x for x in ['self.domain', 'self.agent', 'self.agent.policy', 'self.agent.representation'] if x == lower(obj)]):
                replacements[v] = eval('className(%s)' % obj)
            else:
                try:
                    replacements[v] = str(eval('%s' % v))
                except:
                    print "Warning: Could not interpret path variable", repr(v)

        return path.format(**replacements)
