"""Standard Experiment for Learning Control in RL"""
from Tools import *
import numpy as np
from copy import copy, deepcopy
import re
import Tools.ipshell
import argparse
import json
import Tools.results
from collections import defaultdict

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Experiment(object):
    """
    The Experiment controls the training, testing, and evaluation of the
    agent. Reinforcement learning is based around
    the concept of training an Agent to solve a task and later testing its
    ability to use what it has learned to solve the task.
    This cycle forms a loop that the experiment defines and controls. First
    the agent is repeatedly tasked with solving a problem determined by the
    domain, restarting after some termination condition is reached
    (The sequence of steps between terminations is known as an *episode*).

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

    The :ref:`Experiment <experiment>` class is a superclass that provides
    the basic framework for all RL experiments. It provides the methods and
    attributes that allow child classes to interact with the :ref:`Agent <agent>`
    and :ref:`Domain <domain>` classes within the RLPy library.

    All experiment implementations should inherit from :ref:`Experiment <experiment>`.
    """

    #: The Main Random Seed used to generate other random seeds
    mainSeed = 999999999
    #: Maximum number of runs used for averaging, specified so that enough
    #: random seeds are generated
    maxRuns = 1000
    #: Array of random seeds. This is used to make sure all jobs start with
    #: the same random seed
    randomSeeds = np.random.RandomState(mainSeed).randint(1, mainSeed, maxRuns)

    #: ID of the current experiment running
    id = 1

    #: The domain to be tested on
    domain = None
    #: The agent to be tested
    agent = None

    #: An array that stores all generated results. The purpose of a run
    #: is to fill this array. Size is stats_num x num_policy_checks.
    result = None
    #: The name of the file used to store the data
    output_filename = ''
    #: A simple object that records the prints in a file
    logger = None

    max_steps = 0  #: Total number of interactions
    #: Number of Performance Checks uniformly scattered along the trajectory
    num_policy_checks = 0
    log_interval = 0  #: Number of seconds between log prints

    # TODO make sure these are actually used
    STEP = 0  # Learning Steps
    RETURN              = 1         # Sum of Rewards
    CLOCK_TIME          = 2         # Time in seconds so far
    FEATURE_SIZE        = 3         # Number of features used for value function representation
    EPISODE_LENGTH      = 4
    TERMINAL            = 5         # 0 = No Terminal, 1 = Normal Terminal, 2 = Critical Terminal
    EPISODE_NUMBER      = 6
    DISCOUNTED_RETURN   = 7
    STATS_NUM = 8  #: Number of statistics to be saved

    log_template = '{total_steps: >6}: E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'
    performance_log_template ='{total_steps: >6}: >>> E[{elapsed}]-R[{remaining}]: Return={totreturn: >10.4g}, Steps={steps: >4}, Features = {num_feat}'

    def __init__(self, agent, domain, logger, id=1, max_steps=max_steps,
                 num_policy_checks=10, log_interval=1, path='Results/Temp',
                 checks_per_policy=1, **kwargs):
        self.id = id
        assert id > 0
        self.agent = agent
        #: defines how many episodes should be run to estimate the performance
        #: of a single policy
        self.checks_per_policy = checks_per_policy
        self.domain = domain
        self.max_steps = max_steps
        self.num_policy_checks = num_policy_checks
        self.logger = logger
        self.log_interval = log_interval
        self.logger.line()
        self.logger.log("Experiment:\t\t%s" % className(self))
        self._update_path(path)

    def _update_path(self, path):

        # compile and create output path
        self.full_path = self.compile_path(path)
        checkNCreateDirectory(self.full_path + '/')
        self.logger.log("Output:\t\t\t%s/%s" % (self.full_path, self.output_filename))
        self.logger.setOutput("%s/%d-out.txt" % (self.full_path, self.id))

    def seed_components(self):
        """
        set the initial seeds for all random number generators used during
        the experiment run
        based on the currently set id
        """
        self.output_filename = '{:0>3}-results.json'.format(self.id)
        np.random.seed(self.randomSeeds[self.id - 1])
        self.domain.rand_state = np.random.RandomState(self.randomSeeds[self.id - 1])
        # make sure the performance_domain has a different seed
        self.performance_domain.rand_state = np.random.RandomState(self.randomSeeds[self.id + 20])

    def performanceRun(self, total_steps, visualize=False):
        """
        Execute a single episode using the current policy to evaluate its
        performance. No exploration or learning is enabled.

        Parameters
        ----------

        total_steps: int
            maximum number of steps of the episode to peform
        visualize: boolean, optional (default: False)
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
                plt.title('After ' + str(total_steps) + ' Steps')

            r, ns, eps_term, p_actions = self.performance_domain.step(a)
            # self.logger.log("TEST"+str(eps_length)+"."+str(s)+"("+str(a)+")"+"=>"+str(ns))
            s = ns
            eps_return += r
            eps_discount_return += self.performance_domain.gamma ** eps_length * r
            eps_length += 1
        if visualize:
            self.performance_domain.showDomain(a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying to accomodate them as an MDP

        return eps_return, eps_length, eps_term, eps_discount_return

    def printAll(self):
        """
        prints all information about the experiment
        """
        printClass(self)

    def run_from_commandline(self):
        parser = argparse.ArgumentParser("Run rlpy experiments")
        parser.add_argument("-v", "--show-plot", default=False, action="store_true",
                             help="show a plot at the end with the results of the assessment runs")
        parser.add_argument("-w", "--save", default=False, action="store_true",
                             help="save results to disk")
        parser.add_argument("-p", "--visualize-performance", default=False, action="store_true",
                             help="visualize the interaction with the domain during assessment trials")
        parser.add_argument("-s", "--visualize-steps",default=False, action="store_true",
                             help="visualize steps during learning")
        parser.add_argument("-l", "--visualize-learning",default=False, action="store_true",
                             help="visualize learning progress")
        parser.add_argument("-d", "--debug",default=False, action="store_true",
                            help="enter pdb debugger when receiving SIGURG signal")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--path", type=str)
        args = parser.parse_args()
        if args.seed is not None and args.seed > 0:
            self.id = args.seed
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


    def run(self, visualize_performance=False, visualize_learning=False,
            visualize_steps=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results


        visualize_performance (boolean):
            show a visualization of the steps taken in performance runs
        visualize_learning (boolean):
            show some visualization of the learning status before each
            performance evaluation (e.g. Value function)
        visualize_steps (boolean):
            visualize all steps taken during learning
        debug_on_sigurg (boolean):
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
            Tools.ipshell.ipdb_on_SIGURG()
        self.performance_domain = deepcopy(self.domain)
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.id
        total_steps         = 0
        eps_steps           = 0
        eps_return          = 0
        episode_number      = 0

        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.agent.representation)
        # do a first evaluation to get the quality of the inital policy
        self.evaluate(total_steps, episode_number, visualize_performance)

        start_log_time      = clock()  # Used to bound the number of logs in the file
        self.start_time     = clock()  # Used to show the total time took the process
        self.total_eval_time = 0.
        terminal = True
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                s, terminal, p_actions = self.domain.s0()
                a = self.agent.policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.agent.representation)

                # Output the current status if certain amount of time has been passed
                eps_return      = 0
                eps_steps       = 0
                episode_number += 1
            # Act,Step
            r, ns, terminal, np_actions   = self.domain.step(a)
            na              = self.agent.policy.pi(ns, terminal, np_actions)

            total_steps     += 1
            eps_steps       += 1
            eps_return      += r

            # Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap) and deltaT(start_log_time) > self.log_interval:
                start_log_time  = clock()
                elapsedTime     = deltaT(self.start_time)
                self.logger.log(self.log_template.format(total_steps=total_steps,
                                                         elapsed=hhmmss(elapsedTime),
                                                         remaining=hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps),
                                                         totreturn=eps_return,
                                                         steps=eps_steps,
                                                         num_feat=self.agent.representation.features_num))

            # learning
            self.agent.learn(s, p_actions, a, r, ns, np_actions, na, terminal)
            s, a, p_actions          = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.agent.representation)

            # Check Performance
            if total_steps % (self.max_steps / self.num_policy_checks) == 0:
                self.elapsed_time = deltaT(self.start_time) - self.total_eval_time

                # show policy or value function
                if visualize_learning:
                    self.domain.showLearning(self.agent.representation)

                self.evaluate(total_steps, episode_number, visualize_performance)
                self.total_eval_time += deltaT(self.start_time) - self.elapsed_time - self.total_eval_time
                start_log_time = clock()

        # Visual
        if visualize_steps:
            self.domain.show(a, self.agent.representation)
        self.logger.line()
        self.logger.log("Took %s\n" % (hhmmss(deltaT(self.start_time))))

    def evaluate(self, total_steps, episode_number, visualize=False):
        """
        Evaluate the current agent within an experiment

        Parameters
        ----------

        total_steps: int
                     number of steps used in learning so far
        episode_number: int
                        number of episodes used in learning so far
        """
        # TODO resolve this hack
        if className(self.agent) == 'PolicyEvaluation':
            # Policy Evaluation Case
            self.result = self.agent.STATS
            return

        random_state = np.random.get_state()
        random_state_domain = copy(self.domain.rand_state)
        elapsedTime = deltaT(self.start_time)
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        performance_discounted_return = 0.
        for j in xrange(self.checks_per_policy):
            p_ret, p_step, p_term, p_dret = self.performanceRun(total_steps, visualize=visualize)
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
        self.logger.log(self.performance_log_template.format(total_steps=total_steps,
                                                             elapsed=hhmmss(elapsedTime),
                                                             remaining=hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps),
                                                             totreturn=performance_return,
                                                             steps=performance_steps,
                                                             num_feat=self.agent.representation.features_num))

        random.set_state(random_state)
        self.domain.rand_state = random_state_domain

    def save(self):
        """Saves the experimental results to the results.json file
        """
        results_fn = os.path.join(self.full_path,self.output_filename)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        with open(results_fn, "w") as f:
            json.dump(self.result, f)

    def load(self):
        """loads the experimental results from the results.txt file
        If the results could not be found, the function returns None
        and the results array otherwise.
        """
        results_fn = os.path.join(self.full_path,self.output_filename)
        self.results = Tools.results.load_single(results_fn)
        return self.results

    def plot(self, y="return", x="learning_steps", save=False):
        """Plots the performance of the experiment
        This function has only limited capabilities.
        For more advanced plotting of results consider Tools.Merger
        """
        labels = Tools.results.default_labels
        performance_fig = pl.figure("Performance")
        res = self.result
        plt.plot(res[x], res[y], '-bo', lw=3, markersize=10)
        plt.xlim(0, self.result[0, -1]*1.01)
        y_arr = np.array(rey[y])
        m = y_arr.min()
        M = y_arr.max()
        delta = M-m
        if delta > 0:
            plt.ylim(m-.1*delta-.1, M+.1*delta+.1)
        xlabel = labels[x] if x in labels else x
        ylabel = labels[y] if y in labels else y
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        if save:
            path = os.path.join(self.full_path, "{:3}-performance.pdf".format(self.id))
            performance_fig.savefig(path, transparent=True, pad_inches=.1)
        plt.ioff()
        plt.show()

    def compile_path(self, path):
        """
        An experiment path can be specified with placeholders. For example:
            ``Results/Temp/{domain}/{agent}/{representation}``
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
