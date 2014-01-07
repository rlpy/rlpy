from .Experiment import Experiment
from Tools import Logger
import numpy as np
from collections import defaultdict
from Tools import clock, deltaT, hhmmss

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class PolicyEvaluationExperiment(Experiment):
    """
    Experiment Class for running policy evaluation experiments, that is estimating the state-value function
    w.r.t. a given policy while observing transitions.
    """

    log_template = '{total_steps: >6}: E[{elapsed}]-R[{remaining}]: Features = {num_feat}'
    performance_log_template = '{total_steps: >6}: E[{elapsed}]-R[{remaining}]: RMSE={rmse: >10.4g}, RMSBE={rmsbe: >8.4g}, |TD-Error|={tderr: >8.4g} Features = {num_feat}'
    num_eval_points_per_dim = 41

    def __init__(self, estimator, domain, sample_policy, target_policy=None, logger=None, id=1, max_steps=10000,
                 num_checks=10, log_interval=1, path="Results/Temp", **kwargs):

        self.id = id
        assert id > 0
        self.estimator = estimator
        self.sample_policy = sample_policy
        if target_policy is None:
            target_policy = sample_policy
        self.target_policy = sample_policy
        self.domain = domain
        self.max_steps = max_steps
        self.num_checks = num_checks
        self.logger = logger if logger is not None else Logger()
        self.log_interval = log_interval
        self._update_path(path)


    def seed_components(self):
        """
        set the initial seeds for all random number generators used during
        the experiment run based on the currently set ``id``.
        """
        self.output_filename = '{:0>3}-results.json'.format(self.id)
        np.random.seed(self.randomSeeds[self.id - 1])
        self.domain.random_state = np.random.RandomState(self.randomSeeds[self.id - 1])

    def _estimate_mpsbe_fp_for_partitioning(self, partitioning, num_parts):
        # P, R, d = self.domain.model_for_partitioning(self.target_policy, partitioning, num_parts, num_traj=10000)
        # value at each part of the partitioning
        #import ipdb; ipdb.set_trace()
        #P = P[:-1, :-1]
        #R = R[:-1]
        #v = np.linalg.solve(np.eye(num_parts - 1) - self.domain.gamma * P, R)

        #def mspbe_fp(s, terminal=False):
        #    i = partitioning(s, terminal)
        #    if i == num_parts:
        #        return 0.
        #    return v[i]

        #return mspbe_fp
        num_samples = 200000
        S, S_term, Sn, Sn_term, R = self.domain.transition_samples(self.target_policy, num_samples=num_samples)
        Phi = np.zeros((num_samples, num_parts - 1))
        Phi_n = np.zeros((num_samples, num_parts - 1))
        for i in xrange(num_samples):
            idx = partitioning(S[i], S_term[i])
            if idx != num_parts - 1:
                Phi[i, idx] = 1.
            idx = partitioning(Sn[i], Sn_term[i])
            if idx != num_parts - 1:
                Phi_n[i, idx] = 1.
        del S
        del Sn
        del S_term
        del Sn_term
        A = np.dot(Phi.T, Phi - self.domain.gamma * Phi_n)
        del Phi_n
        b = np.dot(Phi.T, R)
        del Phi
        theta,_,_,_ = np.linalg.lstsq(A, b)
        def mspbe_fp(s, terminal=False):
            i = partitioning(s, terminal)
            if i == num_parts - 1:
                return 0.
            return theta[i]

        return mspbe_fp


    def _evaluate_RMSBE(self, V):
        P, R, _, _ = self.domain.chain_model(self.target_policy)
        delta = V - np.dot(P, V) * self.domain.gamma - R
        return np.sqrt(np.sum(delta**2))

    def _evaluate_matderror(self, f, num_samples=200000):
        S, S_term, Sn, Sn_term, R = self.domain.transition_samples(self.target_policy, num_samples=num_samples)
        delta = 0.
        for i in xrange(num_samples):
            delta += np.abs(f(S[i], S_term[i]) - self.domain.gamma * f(Sn[i], Sn_term[i]) - R[i])
        delta /= num_samples
        return delta

    def evaluate(self, total_steps, episode_number):
        """
        Evaluate the current estimator within an experiment

        :param total_steps: (int)
                     number of steps used in learning so far
        :param episode_number: (int)
                        number of episodes used in learning so far
        """

        random_state = np.random.get_state()
        #random_state_domain = copy(self.domain.random_state)
        elapsedTime = deltaT(self.start_time)
        V_true, s_test, s_term, s_distr = self.domain.V(self.target_policy, discretization=self.num_eval_points_per_dim, num_traj=2000, max_len_traj=200)
        V_pred = np.zeros_like(V_true)
        for i in xrange(s_test.shape[1]):
            V_pred[i] = self.estimator.predict(s_test[:, i], s_term[i])
        rmse = np.sqrt(((V_true - V_pred)**2 * s_distr).sum() / s_distr.sum())
        matderror = self._evaluate_matderror(self.estimator.predict, num_samples=10000)
        #matderror = self._evaluate_matderror(predict_V_true)
        #def predict_V_true(s, t):
        #    for i in xrange(s_test.shape[1]):
        #        if np.all(s == s_test[:,i]):
        #            return V_true[i]

        if hasattr(self.domain, "chain_model"):
            rmsbe = self._evaluate_RMSBE(V_pred)
        else:
            # we do not compute MSBE for domains without a model
            rmsbe = -1.
        self.result["learning_steps"].append(total_steps)
        self.result["rmse"].append(rmse)
        self.result["matderror"].append(matderror)
        self.result["learning_time"].append(self.elapsed_time)
        self.result["num_features"].append(self.estimator.representation.features_num)
        self.result["learning_episode"].append(episode_number)
        # reset start time such that performanceRuns don't count
        self.start_time = clock() - elapsedTime
        if total_steps > 0:
            remaining = hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps)
        else:
            remaining = "?"
        self.logger.log(self.performance_log_template.format(total_steps=total_steps,
                                                             elapsed=hhmmss(elapsedTime),
                                                             remaining=remaining,
                                                             rmsbe=rmsbe,
                                                             rmse=rmse, tderr=matderror,
                                                             num_feat=self.estimator.representation.features_num))

        np.random.set_state(random_state)
        #self.domain.rand_state = random_state_domain

    def run(self,visualize_learning=False, visualize_steps=False, debug_on_sigurg=False):
        """
        Run the experiment and collect statistics / generate the results

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
            Tools.ipshell.ipdb_on_SIGURG()
        self.seed_components()

        self.result = defaultdict(list)
        self.result["seed"] = self.id
        total_steps         = 0
        eps_steps           = 0
        eps_return          = 0
        episode_number      = 0
        # show policy or value function of initial policy
        if visualize_learning:
            self.domain.showLearning(self.estimator.representation)

        start_log_time      = clock()  # Used to bound the number of logs in the file
        self.start_time     = clock()  # Used to show the total time took the process
        self.elapsed_time = 0
        # do a first evaluation to get the quality of the inital policy
        self.evaluate(total_steps, episode_number)
        self.total_eval_time = 0.
        terminal = True
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                self.estimator.episodeTerminated()
                s, terminal, p_actions = self.domain.s0()
                a = self.sample_policy.pi(s, terminal, p_actions)
                # Visual
                if visualize_steps:
                    self.domain.show(a, self.estimator.representation)

                # Output the current status if certain amount of time has been passed
                eps_return      = 0
                eps_steps       = 0
                episode_number += 1
            # Act,Step
            r, ns, terminal, np_actions   = self.domain.step(a)
            self._gather_transition_statistics(s, a, ns, r, learning=True)
            na = self.sample_policy.pi(ns, terminal, np_actions)
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
                                                         num_feat=self.estimator.representation.features_num))

            # learning
            self.estimator.learn(s, a, r, ns, terminal)
            s, a, p_actions = ns, na, np_actions
            # Visual
            if visualize_steps:
                self.domain.show(a, self.estimator.representation)

            # Check Performance
            if total_steps % (self.max_steps / self.num_checks) == 0:
                self.elapsed_time = deltaT(self.start_time) - self.total_eval_time
                #import ipdb; ipdb.set_trace()
                # show policy or value function
                if visualize_learning:
                    self.domain.showLearning(self.estimator.representation)

                self.evaluate(total_steps, episode_number)
                self.total_eval_time += deltaT(self.start_time) - self.elapsed_time - self.total_eval_time
                start_log_time = clock()

        # Visual
        if visualize_steps:
            self.domain.show(a, self.estimator.representation)
        self.logger.line()
        self.logger.log("Took %s\n" % (hhmmss(deltaT(self.start_time))))



