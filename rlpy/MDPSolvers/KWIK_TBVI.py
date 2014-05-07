"""Knows What it Knows style Trajectory Based Value Iteration"""

from .TrajectoryBasedValueIteration import TrajectoryBasedValueIteration
import numpy as np
from rlpy.Tools import clock, randSet, hhmmss, deltaT, findElemArray1D
from rlpy.Tools import hasFunction
__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class KWIK_TBVI(TrajectoryBasedValueIteration):

    """Trajectory Based Value Iteration. This algorithm is different from Value iteration in 2 senses:
    1. It works with any Linear Function approximator
    2. Samples are gathered using the e-greedy policy

    The algorithm terminates if the maximum bellman-error in a consequent set of trajectories is below a threshold
    Based on the KWIK Learning paper of Tom Walsh UAI 2009
    """

    DEBUG = True
    KWIK_Q = None
    KWIK_threshold = None

    def __init__(
            self, job_id, representation, domain, planning_time=np.inf,
            convergence_threshold=.005, ns_samples=100, project_path='.',
            log_interval=500, show=False, epsilon=.1, KWIK_threshold=.1):
        super(
            KWIK_TBVI,
            self).__init__(job_id,
                           representation,
                           domain,
                           planning_time,
                           convergence_threshold,
                           ns_samples,
                           project_path,
                           log_interval,
                           show)
        self.KWIK_threshold = KWIK_threshold
        self.KWIK_Q = np.eye(self.representation.features_num)
        self.epsilon = epsilon

    def solve(self):
        self.result = []
        # Used to show the total time took the process
        self.start_time = clock()
        bellmanUpdates = 0
        converged = False
        iteration = 0
        # Track the number of consequent trajectories with very small observed
        # BellmanError
        converged_trajectories = 0
        while self.hasTime() and not converged:

            # Generate a new episode e-greedy with the current values
            max_Bellman_Error = 0
            step = 0
            s, terminal, p_actions = self.domain.s0()

            # The action is always greedy w.r.t value function unless it is not
            # Known based on KWIK which means it should have Vmax
            Q, a = self.bestKWIKAction(
                s, terminal, p_actions)

            while not terminal and step < self.domain.episodeCap and self.hasTime():
                new_Q = self.Q_KWIKoneStepLookAhead(
                    s,
                    a,
                    self.ns_samples)
                phi_s = self.representation.phi(s, terminal)
                phi_s_a = self.representation.phi_sa(
                    s,
                    terminal,
                    a,
                    phi_s=phi_s)
                old_Q = np.dot(phi_s_a, self.representation.theta)
                bellman_error = new_Q - old_Q
                # print s, old_Q, new_Q, bellman_error
                self.representation.theta   += self.alpha * \
                    bellman_error * phi_s_a
                self.KWIK_update(s, a, new_Q)
                bellmanUpdates += 1
                step += 1

                # Discover features if the representation has the discover
                # method
                if hasattr(self.representation, "discover"):
                    self.representation.discover(phi_s, bellman_error)

                max_Bellman_Error = max(max_Bellman_Error, abs(bellman_error))
                # Simulate new state and action on trajectory
                _, s, terminal, p_actions = self.domain.step(a)
                if np.random.rand() > self.epsilon:
                    a = self.representation.bestAction(s, terminal, p_actions)
                else:
                    a = randSet(self.domain.possibleActions())

            # check for convergence
            iteration += 1
            if max_Bellman_Error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun(
            )
            converged = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES
            self.logger.info(
                'PI #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Return=%0.4f, Steps=%d, Features=%d' % (iteration,
                                                                                                                  hhmmss(
                                                                                                                      deltaT(
                                                                                                                          self.start_time)),
                                                                                                                  bellmanUpdates,
                                                                                                                  max_Bellman_Error,
                                                                                                                  performance_return,
                                                                                                                  performance_steps,
                                                                                                                  self.representation.features_num))
            if self.show:
                self.domain.show(a, self.representation)

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
            self.logger.info('Converged!')
        super(KWIK_TBVI, self).solve()

    def bestKWIKAction(self, s, terminal, p_actions):
        # Return the best action based on the kwik learner. If the state-action
        # is not known it will be among the pool of the best actions because it
        # has V_max value
        phi_s = self.representation.phi(s, terminal)
        Qs = self.representation.Qs(s, terminal, phi_s=phi_s)
        Qs = Qs[p_actions]

        for i, a in enumerate(p_actions):
            if self.KWIK_predict(s, a) is None:
                Qs[i] = self.domain.RMAX / (1 - self.domain.gamma)

        max_ind = findElemArray1D(Qs, Qs.max())

        self.logger.debug('State:' + str(s))
        for i in xrange(len(p_actions)):
            self.logger.debug('Action %d, Q = %0.3f' % (p_actions[i], Qs[i]))
        self.logger.debug(
            'Best: %s, Max: %s' %
            (str(p_actions[max_ind]), str(Qs.max())))
        bestA = p_actions[max_ind]
        bestQ = Qs[max_ind]
        if len(bestA) > 1:
            final_A = randSet(bestA)
        else:
            final_A = bestA[0]

        return bestQ[0], final_A

    def KWIK_V(self, s, terminal, p_actions):
        return self.bestKWIKAction(s, terminal, p_actions)[0]

    def Q_KWIKoneStepLookAhead(self, s, a, ns_samples):
        # Hash new state for the incremental tabular case
        self.continuous_state_starting_samples = 10
        if hasFunction(self, 'addState'):
            self.addState(s)

        gamma = self.domain.gamma
        if hasFunction(self.domain, 'expectedStep'):
            p, r, ns, t, p_actions = self.domain.expectedStep(s, a)
            Q = 0
            for j in xrange(len(p)):
                    Q += p[j, 0] * (r[j, 0] + gamma * self.KWIK_V(ns[j, :], t[j,:], p_actions[j]))
        else:
            # See if they are in cache:
            key = tuple(np.hstack((s, [a])))
            cacheHit = self.expectedStepCached.get(key)
            if cacheHit is None:
# Not found in cache => Calculate and store in cache
                # If continuous domain, sample <continuous_state_starting_samples> points within each discretized grid and sample <ns_samples>/<continuous_state_starting_samples> for each starting state.
                # Otherwise take <ns_samples> for the state.

                # First put s in the middle of the grid:
                # shout(self,s)
                s = self.stateInTheMiddleOfGrid(s)
                # print "After:", shout(self,s)
                if len(self.domain.continuous_dims):
                    next_states = np.empty(
                        (ns_samples, self.domain.state_space_dims))
                    rewards = np.empty(ns_samples)
                    # next states per samples initial state
                    ns_samples_ = ns_samples / \
                        self.continuous_state_starting_samples
                    for i in xrange(self.continuous_state_starting_samples):
                        # sample a random state within the grid corresponding
                        # to input s
                        new_s = s.copy()
                        for d in xrange(self.domain.state_space_dims):
                            w = self.binWidth_per_dim[d]
                            # Sample each dimension of the new_s within the
                            # cell
                            new_s[d] = (np.random.rand() - .5) * w + s[d]
                            # If the dimension is discrete make make the
                            # sampled value to be int
                            if not d in self.domain.continuous_dims:
                                new_s[d] = int(new_s[d])
                        # print new_s
                        ns, r = self.domain.sampleStep(new_s, a, ns_samples_)
                        next_states[i * ns_samples_:(i + 1) * ns_samples_, :] = ns
                        rewards[i * ns_samples_:(i + 1) * ns_samples_] = r
                else:
                    next_states, rewards = self.domain.sampleStep(
                        s, a, ns_samples)
                self.expectedStepCached[key] = [next_states, rewards]
            else:
                # print "USED CACHED"
                next_states, rewards = cacheHit
                Q = np.mean([rewards[i] + gamma * self.KWIK_V(next_states[i, :]) for i in xrange(ns_samples)])
        return Q

    def KWIK_update(self, s, a, KWIK_V):
        # The KWIK Learning algorithm here
        pass

    def KWIK_predict(self, s, a):
        # return the value of a state-action pair. If it is not known then it
        # will return None
        pass
