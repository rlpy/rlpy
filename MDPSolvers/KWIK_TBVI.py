#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by A. Geramifard March 14th 2013 at MIT #
######################################################


from TrajectoryBasedValueIteration import *
class KWIK_TBVI(TrajectoryBasedValueIteration):
    """Trajectory Based Value Iteration. This algorithm is different from Value iteration in 2 senses:
    1. It works with any Linear Function approximator
    2. Samples are gathered using the e-greedy policy

    The algorithm terminates if the maximum bellman-error in a consequent set of trajectories is below a threshold
    Based on the KWIK Learning paper of Tom Walsh UAI 2009
    """

    DEBUG           = True
    KWIK_Q          = None
    KWIK_threshold  = None

    def __init__(self,job_id, representation,domain,logger, planning_time = inf,
                 convergence_threshold = .005, ns_samples = 100, project_path = '.',
                 log_interval = 500, show = False, epsilon = .1, KWIK_threshold = .1):
        super(KWIK_TBVI,self).__init__(job_id, representation,domain,logger, planning_time, convergence_threshold, ns_samples, project_path,log_interval, show)
        self.KWIK_threshold = KWIK_threshold
        self.KWIK_Q = identity(self.representation.features_num)
        self.epsilon = epsilon
        self.logger.log('KWIK Threhsold:\t\t\t%0.2f' % self.KWIK_threshold)

    def solve(self):
        self.result = []
        self.start_time     = clock() # Used to show the total time took the process
        bellmanUpdates      = 0
        converged           = False
        iteration           = 0
        converged_trajectories  = 0 # Track the number of consequent trajectories with very small observed BellmanError
        while self.hasTime() and not converged:

            # Generate a new episode e-greedy with the current values
            max_Bellman_Error       = 0
            step                    = 0
            terminal                = False
            s                       = self.domain.s0()
            # The action is always greedy w.r.t value function unless it is not Known based on KWIK which means it should have Vmax
            Q,a                     = self.bestKWIKAction(s)
            while not terminal and step < self.domain.episodeCap and self.hasTime():
                new_Q           = self.representation.Q_KWIKoneStepLookAhead(s,a, self.ns_samples)
                phi_s           = self.representation.phi(s)
                phi_s_a         = self.representation.phi_sa(s,a,phi_s)
                old_Q           = dot(phi_s_a,self.representation.theta)
                bellman_error   = new_Q - old_Q
                #print s, old_Q, new_Q, bellman_error
                self.representation.theta   += self.alpha * bellman_error * phi_s_a
                self.KWIK_update(s, a, new_Q)
                bellmanUpdates              += 1
                step                        += 1

                #Discover features if the representation has the discover method
                self.representation.discover(phi_s,bellman_error)

                max_Bellman_Error = max(max_Bellman_Error,abs(bellman_error))
                #Simulate new state and action on trajectory
                _,s,terminal    = self.domain.step(a)
                a               = self.representation.bestAction() if random.rand() > self.epsilon else randSet(self.domain.possibleActions())

            #check for convergence
            iteration += 1
            if max_Bellman_Error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            converged = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Return=%0.4f, Steps=%d, Features=%d' % (iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, max_Bellman_Error, performance_return,performance_steps,self.representation.features_num))
            if self.show:  self.domain.show(a,self.representation)

            # store stats
            self.result.append([bellmanUpdates, # index = 0
                               performance_return, # index = 1
                               deltaT(self.start_time), # index = 2
                               self.representation.features_num, # index = 3
                               performance_steps,# index = 4
                               performance_term, # index = 5
                               performance_discounted_return, # index = 6
                               iteration #index = 7
                            ])

        if converged: self.logger.log('Converged!')
        super(TrajectoryBasedValueIteration,self).solve()
    def bestKWIKAction(self,s):
        # Return the best action based on the kwik learner. If the state-action is not known it will be among the pool of the best actions because it has V_max value
        phi_s                   = self.representation.phi(s)
        Qs, A                   = self.representation.Qs(s,phi_s)
        for a,i in enumerate(A):
                if self.KWIK_predict(s,a) is None:
                    Qs[i] = self.domain.RMAX / (1-self.domain.gamma)

        max_ind = findElemArray1D(Qs,Qs.max())
        if self.DEBUG:
            self.logger.log('State:' +str(s))
            self.logger.line()
            for i in arange(len(A)):
                self.logger.log('Action %d, Q = %0.3f' % (A[i], Qs[i]))
            self.logger.line()
            self.logger.log('Best: %s, Max: %s' % (str(A[max_ind]),str(Qs.max())))
            #raw_input()
        bestA = A[max_ind]
        bestQ = Q[max_ind]
        if len(bestA) > 1:
            final_A = randSet(bestA)
        else:
            final_A = bestA[0]

        return bestQ[0], final_A
    def KWIK_V(self,s):
        return self.bestKWIKAction(s)[0]
    def Q_KWIKoneStepLookAhead(self,s,a, ns_samples):
        # Hash new state for the incremental tabular case
        self.continuous_state_starting_samples = 10
        if hasFunction(self,'addState'): self.addState(s)

        gamma   = self.domain.gamma
        if hasFunction(self.domain,'expectedStep'):
            p,r,ns,t    = self.domain.expectedStep(s,a)
            Q           = 0
            for j in arange(len(p)):
                    Q += p[j,0]*(r[j,0] + gamma*self.KWIK_V(ns[j,:]))
        else:
            # See if they are in cache:
            key = tuple(hstack((s,[a])))
            cacheHit     = self.expectedStepCached.get(key)
            if cacheHit is None:
#               # Not found in cache => Calculate and store in cache
                # If continuous domain, sample <continuous_state_starting_samples> points within each discretized grid and sample <ns_samples>/<continuous_state_starting_samples> for each starting state.
                # Otherwise take <ns_samples> for the state.

                #First put s in the middle of the grid:
                #shout(self,s)
                s = self.stateInTheMiddleOfGrid(s)
                #print "After:", shout(self,s)
                if len(self.domain.continuous_dims):
                    next_states = empty((ns_samples,self.domain.state_space_dims))
                    rewards         = empty(ns_samples)
                    ns_samples_ = ns_samples/self.continuous_state_starting_samples # next states per samples initial state
                    for i in arange(self.continuous_state_starting_samples):
                        #sample a random state within the grid corresponding to input s
                        new_s = s.copy()
                        for d in arange(self.domain.state_space_dims):
                            w = self.binWidth_per_dim[d]
                            # Sample each dimension of the new_s within the cell
                            new_s[d] = (random.rand()-.5)*w+s[d]
                            # If the dimension is discrete make make the sampled value to be int
                            if not d in self.domain.continuous_dims:
                                new_s[d] = int(new_s[d])
                        #print new_s
                        ns,r = self.domain.sampleStep(new_s,a,ns_samples_)
                        next_states[i*ns_samples_:(i+1)*ns_samples_,:] = ns
                        rewards[i*ns_samples_:(i+1)*ns_samples_] = r
                else:
                    next_states,rewards = self.domain.sampleStep(s,a,ns_samples)
                self.expectedStepCached[key] = [next_states, rewards]
            else:
                #print "USED CACHED"
                next_states, rewards = cacheHit
                Q = mean([rewards[i] + gamma*self.KWIK_V(next_states[i,:]) for i in arange(ns_samples)])
        return Q
    def KWIK_update(self,s,a,KWIK_V):
        # The KWIK Learning algorithm here
        pass
    def KWIK_predict(self,s,a):
        #return the value of a state-action pair. If it is not known then it will return None
        pass


