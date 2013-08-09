#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by A. Geramifard March 13th 2013 at MIT #
######################################################
# Classical Policy Iteration
# Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the state space
# Once the errors are bounded, the policy is changed
from MDPSolver import *
class PolicyIteration(MDPSolver):

    def __init__(self,job_id, representation,domain,logger, planning_time = inf, convergence_threshold = .005, ns_samples = 100, project_path = '.', log_interval = 5000, show = False, max_PE_iterations = 10):
        super(PolicyIteration,self).__init__(job_id, representation,domain,logger, planning_time, convergence_threshold, ns_samples, project_path,log_interval, show)
        self.max_PE_iterations = max_PE_iterations
        self.logger.log('Max PE Iterations:\t%d' % self.max_PE_iterations)
    def solve(self):
        self.result = []
        self.start_time     = clock() # Used to show the total time took the process

        # Check for Tabular Representation
        rep  = self.representation
        if className(rep) != 'Tabular':
            self.logger.log("Value Iteration works only with the tabular representation.")
            return 0

        no_of_states        = self.representation.agg_states_num
        bellmanUpdates      = 0

        #Initialize the policy
        policy              = eGreedy(deepcopy(self.representation),self.logger, epsilon = 0, forcedDeterministicAmongBestActions = True) # Copy the representation so that the weight change during the evaluation does not change the policy
        policyChanged       = True

        policy_improvement_iteration = 0
        while policyChanged and deltaT(self.start_time) < self.planning_time:

            # Policy Evaluation
            converged = False
            policy_evaluation_iteration = 0
            while not converged and self.hasTime() and policy_evaluation_iteration < self.max_PE_iterations:
                policy_evaluation_iteration += 1
                # Sweep The State Space
                for i in arange(0,no_of_states):
                    if not self.hasTime(): break
                    s = self.representation.stateID2state(i)
                    if not self.domain.isTerminal(s) and len(self.domain.possibleActions(s)):
                        self.BellmanBackup(s,policy.pi(s),self.ns_samples, policy)
                        bellmanUpdates += 1

                        if bellmanUpdates % self.log_interval == 0:
                            performance_return, _,_,_  = self.performanceRun()
                            self.logger.log('[%s]: BellmanUpdates=%d, Return=%0.4f' % (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))

                #check for convergence
                theta_change = linalg.norm(policy.representation.theta - self.representation.theta,inf)
                converged = theta_change < self.convergence_threshold
                self.logger.log('PE #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f' % (policy_evaluation_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change))
                if self.show: self.domain.show(s,policy.pi(s),self.representation)

            #Policy Improvement:
            policy_improvement_iteration += 1
            new_policy = zeros(no_of_states)
            policyChanged = 0
            i = 0
            while i < no_of_states and self.hasTime():
                s = self.representation.stateID2state(i)
                if not self.domain.isTerminal(s) and len(self.domain.possibleActions(s)):
                    for a in self.domain.possibleActions(s):
                        if not self.hasTime(): break
                        self.BellmanBackup(s,a,self.ns_samples, policy)
                    if policy.pi(s) != self.representation.bestAction(s): policyChanged += 1
                i += 1
            policy.representation.theta = self.representation.theta.copy() # This will cause the policy to be copied over
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, Policy Change=%d, Return=%0.4f, Steps=%d' % (policy_improvement_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, policyChanged, performance_return,performance_steps))

            # store stats
            self.result.append([bellmanUpdates, # index = 0
                               performance_return, # index = 1
                               deltaT(self.start_time), # index = 2
                               self.representation.features_num, # index = 3
                               performance_steps,# index = 4
                               performance_term, # index = 5
                               performance_discounted_return, # index = 6
                               policy_improvement_iteration #index = 7
                            ])

        if converged: self.logger.log('Converged!')
        super(PolicyIteration,self).solve()


