######################################################
# Developed by A. Geramifard March 13th 2013 at MIT #
######################################################
# Classical Policy Iteration
# Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the state space
# Once the errors are bounded, the policy is changed  
from MDPSolver import *
class PolicyIteration(MDPSolver):
    def solve(self):
        self.result = []
        self.start_time     = time() # Used to show the total time took the process
        
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
            while not converged and deltaT(self.start_time) < self.planning_time:
                policy_evaluation_iteration += 1
                prev_theta = self.representation.theta.copy()
                # Sweep The State Space
                for i in arange(0,no_of_states):
                    s = array(id2vec(i,rep.bins_per_dim))*self.representation.binWidth_per_dim
                    s += self.domain.statespace_limits[:,0] +.5
                    s[self.domain.continuous_dims] -= .5
                    self.BellmanBackup(s,policy.pi(s),self.ns_samples, policy)                        
                    bellmanUpdates += 1
                
                #check for convergence
                theta_change = linalg.norm(prev_theta - self.representation.theta,inf)
                converged = theta_change < self.convergence_threshold        
                self.logger.log('PE #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f' % (policy_evaluation_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change))
                if self.show: self.domain.show(s,policy.pi(s),self.representation)
            
            #Policy Improvement:
            policy_improvement_iteration += 1
            new_policy = zeros(no_of_states)
            policyChanged = 0
            for i in arange(no_of_states):
                s = array(id2vec(i,rep.bins_per_dim))
                for a in self.domain.possibleActions(s):
                    self.BellmanBackup(s,a,self.ns_samples, policy)
                if policy.pi(s) != self.representation.bestAction(s): policyChanged += 1 
                
            policy.representation.theta = self.representation.theta.copy() # This will cause the policy to be copied over
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, Policy Change =%d, Return = %0.4f' % (policy_improvement_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, policyChanged, performance_return))

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
            

            