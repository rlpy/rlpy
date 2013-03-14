######################################################
# Developed by A. Geramifard March 13th 2013 at MIT #
######################################################
# Classical Policy Iteration
# Performs Bellman Backup on a given s,a pair given a fixed policy by sweeping through the state space
# Once the errors are bounded, the policy is changed  
from MDPSolver import *
class PolicyIteration(MDPSolver):
    mc_ns_samples = 0          # Number of next state samples used to approximate BE expectation
    check_interval = 500      # After how many bellman backups show the performance        
    def __init__(self,job_id,representation,domain,logger, ns_samples= 10, convergence_threshold = .1, project_path = '.'):
        
        self.ns_samples             = ns_samples
        self.convergence_threshold  = convergence_threshold
        super(PolicyIteration, self).__init__(job_id,representation, domain,logger, project_path = project_path)
        self.logger.log('Convergence Threshold:\t%0.3f' % convergence_threshold)
    def solve(self):
            self.result = []
            self.start_time     = time() # Used to show the total time took the process
            
            # Check for Tabular Representation
            rep  = self.representation
            if className(rep) != 'Tabular':
                self.logger.log("Value Iteration works only with the tabular representation.")
                return 0         
                        
            no_of_states    = self.domain.states_num
            bellmanUpdates      = 0
            
            #Initialize the policy to a random policy
            policyChanged       = True
            policy              = empty(no_of_states)
            for i in arange(no_of_states):
                s           = array(id2vec(i,rep.bins_per_dim))
                policy[i]   = randSet(self.domain.possibleActions(s))
            
            policy_improvement_iteration = 0
            while policyChanged and deltaT(self.start_time) < self.planning_time:
                policy_improvement_iteration += 1
                self.logger.log("Policy Iteration #%d:" % policy_improvement_iteration)
    
                # Policy Evaluation
                converged = False
                policy_evaluation_iteration = 0
                while not converged:
                    policy_evaluation_iteration += 1
                    prev_theta = self.representation.theta.copy()
                    # Sweep The State Space
                    for i in arange(0,no_of_states):
                        s       = array(id2vec(i,rep.bins_per_dim))
                        self.BellmanBackup(s,policy[i],self.ns_samples)                        
                        bellmanUpdates += 1
                    
                    #check for convergence
                    theta_change = linalg.norm(prev_theta - self.representation.theta,inf)
                    converged = theta_change < self.convergence_threshold        
                    self.logger.log('#%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f' % (policy_evaluation_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change))
                    #self.domain.show(s,policy[0],self.representation)
                
                #Policy Improvement:
                new_policy = zeros(no_of_states)
                for i in arange(no_of_states):
                    s               = array(id2vec(i,rep.bins_per_dim))
                    _,new_policy[i] = self.representation.V_oneStepLookAhead(s,self.ns_samples)
                
                # See policy change
                policyChanged       = (policy - new_policy).nonzero()[0]
                #print policyChanged
                policyChanged       = len(policyChanged)
                policy              = new_policy
                performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
                self.logger.log('#%d [%s]: BellmanUpdates=%d, Policy Change =%d, Return = %0.4f' % (policy_improvement_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, policyChanged, performance_return))

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
                    
            self.result = array(self.result).T
            print self.representation.V(self.domain.s0())
            if converged:
                self.logger.log('Converged')
            self.saveStats()
            

            