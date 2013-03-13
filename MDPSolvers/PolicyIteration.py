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
    def __init__(self,job_id,representation,domain,logger, ns_samples= 10, convergence_threshold = .005, project_path = '.'):
        
        self.ns_samples             = ns_samples
        self.convergence_threshold  = convergence_threshold
        super(ValueIteration, self).__init__(job_id,representation, domain,logger, project_path = project_path)
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
            converged           = False
            #Initialize the policy to a random policy
            policyChanged       = True
            policy              = array([randSet(self.domain.possibleActions(s)) for s in arange(no_of_states)])
            
            while policyChanged and deltaT(self.start_time) < self.planning_time:
                prev_theta = self.representation.theta.copy()
                # Sweep The State Space
                for i in arange(0,no_of_states):
                    s       = array(id2vec(i,rep.bins_per_dim))
                    actions = self.domain.possibleActions(s)
                    # Sweep The Actions
                    for a in actions:
                        self.BellmanBackup(s,a,ns_samples = self.ns_samples)                        
                        bellmanUpdates += 1

                        if False and bellmanUpdates % self.check_interval == 0:
                            performance_return, _,_,_  = self.performanceRun()
                            self.logger.log('[%s]: BellmanUpdates=%d, Return=%0.4f' % (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))

                #check for convergence
                theta_change = linalg.norm(prev_theta - self.representation.theta,inf)
                performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
                converged = theta_change < self.convergence_threshold        
                self.logger.log('[%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f, Return = %0.4f' % (hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change, performance_return))
                #self.domain.show(s,a,self.representation)
                
                # store stats
                self.result.append([bellmanUpdates, # index = 0 
                                   performance_return, # index = 1 
                                   deltaT(self.start_time), # index = 2
                                   self.representation.features_num, # index = 3
                                   performance_steps,# index = 4
                                   performance_term # index = 5
                                ])
                    
            self.result = array(self.result)
            if converged:
                self.logger.log('Converged')
            self.saveStats()
            

            