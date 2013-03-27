######################################################
# Developed by N. Kemal Ure Dec 10th 2012 at MIT #
# Editted by A. Geramifard March 13th 2013 at MIT #
######################################################
# Classical Value Iteration
# Performs full Bellman Backup on a given s,a pair by sweeping through the state space  
from MDPSolver import *
class ValueIteration(MDPSolver):
    def solve(self):
        self.result = []
        self.start_time     = time() # Used to show the total time took the process
        # Check for Tabular Representation
        rep  = self.representation
        if className(rep) != 'Tabular':
            self.logger.log("Value Iteration works only with the tabular representation.")
            return 0         
                    
        no_of_states        = self.representation.agg_states_num
        prev_return         = inf   # used to track the performance improvement. 
        bellmanUpdates      = 0
        converged           = False
        iteration           = 0
        while deltaT(self.start_time) < self.planning_time and not converged:
            prev_theta = self.representation.theta.copy()
            # Sweep The State Space
            for i in arange(0,no_of_states):
                s = array(id2vec(i,rep.bins_per_dim))*self.representation.binWidth_per_dim
                s += self.domain.statespace_limits[:,0] +.5
                s[self.domain.continuous_dims] -= .5
                actions = self.domain.possibleActions(s)
                # Sweep The Actions
                for a in actions:
                    self.BellmanBackup(s,a,ns_samples = self.ns_samples)                        
                    bellmanUpdates += 1

                    if False and bellmanUpdates % self.check_interval == 0:
                        performance_return, _,_,_  = self.performanceRun()
                        self.logger.log('[%s]: BellmanUpdates=%d, Return=%0.4f' % (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))
                
                if deltaT(self.start_time) > self.planning_time: break
                    
            #check for convergence
            iteration += 1
            theta_change = linalg.norm(prev_theta - self.representation.theta,inf)
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            converged = theta_change < self.convergence_threshold        
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f, Return = %0.4f' % (iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change, performance_return))
            if self.show: self.domain.show(s,a,self.representation)
            
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
        super(ValueIteration,self).solve()
            