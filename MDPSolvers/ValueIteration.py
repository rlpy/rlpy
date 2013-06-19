#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        self.start_time     = clock() # Used to show the total time took the process
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
        while self.hasTime() and not converged:
            prev_theta = self.representation.theta.copy()
            # Sweep The State Space
            for i in arange(0,no_of_states):
                if not self.hasTime(): break
                s = self.representation.stateID2state(i)
                actions = self.domain.possibleActions(s)
                # Sweep The Actions
                for a in actions:
                    if not self.hasTime(): break
                    self.BellmanBackup(s,a,ns_samples = self.ns_samples)                        
                    bellmanUpdates += 1

                    if bellmanUpdates % self.log_interval == 0:
                        performance_return, _,_,_  = self.performanceRun()
                        self.logger.log('[%s]: BellmanUpdates=%d, Return=%0.4f' % (hhmmss(deltaT(self.start_time)), bellmanUpdates, performance_return))
                
                
                    
            #check for convergence
            iteration += 1
            theta_change = linalg.norm(prev_theta - self.representation.theta,inf)
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            converged = theta_change < self.convergence_threshold        
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f, Return=%0.4f, Steps=%d' % (iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, theta_change, performance_return, performance_steps))
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
            