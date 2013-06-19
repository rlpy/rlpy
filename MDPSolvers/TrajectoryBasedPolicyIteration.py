#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by A. Geramifard March 15th 2013 at MIT #
######################################################
# Trajectory Based Policy Iteration:
# Loop until the weight change to the value function is small for some number of trajectories (cant check policy because we dont store anything in the size of the state-space)
# 1. Update the evaluation of the policy till the change is small.
# 2. Update the policy
#
# * There is solveInMatrixFormat function which does policy evaluation in one shot using samples collected in the matrix format. Since the algorithm toss out the samples convergence is hardly reached, because the policy may alternate.
from MDPSolver import *
class TrajectoryBasedPolicyIteration(MDPSolver):
    epsilon     = None # Probability of taking a random action during each decision making
    alpha       = .1 # step size parameter to adjust the weights. If the representation is tabular you can set this to 1.
    MIN_CONVERGED_TRAJECTORIES = 5 # Minimum number of trajectories required for convergence in which the max bellman error was below the threshold
    def __init__(self,job_id, representation,domain,logger, planning_time = inf, convergence_threshold = .005, ns_samples = 100, project_path = '.', log_interval = 500, show = False, epsilon = .1, max_PE_iterations = 10):
        super(TrajectoryBasedPolicyIteration,self).__init__(job_id, representation,domain,logger, planning_time, convergence_threshold, ns_samples, project_path,log_interval, show)
        self.epsilon = epsilon
        self.max_PE_iterations = max_PE_iterations
        if className(representation) == 'Tabular': 
            self.alpha = 1
        else:
            self.logger.log('gradient step:\t\t\t%0.2f' % self.alpha)
        self.logger.log('epsilon:\t\t\t%0.2f' % self.epsilon)
        self.logger.log('# Trajectories used for convergence: %d' % self.MIN_CONVERGED_TRAJECTORIES)
        self.logger.log('Max PE Iterations:\t%d' % self.max_PE_iterations)
    def solve(self):
        self.result         = []
        self.start_time     = clock() # Used to show the total time took the process
        bellmanUpdates      = 0
        converged           = False
        PI_iteration        = 0
        # The policy is maintained as separate copy of the representation.
        # This way as the representation is updated the policy remains intact
        policy              = eGreedy(deepcopy(self.representation),self.logger, epsilon = 0, forcedDeterministicAmongBestActions = True) # Copy the representation so that the weight change during the evaluation does not change the policy
        
        while self.hasTime() and not converged:
            # Policy Evaluation
            PE_iteration            = 0
            evaluation_is_accurate  = False
            converged_trajectories  = 0
            while not evaluation_is_accurate and deltaT(self.start_time) < self.planning_time and PE_iteration < self.max_PE_iterations:
                # Generate a new episode e-greedy with the current values
                max_Bellman_Error       = 0
                step                    = 0
                terminal                = False
                s                       = self.domain.s0()
                a                       = policy.pi(s) if random.rand() > self.epsilon else randSet(self.domain.possibleActions(s)) 
                while not terminal and step < self.domain.episodeCap and self.hasTime():
                    
                    #print "Policy Features = %d" % policy.representation.features_num
                    #print "Policy iFDD Feature = %d" % len(policy.representation.iFDD_features.keys())
                    #print "Policy iFDD Potentials = %d" % len(policy.representation.iFDD_potentials.keys())
                    #print "Policy iFDD Sorted = %d" % len(policy.representation.sortediFDDFeatures.h)
                    #print "Policy iFDD index2feature = %d" % len(policy.representation.featureIndex2feature.keys())
                    #print "Policy theta = %d" % len(self.representation.theta)
                    new_Q           = self.representation.Q_oneStepLookAhead(s,a, self.ns_samples,policy)
                    phi_s           = self.representation.phi(s)
                    phi_s_a         = self.representation.phi_sa(s,a,phi_s)
                    old_Q           = dot(phi_s_a,self.representation.theta)
                    bellman_error   = new_Q - old_Q
                    
                    #print s, old_Q, new_Q, bellman_error
                    self.representation.theta   += self.alpha * bellman_error * phi_s_a
                    bellmanUpdates              += 1
                    step                        += 1
                    max_Bellman_Error = max(max_Bellman_Error,abs(bellman_error))
                    
                    #Discover features if the representation has the discover method
                    discover_func = getattr(self.representation,'discover',None) # None is the default value if the discover is not an attribute
                    if discover_func and callable(discover_func):
                        discovered = self.representation.discover(phi_s,bellman_error)
                        #if discovered: 
                        #    print "Features = %d" % self.representation.features_num
                    
                    #Simulate new state and action on trajectory
                    _,s,terminal    = self.domain.step(s,a)
                    a               = policy.pi(s) if random.rand() > self.epsilon else randSet(self.domain.possibleActions(s)) 
            
                #check for convergence of policy evaluation
                PE_iteration += 1
                if max_Bellman_Error < self.convergence_threshold:
                    converged_trajectories += 1
                else:
                    converged_trajectories = 0
                evaluation_is_accurate = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES      
                self.logger.log('PE #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Features=%d' % (PE_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, max_Bellman_Error, self.representation.features_num))
            
            # Policy Improvement (Updating the representation of the value function will automatically improve the policy
            PI_iteration += 1
            
            #Calculate the change in the theta as L2-norm
            #Theta may have increased in size if the representation is expanded.
            paddedTheta = padZeros(policy.representation.theta,len(self.representation.theta))
            delta_theta = linalg.norm(paddedTheta-self.representation.theta)
            converged = delta_theta < self.convergence_threshold 

            #Update the underlying value function of the policy
            policy.representation = deepcopy(self.representation)
            
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            self.logger.line()
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, ||delta-theta||=%0.4f, Return=%0.3f, steps=%d, features=%d' % (PI_iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, delta_theta, performance_return, performance_steps, self.representation.features_num))
            self.logger.line()
            if self.show:  self.domain.show(s,a,self.representation)
            
            # store stats
            self.result.append([bellmanUpdates, # index = 0 
                               performance_return, # index = 1 
                               deltaT(self.start_time), # index = 2
                               self.representation.features_num, # index = 3
                               performance_steps,# index = 4
                               performance_term, # index = 5
                               performance_discounted_return, # index = 6
                               PI_iteration #index = 7 
                            ])
                
        if converged: self.logger.log('Converged!')
        super(TrajectoryBasedPolicyIteration,self).solve()
               
    def solveInMatrixFormat(self):
        # while delta_theta > threshold
        #  1. Gather data following an e-greedy policy
        #  2. Calculate A and b estimates 
        #  3. calculate new_theta, and delta_theta
        # return policy greedy w.r.t last theta
        self.policy         = eGreedy(self.representation,self.logger, epsilon = self.epsilon)
        self.samples_num     = 1000 # Number of samples to be used for each policy evaluation phase. L1 in the Geramifard et. al. FTML 2012 paper
        self.result         = []
        self.start_time     = clock() # Used to show the total time took the process
        samples             = 0
        converged           = False
        iteration           = 0
        while deltaT(self.start_time) < self.planning_time and not converged:
            
            #  1. Gather samples following an e-greedy policy
            S,Actions,NS,R,T  = self.policy.collectSamples(self.samples_num)
            samples     += self.samples_num
            #  2. Calculate A and b estimates
            L1      = self.samples_num
            a_num   = self.domain.actions_num
            n       = self.representation.features_num
            gamma   = self.domain.gamma
    
            self.A = zeros((n*a_num,n*a_num))
            self.b = zeros((n*a_num,1))
            for i in arange(self.samples_num):
                 phi_s_a        = self.representation.phi_sa(S[i],Actions[i,0]).reshape((-1,1))
                 E_phi_ns_na    = self.calculate_expected_phi_ns_na(S[i], Actions[i,0], self.ns_samples).reshape((-1,1))
                 d              = phi_s_a - gamma * E_phi_ns_na
                 self.A         += outer(phi_s_a,d.T)
                 self.b         += phi_s_a*R[i,0]
            
            #  3. calculate new_theta, and delta_theta
            new_theta,solve_time = solveLinear(regularize(self.A),self.b)
            iteration += 1
            if solve_time > 1: self.logger.log('#%d: Finished Policy Evaluation. Solve Time = %0.2f(s)' % (iteration, solve_time))
            delta_theta = linalg.norm(new_theta-self.representation.theta,inf)
            converged   = delta_theta < self.convergence_threshold
            self.representation.theta = new_theta
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            self.logger.log('#%d [%s]: Samples=%d, ||weight-Change||=%0.4f, Return = %0.4f' % (iteration, hhmmss(deltaT(self.start_time)), samples, delta_theta, performance_return))
            if self.show:  self.domain.show(S[-1],Actions[-1],self.representation)
            
            # store stats
            self.result.append([samples, # index = 0 
                               performance_return, # index = 1 
                               deltaT(self.start_time), # index = 2
                               self.representation.features_num, # index = 3
                               performance_steps,# index = 4
                               performance_term, # index = 5
                               performance_discounted_return, # index = 6
                               iteration #index = 7 
                            ])
                
        if converged: self.logger.log('Converged!')
        super(TrajectoryBasedPolicyIteration,self).solve()
        
    def calculate_expected_phi_ns_na(self,s,a, ns_samples):
        # calculate the expected next feature vector (phi(ns,pi(ns)) given s and a. Eqns 2.20 and 2.25 in [Geramifard et. al. 2012 FTML draft] 
        if hasFunction(self.domain,'expectedStep'):
            p,r,ns,t    = self.domain.expectedStep(s,a)
            phi_ns_na   = zeros(self.representation.features_num*self.domain.actions_num)
            for j in arange(len(p)):
                na = self.policy.pi(ns[j])
                phi_ns_na += p[j]*self.representation.phi_sa(ns[j],na)
        else:
            next_states,rewards = self.domain.sampleStep(s,a,ns_samples)
            phi_ns_na = mean([self.representation.phisa(next_states[i],self.policy(next_states[i])) for i in arange(ns_samples)])
        return phi_ns_na            

        
                     
