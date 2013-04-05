######################################################
# Developed by A. Geramifard March 14th 2013 at MIT #
######################################################
# Trajectory Based Value Iteration. This algorithm is different from Value iteration in 2 senses:
# 1. It works with any Linear Function approximator
# 2. Samples are gathered using the e-greedy policy
# The algorithm terminates if the maximum bellman-error in a consequent set of trajectories is below a threshold
from MDPSolver import *
class TrajectoryBasedValueIteration(MDPSolver):
    epsilon     = None # Probability of taking a random action during each decision making
    alpha       = .1 # step size parameter to adjust the weights. If the representation is tabular you can set this to 1.
    MIN_CONVERGED_TRAJECTORIES = 5 # Minimum number of trajectories required for convergence in which the max bellman error was below the threshold
    def __init__(self,job_id, representation,domain,logger, planning_time = inf, convergence_threshold = .005, ns_samples = 100, project_path = '.', log_interval = 500, show = False, epsilon = .1):
        super(TrajectoryBasedValueIteration,self).__init__(job_id, representation,domain,logger, planning_time, convergence_threshold, ns_samples, project_path,log_interval, show)
        self.epsilon = epsilon
        if className(representation) == 'Tabular': 
            self.alpha = 1
        else:
            self.logger.log('alpha:\t\t\t%0.2f' % self.alpha)
        self.logger.log('epsilon:\t\t\t%0.2f' % self.epsilon)
        self.logger.log('# Trajectories used for convergance: %d' % self.MIN_CONVERGED_TRAJECTORIES)
    def solve(self):
        self.result = []
        self.start_time     = time() # Used to show the total time took the process
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
            a                       = self.representation.bestAction(s) if random.rand() > self.epsilon else randSet(self.domain.possibleActions(s)) 
            while not terminal and step < self.domain.episodeCap and self.hasTime():
                new_Q           = self.representation.Q_oneStepLookAhead(s,a, self.ns_samples)
                phi_s           = self.representation.phi(s)
                phi_s_a         = self.representation.phi_sa(s,a,phi_s)
                old_Q           = dot(phi_s_a,self.representation.theta)
                bellman_error   = new_Q - old_Q
                #print s, old_Q, new_Q, bellman_error
                self.representation.theta   += self.alpha * bellman_error * phi_s_a
                bellmanUpdates              += 1
                step                        += 1
                
                #Discover features if the representation has the discover method
                discover_func = getattr(self.representation,'discover',None) # None is the default value if the discover is not an attribute
                if discover_func and callable(discover_func):
                    self.representation.discover(phi_s,bellman_error)

                max_Bellman_Error = max(max_Bellman_Error,abs(bellman_error))
                #Simulate new state and action on trajectory
                _,s,terminal    = self.domain.step(s,a)
                a               = self.representation.bestAction(s) if random.rand() > self.epsilon else randSet(self.domain.possibleActions(s)) 
            
            #check for convergence
            iteration += 1
            if max_Bellman_Error < self.convergence_threshold:
                converged_trajectories += 1
            else:
                converged_trajectories = 0
            performance_return, performance_steps, performance_term, performance_discounted_return  = self.performanceRun()
            converged = converged_trajectories >= self.MIN_CONVERGED_TRAJECTORIES      
            self.logger.log('PI #%d [%s]: BellmanUpdates=%d, ||Bellman_Error||=%0.4f, Return=%0.4f, Steps=%d, Features=%d' % (iteration, hhmmss(deltaT(self.start_time)), bellmanUpdates, max_Bellman_Error, performance_return,performance_steps,self.representation.features_num))
            if self.show:  self.domain.show(s,a,self.representation)
            
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
        
