######################################################
# Developed by N. Kemal Ure Dec 10th 2012 at MIT #
######################################################
from Experiment import *
# Collects samples by sweeping through the state-action space
class SweepExperiment (Experiment):
    # Statistics are saved as : 
    # ITERATION       = 0 
    # RETURN          = 1 
    # CLOCK_TIME      = 2 
    # FEATURE_SIZE    = 3 
    max_iterations      = 0     # Total number of iterations
    performanceChecks   = 0     # Number of Performance Checks uniformly scattered along the trajectory
    STATS_NUM           = 4     # Number of statistics to be saved
    LOG_INTERVAL        = 0     # Number of seconds between log prints
    def __init__(self,agent,domain, logger,
                 id = 1,
                 max_iterations = 10, 
                 performanceChecks = 10,
                 show_all   = False, 
                 show_performance = False,
                 log_interval = 1,
                 output_path     = 'Results/Temp',
                 output_filename = 'results.txt',
                 plot_performance = True):
        super(SweepExperiment,self).__init__(id,agent,domain,logger, show_all, show_performance,output_path = output_path, output_filename = output_filename, plot_performance=plot_performance)
        self.max_iterations = max_iterations
        self.max_steps          = (self.max_iterations)*(self.domain.actions_num)*(self.domain.states_num)
        self.performanceChecks  = performanceChecks
        self.LOG_INTERVAL       = log_interval
      
        self.logger.log("Max Steps: \t\t%d" % self.max_steps)
        self.logger.log("Performance Checks:\t%d" % performanceChecks)
        
    def run(self):
            start_log_time      = time() # Used to bound the number of logs in the file  
            self.start_time     = time() # Used to show the total time took the process
        
            print '-----------------------'
            print 'Experiment Started !! '
    # Tabular Rep
            rep  = Tabular(self.domain,self.logger)        
            
            
    # Run the Sweep experiment and collect statistics
            no_of_states = self.domain.states_num
            
            #print 'Limits :', rep.bins_per_dim
            #print 'Number of total iterations ',self.max_iterations
            for iter in range(0,self.max_iterations):
            #    print 'At Sweep ', iter
                prev_theta =  self.agent.representation.theta.copy()
                for i in range(0,no_of_states):
                    #print 'At state index: ',i
                    
                    s = id2vec(i,rep.bins_per_dim)
                    s = array(s, int)
                    #print 'At state: ',s
                    #if(not self.domain.isTerminal(s)):
                                      
                    actions = self.domain.possibleActions(s)
                    
                    for j in actions:
                     #print 'At action: ',j                           
                     self.agent.learn(s =s,a =j)                        
                            

            #Print Current performance (after each iteration)
                start_log_time  = time()
                elapsedTime     = deltaT(self.start_time) 
                performance_return, performance_steps, performance_term  = self.performanceRun(iter)
               # self.logger.log('%d: E[%s-R[%s]: Return=%0.2f' % (iter, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_iterations-iter)/self.max_iterations),performance_return))
                print '{}:  Return= {}' . format( iter,performance_return)
                print 'L2 Norm Difference of Theta :',  linalg.norm(self.agent.policy.representation.theta - prev_theta)
                #print ' Theta, ', self.agent.representation.theta
            
    def save(self):
        super(SweepExperiment,self).save()
        #Plot Performance
        if self.plot_performance:
            performance_fig = pl.figure(2)
            pl.plot(self.result[0,:],self.result[1,:],'-bo',lw=3,markersize=10)
            pl.xlim(0,self.result[0,-1])
            m = min(self.result[1,:])
            M = max(self.result[1,:])
            pl.ylim(m-.1*abs(M),M+.1*abs(M))
            pl.xlabel('steps',fontsize=16)
            pl.ylabel('Performance',fontsize=16)
            performance_fig.savefig(self.output_path+'/'+str(self.id)+'-performance.pdf', transparent=True, pad_inches=.1)
