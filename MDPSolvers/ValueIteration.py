######################################################
# Developed by N. Kemal Ure Dec 10th 2012 at MIT #
######################################################
# Classical Value Iteration
# Performs full Bellman Backup on a given s,a pair by sweeping through the state space  
from MDPSolver import *
class ValueIteration(MDPSolver):
    mc_ns_samples = 0         # Number of next state samples used to approximate BE expectation
    max_iterations      = 0     # Total number of iterations          
        
    def __init__(self,representation,domain,logger, mc_ns_samples= 10,max_iterations = 3):
        
        self.phi_sa_size        = domain.actions_num * representation.features_num
        self.mc_ns_samples = mc_ns_samples
        self.max_iterations = max_iterations
        super(ValueIteration, self).__init__(representation, domain,logger)
        
    def solve(self):
            
            start_log_time      = time() # Used to bound the number of logs in the file  
            self.start_time     = time() # Used to show the total time took the process
        
            print '-----------------------'
            print 'Value Iteration Started !! '
            
    # Tabular Representation
            rep  = Tabular(self.domain,self.logger)        
                        
    # Sweep The State Space
            no_of_states = self.domain.states_num
            
            for iter in arange(0,self.max_iterations):
                
                print 'At Sweep: ', iter
                prev_theta =  self.representation.theta.copy()
                for i in arange(0,no_of_states):
                #for i in arange(0,3):
                    #print 'At state index: ',i
                    
                    s = array(id2vec(i,rep.bins_per_dim))
                    #s = array(s, int)
                    print 'At state: ',s
                    #if(not self.domain.isTerminal(s)):
                                      
                    actions = self.domain.possibleActions(s)
                    
                    for j in actions:
                     #print 'At action: ',j                           
                     self.MCBellmanBackup(s =s,a =j,mc_ns_samples = self.mc_ns_samples)                        
                            

            #Print Current performance (after each iteration)
                start_log_time  = time()
                elapsedTime     = deltaT(self.start_time) 
                performance_return, performance_steps, performance_term  = self.performanceRun(iter)
               # self.logger.log('%d: E[%s-R[%s]: Return=%0.2f' % (iter, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_iterations-iter)/self.max_iterations),performance_return))
                print '{}:  Return= {}' . format( iter,performance_return)
                print 'L2 Norm Difference of Theta :',  linalg.norm(self.representation.theta - prev_theta)
                #print ' Theta, ', self.agent.representation.theta          
        
        
        
        
        
      