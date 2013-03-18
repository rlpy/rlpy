######################################################
# Developed by N. Kemal Ure Dec 24th 2012 at MIT #
######################################################
import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Policies import *
from Tools import *
from Representations import *
class MDPSolver(object):
    representation      = None          # Link to the representation object 
    domain              = None          # Link to the domain object
    logger              = None          # A simple objects that record the prints in a file
    planning_time       = None          # Amount of time in seconds provided for the solver. After this it returns its performance.
    id                  = None          # The job id of this run of the algorithm
    mainSeed            = 999999999     # To make sure all same job ids see the same random sequence
    maxRuns             = 100           # Maximum number of runs of an algorithm for averaging    
    convergence_threshold = None        # Threshold to determine the convergence of the planner
    ns_samples          = None          # Number of samples to be used to generate estimated bellman backup if the domain does not provide explicit probablities though expectedStep function.
    log_interval        = None          # Number of bellman backups before reporting the performance. (Not planners may use this 
    show                = None          # Show the learning if possible?
    def __init__(self,job_id, representation,domain,logger, planning_time = inf, convergence_threshold = .005, ns_samples = 100, project_path = '.', log_interval = 500, show = False):
        self.id = job_id
        self.representation = representation
        self.domain = domain
        self.logger = logger
        self.ns_samples = ns_samples
        self.planning_time = planning_time
        self.project_path = project_path
        self.log_interval = log_interval
        self.show = show
        self.convergence_threshold = convergence_threshold
        # Set random seed for this job id
        random.seed(self.mainSeed)
        self.randomSeeds = alborzrandint(1,self.mainSeed,self.maxRuns,1)
        random.seed(self.randomSeeds[self.id-1,0])
        self.logger.setOutput("%s/%d-out.txt" % (self.project_path, self.id))
        if self.logger:
            self.logger.line()
            self.logger.log("Job ID:\t\t\t%d" % self.id)
            self.logger.log("Solver:\t\t\t"+str(className(self)))
            self.logger.log("Max Time:\t\t%0.0f(s)" %planning_time)
            self.logger.log('Convergence Threshold:\t%0.3f' % convergence_threshold)
            if not hasFunction(self.domain,'expectedStep'):
                self.logger.log('Next Step Samples:\t%d' % ns_samples)
            self.logger.log('Log Interval:\t\t%d (Backups)' % log_interval)
            self.logger.log('Show Learning:\t\t%d' % show)
    def solve(self):
        #Abstract
        self.result = array(self.result).T
        self.logger.log('Value of S0 is = %0.5f' % self.representation.V(self.domain.s0()))
        self.saveStats()

    def printAll(self):
        printClass(self)  
    def BellmanBackup(self,s,a,ns_samples, policy = None):
        # Applied Bellman Backup to state-action pair s,a
        # i.e. Q(s,a) = E[r + gamma * V(s')]
        # If policy is given then Q(s,a) =  E[r + gamma * Q(s',pi(s')]                                     
        Q                                       = self.representation.Q_oneStepLookAhead(s,a,ns_samples,policy)
        s_index                                 = vec2id(self.representation.binState(s),self.representation.bins_per_dim)
        theta_index                             = self.domain.states_num*a + s_index
        self.representation.theta[theta_index]  =  Q
    def performanceRun(self):
        # Set Exploration to zero and sample one episode from the domain
        eps_length  = 0
        eps_return  = 0
        eps_term    = 0
        eps_discounted_return = 0
        
        s           = self.domain.s0()
        terminal    = False
        #if self.show_performance: 
        #    self.domain.showLearning(self.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.representation.bestAction(s)
            #if self.show_performance: 
                #self.domain.showDomain(s,a)
             #   pl.title('After '+str(total_steps)+' Steps')

            r,ns,eps_term    = self.domain.step(s, a)
            s               = ns
            eps_discounted_return += self.domain.gamma**eps_length*r
            eps_return     += r
            eps_length     += 1
        #if self.show_performance: self.domain.showDomain(s,a)
        #self.agent.policy.turnOnExploration()
        return eps_return, eps_length, eps_term, eps_discounted_return   
    def saveStats(self):
        checkNCreateDirectory(self.project_path+'/')
        savetxt('%s/%d-result.txt' % (self.project_path,self.id),self.result, fmt='%.18e', delimiter='\t')
                