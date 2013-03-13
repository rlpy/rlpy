######################################################
# Developed by N. Kemal Ure Dec 24th 2012 at MIT #
######################################################
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
    def __init__(self,job_id, representation,domain,logger, planning_time = inf, project_path = '.'):
        self.id = job_id
        self.representation = representation
        self.domain = domain
        self.logger = logger
        self.planning_time = planning_time
        self.project_path = project_path
        # Set random seed for this job id
        random.seed(self.mainSeed)
        self.randomSeeds = alborzrandint(1,self.mainSeed,self.maxRuns,1)
        random.seed(self.randomSeeds[self.id-1,0])

        if self.logger:
            self.logger.line()
            self.logger.log("Job ID:\t\t\t%d" % self.id)
            self.logger.log("Solver:\t\t\t"+str(className(self)))
            self.logger.log("Max Time:\t\t%0.0f(s)" %planning_time)
    def solve(self):
        abstract
    def printAll(self):
        printClass(self)  
    def BellmanBackup(self,s,a,ns_samples):
        # Performs Bellman backup assuming the model is known if expectedStep function is available it runs a full backup otherwise it uses <ns_samples> samples to estimate the bellman backup
        if hasFunction(self.domain,'expectedStep'):
            self.fullBellmanBackup(s,a)
        else:
            self.approximateBellmanBackup(s,a,ns_samples)
    def approximateBellmanBackup(self,s,a,ns_samples):
        
        # Perform approximate Bellman Backup on a given s,a pair where ns_samples are generated and used for backup
        gamma               = self.representation.domain.gamma
        
        # Sample Next States and rewards
        next_states,rewards = self.domain.sampleStep(s,a,ns_samples)
        update = mean([rewards[i] + gamma*self.representation.V(next_states[i,:]) for i in arange(ns_samples)])        

        # Update the representation
        s_index                                 = vec2id(self.representation.binState(s),self.representation.bins_per_dim)
        theta_index                             = self.domain.states_num*a + s_index
        self.representation.theta[theta_index]  =  update
        print theta_index
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
    def fullBellmanBackup(self,s,a):
        # Performs a full Bellman backup assuming the model is known and it has a function named expectedStep that returns the probabilities for each possible s',r, t
        gamma       = self.domain.gamma

        p,r,ns,t    = self.domain.expectedStep(s,a)
        update = 0
        for i in arange(len(p)):
            update += p[i]*(r[i] + gamma*self.representation.V(ns[i,:]))
        
        # Update the representation
        s_index                                 = vec2id(self.representation.binState(s),self.representation.bins_per_dim)
        theta_index                             = self.domain.states_num*a + s_index
        self.representation.theta[theta_index]  =  update[0]
    def saveStats(self):
        checkNCreateDirectory(self.project_path+'/')
        savetxt('%s/%d-result.txt' % (self.project_path,self.id),self.result, fmt='%.18e', delimiter='\t')

            