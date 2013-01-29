######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from Agents import *
from Domains import *
from Representations import *

class Experiment(object):
    showDomain = False      # Show the domain during execution?
    mainSeed = 999999999    # Main Random Seed used to generate other random seeds
    randomSeeds = None      # Array of random seeds. This is used to make sure all jobs start with the same random seed 
    maxRuns = 100          # Maximum number of runs used for averaging. This is specified so that enough random seeds are generated
    id = 1                  # ID of the experiment running
    domain = None           # link to the domain object
    agent = None            # link to the agent object
    show_all = 0            # Show the domain and the value function during the experiment
    show_performance = 0    # Show the domain and the value function during the performance runs
    result_fig = None       # Figure window generated to show the results
    result  = None          # All data is saved in the result array: stats_num-by-performanceChecks
    output_filename = ''    # The name of the file used to store the data
    logger = None           # An object to record the print outs in a file
    def __init__(self,id, agent, domain,logger, exp_naming, show_all, show_performance, project_path = 'Results/Temp_Project', plot_performance = 1):
        self.id = id
        # Find the corresponding random seed for the experiment id
        assert id > 0
        random.seed(self.mainSeed)
        self.randomSeeds = alborzrandint(1,self.mainSeed,self.maxRuns,1)
        #for x in self.randomSeeds:
        #    print('%d'%x) 
        random.seed(self.randomSeeds[self.id-1,0])
        self.agent              = agent
        self.domain             = domain
        self.output_filename    = '%d-results.txt' % (id)
        self.project_path       = project_path
        self.experiment_path    = self.makeExperimentName(exp_naming)
        self.full_path          = self.project_path+ '/' + self.experiment_path
        checkNCreateDirectory(self.full_path+'/')
        self.show_all           = show_all
        self.show_performance   = show_performance
        self.plot_performance   = plot_performance   
        self.logger             = logger
        if show_all or show_performance: 
            self.result_fig = pl.figure(1,figsize=(14, 10))
            createColorMaps()
        self.logger.line()
        self.logger.log("Experiment:\t\t%s" % className(self))
        self.logger.log("Output:\t\t\t%s/%s" % (self.full_path, self.output_filename))
    def performanceRun(self,total_steps):
        # Set Exploration to zero and sample one episode from the domain
        eps_length  = 0
        eps_return  = 0
        eps_term    = 0
        self.agent.policy.turnOffExploration()
        s           = self.domain.s0()
        terminal    = False
        if self.show_performance: 
            self.domain.showLearning(self.agent.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.agent.policy.pi(s)
            if self.show_performance: 
                self.domain.showDomain(s,a)
                pl.title('After '+str(total_steps)+' Steps')

            r,ns,eps_term    = self.domain.step(s, a)
            #self.logger.log("TEST"+str(eps_length)+"."+str(s)+"("+str(a)+")"+"=>"+str(ns))
            s               = ns
            eps_return     += r
            eps_length     += 1
        if self.show_performance: self.domain.showDomain(s,a)
        self.agent.policy.turnOnExploration()
        return eps_return, eps_length, eps_term
    def printAll(self):
        printClass(self)
    def save(self):
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        savetxt(self.full_path+'/'+self.output_filename,self.result, fmt='%0.4f', delimiter='\t')
        self.logger.line()
        self.logger.log("Took %s\nResults\t=> %s/%s" % (hhmmss(deltaT(self.start_time)), self.full_path, self.output_filename))
        # Set the output path for the logger
        # This is done here because it is dependent on the combination of agent, representation, and domain
        self.logger.save("%s/%d-out.txt" % (self.full_path, self.id))
    def makeExperimentName(self,variables):
        # Creates a string name for the experiment by connecting the values corresponding to the variables mentioned in X  
        # Example: X = ['domain','agent','representation','LEARNING_STEPS']
        # Output: 'PitMaze-SARSA-Tabular-10000'
        exp_name = ''
        
        if len(variables) == 0:
            return '.'

        for v in variables:
            #Append required prefixes to make variables reachable:
            if lower(v).startswith('representation') or lower(v).startswith('policy'):
                v = 'self.agent.' + v
            else:
                v = 'self.' + v
            
            if len([x for x in ['self.domain','self.agent','self.agent.policy','self.agent.representation'] if x == lower(v)]):
                exp_name += eval('className(%s)' % v)
                exp_name += '-'
            else:
                try:
                    eval('%s' % v)
                    exp_name += str(eval('%s' % v))
                    exp_name += '-'
                except:
                    pass
        return exp_name[:-1]  
