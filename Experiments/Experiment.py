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
    maxRuns = 100           # Maximum number of runs used for averaging. This is specified so that enough random seeds are generated
    id = 0                  # ID of the experiment running
    domain = None           # link to the domain object
    agent = None            # link to the agent object
    show_all = 0            # Show the domain and the value function during the experiment
    show_performance = 0    # Show the domain and the value function during the performance runs
    result_fig = None       # Figure window generated to show the results
    result  = None          # All data is saved in the result array: stats_num-by-performanceChecks
    def __init__(self,id, agent, domain, show_all, show_performance):
        self.id = id
        # Find the corresponding random seed for the experiment id
        random.seed(self.mainSeed)
        self.randomSeeds = random.randint(self.mainSeed,size=self.maxRuns)
        random.seed(self.randomSeeds[self.id])
        self.agent              = agent
        self.domain             = domain
        self.show_all           = show_all
        self.show_performance   = show_performance
        if show_all or show_performance: 
            self.result_fig = pl.figure(1,figsize=(14, 10))
            createColorMaps()
        print join(["-"]*30)
        print "Experiment:\t", className(self)
        print "Agent:\t", className(self)
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

            r,s,eps_term    = self.domain.step(s, a)
            eps_return     += r
            eps_length     += 1
        if self.show_performance: self.domain.showDomain(s,a)
        self.agent.policy.turnOnExploration()
        return eps_return, eps_length, eps_term
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
    def save(self,filename):
        savetxt(filename,self.result)
        print filename,'<== Saved Results!'
