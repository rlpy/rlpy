#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Bob Klein
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## \file Experiment.py
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT 
######################################################
from Tools import *
from Agents import *
from Domains import *
from Representations import *

## The Experiment controls the training, testing, and evaluation of the \ref Agents.Agent.Agent "Agent".
# Reinforcement learning is based around the concept of training an Agent to solve a task and later testing its ability to use what it has learned to solve the task.
# This cycle forms a loop that the %Experiment defines and controls. First the Agent is repeatedly tasked with solving a problem determined by the \ref Domains.Domain.Domain "Domain",
# restarting after some termination condition is reached. (The sequence of steps between
# terminations is known as an "episode.")
# Each time the Agent attempts to solve the task, it learns more about how to accomplish its goal. The %Experiment controls this loop of "training
# sessions", iterating over each step in which the Agent and Domain interact. After a set number of training sessions defined by the %Experiment, the Agent's current policy is tested for its performance on the task.
# The %Experiment collects data on the Agent's performance and then puts the Agent through more training sessions. After a set number of loops, training sessions followed by an evaluation, the %Experiment
# is complete and the gathered data is printed and saved. For each section, training and evaluation, the %Experiment determines whether or not the visualization of the step should generated.
#
# The \c %Experiment class is a superclass that provides the basic framework for all RL experiments. It provides the methods and attributes
# that allow child classes to interact with the \c Agent and \c Domain classes within the RLPy library. \n
# All new experiment implimentations should inherit from \c %Experiment.

class Experiment(object):
	## Determines whether the domain is shown during performance runs
    showDomain = False      
	## The Main Random Seed used to generate other random seeds
    mainSeed = 999999999    
	## Array of random seeds. This is used to make sure all jobs start with the same random seed
    randomSeeds = None      
	## Maximum number of runs used for averaging, specified so that enough random seeds are generated	
    maxRuns = 100          
	## ID of the current experiment running
    id = 1                
	## The \ref Domains.Domain.Domain "Domain" to be tested on	
    domain = None           
	## The \ref Agents.Agent.Agent "Agent" to be tested
    agent = None            
	## Determines whether the domain and the performance are shown during the experiment
    show_all = 0            
	## Determines whether the policy and the value function are shown during the performance runs
    show_performance = 0    
	## The Figure window generated to show the results
    result_fig = None       
	## An array that stores all generated. Size is stats_num-by-performanceChecks.
    result  = None          
	## The name of the file used to store the data
    output_filename = ''    
	## A simple object that records the prints in a file
    logger = None  
	
	## Initializes the \c %Experiment object. See code
	# \ref Experiment_init "Here".
	
	# [init code]
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
        self.logger.setOutput("%s/%d-out.txt" % (self.full_path, self.id))
	# [init code]
	
	
	## Execute a single episode using the current policy to evaluate its performance; no exploration or learning is enabled.  See code
	# \ref Experiment_performanceRun "Here".
	
	# [performanceRun code]
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
	# [performanceRun code]
	
	
	## Prints class info. See code
	# \ref Experiment_printAll "Here".
	
	# [printAll code]
    def printAll(self):
        printClass(self)
	# [printAll code]
	
	
	## Runs the experimental data based on batch or online setting
    def run(self):
		abstract
		
	## Saves experimental data. See code
	# \ref Experiment_save "Here".
    
	# [save code]
    def save(self):
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        savetxt(self.full_path+'/'+self.output_filename,self.result, fmt='%.18e', delimiter='\t')
        self.logger.line()
        self.logger.log("Took %s\nResults\t=> %s/%s" % (hhmmss(deltaT(self.start_time)), self.full_path, self.output_filename))
        # Set the output path for the logger
        # This is done here because it is dependent on the combination of agent, representation, and domain
	# [save code]
	
	
	## Creates a string name for the experiment by connecting the values corresponding to the given variables.  
	# See code \ref Experiment_makeExperimentName "Here".
	# @param variables The list of variables to be used for the experiment name. \n \b Example: [ 'domain', 'agent', 'representation', 'LEARNING_STEPS']
	# @returns The string name associated with the variables. \n \b Example: 'GridWorld-SARSA-Tabular-10000'
	
	# [makeExperimentName code]
    def makeExperimentName(self,variables):
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
	# [makeExperimentName code]
