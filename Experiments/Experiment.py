# See http://acl.mit.edu/RLPy for documentation and future code updates

# Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# Neither the name of ACL nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific
# prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# \file Experiment.py
#
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT
#
from Tools import *
from Agents import *
from Domains import *
from Representations import *
import numpy as np
# The Experiment controls the training, testing, and evaluation of the \ref Agents.Agent.Agent "Agent".
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
# All new experiment implementations should inherit from \c %Experiment.


class Experiment(object):
        # Determines whether the domain is shown during performance runs
    showDomain = False
        # The Main Random Seed used to generate other random seeds
    mainSeed = 999999999
        # Array of random seeds. This is used to make sure all jobs start with
        # the same random seed
    randomSeeds = None
        # Maximum number of runs used for averaging, specified so that enough
        # random seeds are generated
    maxRuns = 100
        # ID of the current experiment running
    id = 1
        # The \ref Domains.Domain.Domain "Domain" to be tested on
    domain = None
        # The \ref Agents.Agent.Agent "Agent" to be tested
    agent = None
        # Determines whether the domain and the performance are shown during
        # the experiment
    show_all = 0
        # Determines whether the policy and the value function are shown during
        # the performance runs
    show_performance = 0
        # The Figure window generated to show the results
    result_fig = None
        # An array that stores all generated. Size is
        # stats_num-by-performanceChecks.
    result = None
        # The name of the file used to store the data
    output_filename = ''
        # A simple object that records the prints in a file
    logger = None

    max_steps           = 0     # Total number of interactions
    performanceChecks   = 0     # Number of Performance Checks uniformly scattered along the trajectory
    LOG_INTERVAL        = 0     # Number of seconds between log prints

    #Following Statistics for Online Experiments
    # Statistics are saved as :
    DEBUG               = 0         # Show s,a,r,s'
    STEP                = 0         # Learning Steps
    RETURN              = 1         # Sum of Rewards
    CLOCK_TIME          = 2         # Time in seconds so far
    FEATURE_SIZE        = 3         # Number of features used for value function representation
    EPISODE_LENGTH      = 4
    TERMINAL            = 5         # 0 = No Terminal, 1 = Normal Terminal, 2 = Critical Terminal
    EPISODE_NUMBER      = 6
    DISCOUNTED_RETURN   = 7
    STATS_NUM           = 8         # Number of statistics to be saved        # Initializes the \c %Experiment object. See code
        # \ref Experiment_init "Here".

        # [init code]
        #(agent,domain,logger,exp_naming = EXPERIMENT_NAMING, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)
    def __init__(self, agent, domain, logger, exp_naming=['domain','agent','representation'], id = 1, max_steps = max_steps, show_all= False, performanceChecks = 10, show_performance = False,log_interval = 1, project_path = 'Results/Temp_Project', plot_performance = True):
#    def __init__(self, id, agent, domain, logger, exp_naming, show_all, show_performance, project_path='Results/Temp_Project', plot_performance=1):
        self.id = id
        # Find the corresponding random seed for the experiment id
        assert id > 0
        random.seed(self.mainSeed)
        self.randomSeeds = randint(1, self.mainSeed, self.maxRuns, 1)
        # for x in self.randomSeeds:
        #    print('%d'%x)
        random.seed(self.randomSeeds[self.id - 1, 0])
        self.agent = agent
        self.domain = domain
        self.output_filename = '%d-results.txt' % (id)
        self.project_path = project_path
        self.experiment_path = self.makeExperimentName(exp_naming)
        self.full_path = self.project_path + '/' + self.experiment_path
        checkNCreateDirectory(self.full_path + '/')
        self.max_steps = max_steps
        self.show_all = show_all
        self.performanceChecks = performanceChecks
        self.show_performance = show_performance
        self.plot_performance = plot_performance
        self.logger = logger
        if show_all or show_performance:
            self.result_fig = pl.figure(1, figsize=(14, 10))
            createColorMaps()
        self.logger.line()
        self.logger.log("Experiment:\t\t%s" % className(self))
        self.logger.log("Output:\t\t\t%s/%s" % (
            self.full_path, self.output_filename))
        self.logger.setOutput("%s/%d-out.txt" % (self.full_path, self.id))
        # [init code]

        # Execute a single episode using the current policy to evaluate its performance; no exploration or learning is enabled.  See code
        # \ref Experiment_performanceRun "Here".
        # [performanceRun code]
    def performanceRun(self, total_steps):
        # Set Exploration to zero and sample one episode from the domain
        eps_length          = 0
        eps_return          = 0
        eps_discount_return = 0
        eps_term            = 0
        random_state = np.random.get_state()

        self.agent.policy.turnOffExploration()

        #hidden states are non Markovian variables that can exist in some problems such as the time dependent noise in the helicopter.
        hidden_state = self.domain.hidden_state_
        if isinstance(hidden_state, ndarray):
            hidden_state = hidden_state.copy()

        s = self.domain.s0()
        if self.show_performance:
            self.domain.showLearning(self.agent.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a = self.agent.policy.pi(s)
            if self.show_performance:
                self.domain.showDomain(s, a)
                pl.title('After ' + str(total_steps) + ' Steps')

            r, ns, eps_term = self.domain.step(s, a)
            # self.logger.log("TEST"+str(eps_length)+"."+str(s)+"("+str(a)+")"+"=>"+str(ns))
            s = ns
            eps_return          += r
            eps_discount_return += self.domain.gamma**eps_length * r
            eps_length          += 1
        if self.show_performance:
            self.domain.showDomain(s, a)
        self.agent.policy.turnOnExploration()
        # This hidden state is for domains (such as the noise in the helicopter domain) that include unobservable elements that are evolving over time
        # Ideally the domain should be formulated as a POMDP but we are trying to accomodate them as an MDP
        self.domain.hidden_state_ = hidden_state

        random.set_state(random_state)
        return eps_return, eps_length, eps_term , eps_discount_return
        # [performanceRun code]

        # Prints class info. See code
        # \ref Experiment_printAll "Here".
        # [printAll code]
    def printAll(self):
        printClass(self)
        # [printAll code]

        # Runs the experimental data based on batch or online setting
    def run(self):
        """
        Run the online experiment and collect statistics
        """
        self.result         = zeros((self.STATS_NUM,self.performanceChecks))
        terminal            = True
        total_steps         = 0
        eps_steps           = 0
        performance_tick    = 0
        eps_return          = 0
        episode_number      = 0
        start_log_time      = clock() # Used to bound the number of logs in the file
        self.start_time     = clock() # Used to show the total time took the process

        #if self.show_all: self.domain.showLearning(self.agent.representation)
        while total_steps < self.max_steps:
            if terminal or eps_steps >= self.domain.episodeCap:
                s           = self.domain.s0()
                a           = self.agent.policy.pi(s)
                #Visual
                if self.show_all: self.domain.show(s,a, self.agent.representation)
                # Hash new state for the tabular case
                if isinstance(self.agent.representation,IncrementalTabular): self.agent.representation.addState(s)
                # Output the current status if certain amount of time has been passed
                eps_return      = 0
                eps_steps       = 0
                episode_number += 1

            #Act,Learn,Step
            r,ns,terminal   = self.domain.step(s, a)
            na              = self.agent.policy.pi(ns)
            total_steps     += 1
            eps_steps       += 1
            eps_return      += r

            #Print Current performance
            if (terminal or eps_steps == self.domain.episodeCap) and deltaT(start_log_time) > self.LOG_INTERVAL:
                start_log_time  = clock()
                elapsedTime     = deltaT(self.start_time)
                self.logger.log('%d: E[%s]-R[%s]: Return=%+0.2f, Steps=%d, Features = %d' % (total_steps, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps), eps_return, eps_steps, self.agent.representation.features_num))

            # Hash new state for the tabular case
            if isinstance(self.agent.representation,IncrementalTabular): self.agent.representation.addState(ns)
            self.agent.learn(s,a,r,ns,na,terminal)
            s,a          = ns,na
            #Visual
            if self.show_all: self.domain.show(s,a, self.agent.representation)

            #Check Performance
            if  total_steps % (self.max_steps/self.performanceChecks) == 0:
                if className(self.agent) == 'PolicyEvaluation':
                    #Policy Evaluation Case
                    self.result = self.agent.STATS
                else:
                    #Control Case
                    performance_return, performance_steps, performance_term, performance_discounted_return = self.performanceRun(total_steps)
                    elapsedTime                     = deltaT(self.start_time)
                    self.result[:,performance_tick] = [total_steps, # index = 0
                                                       performance_return, # index = 1
                                                       elapsedTime, # index = 2
                                                       self.agent.representation.features_num, # index = 3
                                                       performance_steps,# index = 4
                                                       performance_term, # index = 5
                                                       episode_number, # index = 6
                                                       performance_discounted_return] # index = 7

                    self.logger.log('%d >>> E[%s]-R[%s]: Return=%+0.2f, Steps=%d, Features = %d' % (total_steps, hhmmss(elapsedTime), hhmmss(elapsedTime*(self.max_steps-total_steps)/total_steps), performance_return, performance_steps, self.agent.representation.features_num))
                    start_log_time      = clock()
                    performance_tick    += 1

        #Visual
        if self.show_all: self.domain.show(s,a, self.agent.representation)
        if self.show_all or self.show_performance and self.result_fig is not None:
            self.result_fig.savefig(self.full_path+'/lastSnapShot.pdf', transparent=True, pad_inches=0)

        # [save code]
    def save(self):
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        savetxt(self.full_path+'/'+self.output_filename,
                self.result, fmt='%.18e', delimiter='\t')
        self.logger.line()
        self.logger.log("Took %s\nResults\t=> %s/%s" % (hhmmss(deltaT(
            self.start_time)), self.full_path, self.output_filename))
        #Plot Performance
        if self.plot_performance:
            performance_fig = pl.figure(2)
            if isinstance(self.agent,PolicyEvaluation):
                row = 2
                ylabel = "||TD-Error||"
            elif isinstance(self.agent.representation.domain,Pendulum_InvertedBalance):
                row = self.EPISODE_LENGTH
                ylabel = "Episode Length"
            else:
                row = self.RETURN
                ylabel = "Performance"
            pl.plot(self.result[0,:],self.result[row,:],'-bo',lw=3,markersize=10)
            pl.xlim(0,self.result[0,-1]*1.01)
            m = min(self.result[row,:])
            M = max(self.result[row,:])
            delta = M-m
            if delta > 0:
                pl.ylim(m-.1*delta-.1,M+.1*delta+.1)
            pl.xlabel('steps', fontsize=16)
            pl.ylabel(ylabel, fontsize=16)
            performance_fig.savefig(self.full_path+'/'+str(self.id)+'-performance.pdf', transparent=True, pad_inches=.1)
            pl.ioff()
            pl.show()


    def makeExperimentName(self, variables):
        exp_name = ''

        if len(variables) == 0:
            return '.'

        for v in variables:
            # Append required prefixes to make variables reachable:
            if lower(v).startswith('representation') or lower(v).startswith('policy'):
                v = 'self.agent.' + v
            else:
                v = 'self.' + v

            if len([x for x in ['self.domain', 'self.agent', 'self.agent.policy', 'self.agent.representation'] if x == lower(v)]):
                exp_name += eval('className(%s)' % v)
                exp_name += '-'
            else: # We want to use the actual value instead, not the className()
                try:
                    eval('%s' % v)
                    exp_name += str(eval('%s' % v))
                    exp_name += '-'
                except:
                    print 'Couldnt parse the command in exp_naming %s' % exp_name
        return exp_name[:-1]
        # [makeExperimentName code]
