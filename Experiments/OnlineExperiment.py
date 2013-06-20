#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Experiment import *

class OnlineExperiment (Experiment):
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
    STATS_NUM           = 8         # Number of statistics to be saved

    max_steps           = 0     # Total number of interactions
    performanceChecks   = 0     # Number of Performance Checks uniformly scattered along the trajectory
    LOG_INTERVAL        = 0     # Number of seconds between log prints
    def __init__(self,agent,domain, logger,
                 exp_naming = ['domain','agent','representation'],
                 id = 1,
                 max_steps = 10000,
                 performanceChecks = 10,
                 show_all   = False,
                 show_performance = False,
                 log_interval = 1,
                 project_path    = 'Results/Temp',
                 output_filename = '1-results.txt',
                 plot_performance = True):
        self.max_steps          = max_steps
        self.performanceChecks  = performanceChecks
        self.LOG_INTERVAL       = log_interval
        super(OnlineExperiment,self).__init__(id,agent,domain,logger, exp_naming, show_all, show_performance,project_path = project_path, plot_performance=plot_performance)
        self.logger.log("Learning Steps:\t\t%d" % max_steps)
        self.logger.log("Performance Checks:\t%d" % performanceChecks)
        self.logger.log("Log Intervals:\t\t%d(s)" % self.LOG_INTERVAL)

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
        if self.show_all: self.domain.showLearning(self.agent.representation)
        while total_steps < self.max_steps:
            if terminal:
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
            terminal        = terminal or eps_steps >= self.domain.episodeCap

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

    def save(self):
        super(OnlineExperiment,self).save()
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
