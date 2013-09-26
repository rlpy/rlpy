#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by N. Kemal Ure Dec 24th 2012 at MIT #
######################################################
#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.'
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Policies import *
from Tools import *
from Representations import *
class MDPSolver(object):
    """MDPSolver is the base class for model based reinforcement learning agents and 
    planners.

    Args:
        job_id (int):   Job ID number used for running multiple jobs on a cluster.

        representation (Representation):    Representation used for the value function.

        domain (Domain):    Domain (MDP) to solve.

        logger (Logger):    Logger object to log information and debugging.

        planning_time (int):    Maximum amount of time in seconds allowed for planning. Defaults to inf (unlimited).

        convergence_threshold (float):  Threshold for determining if the value function has converged.

        ns_samples (int):   How many samples of the successor states to take.

        project_path (str): Output path for saving the results of running the MDPSolver on a domain.

        log_interval (int): Minimum number of seconds between displaying logged information.

        show (bool):    Enable visualization?

    """

    representation      = None          # Link to the representation object
    domain              = None          # Link to the domain object
    logger              = None          # A simple objects that record the prints in a file
    planning_time       = None          # Amount of time in seconds provided for the solver. After this it returns its performance.
    id                  = None          # The job id of this run of the algorithm
    mainSeed            = 999999999     # To make sure all same job ids see the same random sequence
    maxRuns             = 100           # Maximum number of runs of an algorithm for averaging
    convergence_threshold = None        # Threshold to determine the convergence of the planner
    ns_samples          = None          # Number of samples to be used to generate estimated bellman backup if the domain does not provide explicit probablities though expectedStep function.
    log_interval        = None          # Number of bellman backups before reporting the performance. (Not all planners may use this)
    show                = None          # Show the learning if possible?

    def __init__(self,job_id, representation,domain,logger, planning_time = inf, convergence_threshold = .005, ns_samples = 100, project_path = '.', log_interval = 5000, show = False):
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
        self.randomSeeds = randint(1,self.mainSeed,self.maxRuns,1)
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
        """Solve the domain MDP."""
        #Abstract
        self.result = array(self.result).T
        self.logger.log('Value of S0 is = %0.5f' % self.representation.V(*self.domain.s0()))
        self.saveStats()

    def printAll(self):
        printClass(self)

    def BellmanBackup(self,s,a,ns_samples, policy = None):
        """Applied Bellman Backup to state-action pair s,a
        i.e. Q(s,a) = E[r + gamma * V(s')]
        If policy is given then Q(s,a) =  E[r + gamma * Q(s',pi(s')]

        Args:
            s (ndarray):        The current state
            a (int):            The action taken in state s
            ns_samples(int):    Number of next state samples to use.
            policy (Policy):    Policy object to use for sampling actions.
        """
        Q                                       = self.representation.Q_oneStepLookAhead(s,a,ns_samples,policy)
        s_index                                 = vec2id(self.representation.binState(s),self.representation.bins_per_dim)
        theta_index                             = self.representation.agg_states_num*a + s_index
        self.representation.theta[theta_index]  =  Q

    def performanceRun(self):
        """Set Exploration to zero and sample one episode from the domain."""

        eps_length  = 0
        eps_return  = 0
        eps_term    = False
        eps_discounted_return = 0

        s, eps_term, p_actions           = self.domain.s0()
        terminal    = False
        #if self.show_performance:
        #    self.domain.showLearning(self.representation)

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.representation.bestAction(s, eps_term, p_actions)
            r,ns,eps_term, p_actions    = self.domain.step(a)
            s               = ns
            eps_discounted_return += self.domain.gamma**eps_length*r
            eps_return     += r
            eps_length     += 1
        return eps_return, eps_length, eps_term, eps_discounted_return

    def saveStats(self):
        checkNCreateDirectory(self.project_path+'/')
        savetxt('%s/%d-results.txt' % (self.project_path,self.id),self.result, fmt='%.18e', delimiter='\t')

    def hasTime(self):
        """Return a boolean stating if there is time left for planning."""
        return deltaT(self.start_time) < self.planning_time


