#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Bob Klein
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################

import sys, os
#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)
  

from Tools import *
from Domains import *
from Agents import *
from Representations import *
from Policies import *
from Experiments import *
#from pandas.tests.test_series import CheckNameIntegration

def main(jobID=-1,              # Used as an indicator for each run of the algorithm
         PROJECT_PATH = '.',    # Path to store the results. Notice that a directory is automatically generated within this directory based on the EXPERIMENT_NAMING 
         SHOW_FINAL_PLOT = 0,   # Draw the final plot when the run is finished? Automatically set to False if jobID == -1
         MAKE_EXP_NAME = 1      # This flag should be set 0 if the job is submitted through the condor cluster so no extra directory is built. Basically all the results are stored in the directory where the main file is.
         ):

    # Etc
    #----------------------
    PERFORMANCE_CHECKS  = 10
    LEARNING_STEPS      = 2000
    #EXPERIMENT_NAMING   = ['domain','agent','representation']
    EXPERIMENT_NAMING   = ['domain','representation','max_steps','representation.batchThreshold'] 
    EXPERIMENT_NAMING   = [] if not MAKE_EXP_NAME else EXPERIMENT_NAMING
    RUN_IN_BATCH        = jobID != -1
    SHOW_ALL            = 0 and not RUN_IN_BATCH
    SHOW_PERFORMANCE    = 1 and not RUN_IN_BATCH
    PLOT_PERFORMANCE    = 1 and not RUN_IN_BATCH
    LOG_INTERVAL        = 1 if not RUN_IN_BATCH else 60 # if make_exp_name = false then we assume the job is running on the cluster hence increase the intervals between logs to reduce output txt size 
    JOB_ID              = 1 if jobID == -1 else jobID
    PROJECT_PATH        = '.' if PROJECT_PATH == None else PROJECT_PATH
    DEBUG               = 0
    logger              = Logger()
    MAX_ITERATIONS      = 10
    # Domain ----------------------
    #MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    MAZE                = '/Domains/GridWorldMaps/4x5.txt'
    NOISE               = 0.3   # Noise parameters used for some of the domains such as the GridWorld
    # Representation ----------------------
    DISCRITIZATION              = 20    # Number of bins used to discritize each continuous dimension. Used for some representations 
    # Policy ----------------------
    EPSILON                 = .2 # EGreedy Often is .1 CHANGE ME <<<
    #Agent ----------------------
    alpha_decay_mode        = 'boyan' # Decay rate parameter; See Agent.py initialization for more information
    initial_alpha           = .1
    boyan_N0                = 100
    LAMBDA                  = 0
    # DOMAIN
    #=================
    #domain          = GridWorld(RL_PYTHON_ROOT+'/'+MAZE, noise = NOISE, logger = logger)
    domain          = Pendulum_InvertedBalance(episodeCap = 300, logger = logger);
    
    # REPRESENTATION
    #================
    representation  = Tabular(domain,logger,discretization = DISCRITIZATION) # Optional parameter discretization, for continuous domains
    
    # POLICY
    #================
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    
    # LEARNING AGENT
    #================
    agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    
    experiment      = OnlineExperiment(agent,domain,logger,exp_naming = EXPERIMENT_NAMING, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)
    
#    for x in range(10):
#        print('%0.10f'%random.rand()) 
    
    experiment.run()
    experiment.save()
    
if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         main(jobID = -1,PROJECT_PATH = 'Results/IShouldRun', MAKE_EXP_NAME = 1)
     else: # Batch Mode through command line
         main(int(sys.argv[1]),sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
