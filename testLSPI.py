#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
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
         MAKE_EXP_NAME = 1      # This flag should be set 0 if the job is submitted through the condor cluster so no extra directory is built. Basically all the results are stored in the directory where the main file is.
         ):

    # Etc
    #----------------------
    PERFORMANCE_CHECKS  = 1
    LEARNING_STEPS      = 10000
    #EXPERIMENT_NAMING   = ['domain','agent','representation']
    EXPERIMENT_NAMING   = ['domain','representation','max_steps','representation.batchThreshold'] 
    EXPERIMENT_NAMING   = [] if not MAKE_EXP_NAME else EXPERIMENT_NAMING
    RUN_IN_BATCH        = jobID != -1
    SHOW_ALL            = 0 and not RUN_IN_BATCH
    SHOW_PERFORMANCE    = 0 and not RUN_IN_BATCH
    PLOT_PERFORMANCE    = 0 and not RUN_IN_BATCH
    LOG_INTERVAL        = 1 if not RUN_IN_BATCH else 60 # if make_exp_name = false then we assume the job is running on the cluster hence increase the intervals between logs to reduce output txt size 
    JOB_ID              = 1 if jobID == -1 else jobID
    PROJECT_PATH        = '.' if PROJECT_PATH == None else PROJECT_PATH
    DEBUG               = 0
    logger              = Logger()
    MAX_ITERATIONS      = 10
    # Domain ----------------------
    #MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    MAZE                = '/Domains/GridWorldMaps/4x5.txt'
    #MAZE                = '/Domains/GridWorldMaps/3x3.txt'
    INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/4x4_1A_1I.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/5Machines.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/9Star.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/10Machines.txt'
    NETWORKNMAP         = '/Domains/SystemAdministratorMaps/16-5Branches.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/20MachTutorial.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/20Ring.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/10Ring.txt'
    NOISE               = 0.3   # Noise parameters used for some of the domains such as the GridWorld
    BLOCKS              = 4     # Number of blocks for the BlocksWorld domain
    # Representation ----------------------
    DISCRITIZATION              = 20    # Number of bins used to discritize each continuous dimension. Used for some representations 
    RBFS                        = {'GridWorld':10, 'CartPole':20, 'BlocksWorld':100,
                                'SystemAdministrator':500, 'PST':1000, 'Pendulum_InvertedBalance': 9 } # Values used in tutorial
    iFDDOnlineThreshold         = inf #{'Pendulum':.001, 'BlocksWorld':.05, 'SystemAdministrator':10} 
    BatchDiscoveryThreshold     = .6 if not 'BatchDiscoveryThreshold' in globals() else BatchDiscoveryThreshold  # Minimum relevance required for representation expansion techniques to add a feature 
    iFDD_CACHED                 = 1     # Results will remain IDENTICAL, but often faster
    Max_Batch_Feature_Discovery = 20    # Maximum Number of Features discovered on each iteration in the batch mode of iFDD
    BEBFNormThreshold           = {'BlocksWorld':0.005, 'Pendulum_InvertedBalance':0.20}  # If the maximum norm of the td_errors is less than this value, representation expansion halts until the next LSPI iteration (if any).
    BEBF_svm_epsilon            = {'BlocksWorld':0.0005,'Pendulum_InvertedBalance':0.1} # See BEBF; essentially the region in which no penalty is applied for training
    FourierOrder                = 3     # 
    iFDD_Sparsify               = 0     # Should be on for online and off for batch methods. Sparsify the output feature vectors at iFDD? [wont make a difference for 2 dimensional spaces.
    iFDD_Plus                   = 1     # True: relevance = abs(TD_Error)/norm(feature), False: relevance = sum(abs(TD_error)) [ICML 11]  
    OMPTD_BAG_SIZE              = 10000
    # Policy ----------------------
    EPSILON                 = .1 # EGreedy
    #Agent ----------------------
    alpha_decay_mode        = 'boyan' # Decay rate parameter; See Agent.py initialization for more information
    initial_alpha           = .1
    boyan_N0                = 100
    LAMBDA                  = 0
    LSPI_iterations         = 2 if not 'LSPI_iterations' in globals() else LSPI_iterations  #Maximum Number of LSPI Iterations
    LSPI_windowSize         = LEARNING_STEPS/PERFORMANCE_CHECKS
    LSPI_WEIGHT_DIFF_TOL    = 1e-3 # Minimum Weight Difference required to keep the LSPI loop going
    RE_LSPI_iterations      = 2
    
    # DOMAIN
    #================
    #domain          = ChainMDP(10, logger = logger)
    #domain          = GridWorld(RL_PYTHON_ROOT+'/'+MAZE, noise = NOISE, logger = logger)
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE, logger = logger)
    #domain          = MountainCar(noise = NOISE,logger = logger)
    #domain          = SystemAdministrator(networkmapname=RL_PYTHON_ROOT+'/'+NETWORKNMAP,logger = logger)
    #domain          = PST(NUM_UAV = 2, motionNoise = 0,logger = logger)
    #domain          = IntruderMonitoring(RL_PYTHON_ROOT+'/'+INTRUDERMAP,logger)
    domain          = Pendulum_InvertedBalance(logger = logger);
    #domain          = Pendulum_SwingUp(logger = logger);
    #domain          = CartPole_InvertedBalance(logger = logger);
    #domain          = CartPole_SwingUp(logger = logger);
    #domain          = FiftyChain(logger = logger)
    
    # REPRESENTATION
    #================
    #initial_rep     = IndependentDiscretizationCompactBinary(domain,logger, discretization = DISCRITIZATION)
    initial_rep     = IndependentDiscretization(domain,logger, discretization = DISCRITIZATION)
    #representation  = initial_rep
    #representation  = Tabular(domain,logger,discretization = DISCRITIZATION) # Optional parameter discretization, for continuous domains
    #representation  = IncrementalTabular(domain,logger)
    #representation  = IndependentDiscretizationCompactBinary(domain,logger, discretization = DISCRITIZATION)
    #representation  = RBF(domain,logger, rbfs = RBFS[className(domain)])
    #representation  = Fourier(domain,logger,order=FourierOrder)
    #representation  = BEBF(domain,logger, batchThreshold=BEBFNormThreshold[className(domain)], svm_epsilon=BEBF_svm_epsilon[className(domain)])
    representation  = iFDD(domain,logger,iFDDOnlineThreshold,initial_rep,sparsify = iFDD_Sparsify,discretization = DISCRITIZATION,useCache=iFDD_CACHED,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, iFDDPlus = iFDD_Plus)
    #representation  = OMPTD(domain,logger, initial_representation = initial_rep, discretization = DISCRITIZATION,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, bagSize = OMPTD_BAG_SIZE, sparsify = iFDD_Sparsify)
    
    # POLICY
    #================
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    
    # LEARNING AGENT
    #================
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    #agent           = LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize)
    agent           = RE_LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations)
    #agent           = RE_LSPI_SARSA(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations,initial_alpha,LAMBDA,alpha_decay_mode, boyan_N0)
    #agent           =  Q_LEARNING(representation,policy,domain,logger)
    
    experiment      = OnlineExperiment(agent,domain,logger,exp_naming = EXPERIMENT_NAMING, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)
    
#    for x in range(10):
#        print('%0.10f'%random.rand()) 
    
    experiment.run()
    experiment.save()
    
if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         main(jobID = -1,PROJECT_PATH = 'Results/Temp', MAKE_EXP_NAME = 1)
     else: # Batch Mode through command line
         main(int(sys.argv[1]),sys.argv[2], int(sys.argv[4]))
