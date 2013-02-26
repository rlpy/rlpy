#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################

import sys, os
#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RL-Python/Tools.py'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'

RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RL-Python')
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
         SHOW_FINAL_PLOT = 1,   # Draw the final plot when the run is finished? Automatically set to False if jobID == -1
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
    #MAZE                = '/Domains/PitmazeMaps/1x3.txt'
    MAZE                = '/Domains/PitmazeMaps/4x5.txt'
    #MAZE                = '/Domains/PitmazeMaps/3x3.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/1x3_1A_1I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/2x2_1A_1I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/2x3_1A_1I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/2x3_2A_1I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/3x3_2A_2I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/4x4_1A_1I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/5x5_2A_2I.txt'
    INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/5x5_2A_2I.txt'
    #INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/5x5_4A_3I.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/5Machines.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/9Star.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/10Machines.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/16-5Branches.txt'
    NETWORKNMAP         = '/Domains/SystemAdministratorMaps/20MachTutorial.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/20Ring.txt'
    #NETWORKNMAP         = '/Domains/SystemAdministratorMaps/10Ring.txt'
    NOISE               = 0.3   # Noise parameters used for some of the domains such as the pitmaze
    BLOCKS              = 6     # Number of blocks for the BlocksWorld domain
    # Representation ----------------------
    DISCRITIZATION              = 20    # Number of bins used to discritize each continuous dimension. Used for some representations 
    RBFS                        = {'PitMaze':10, 'CartPole':20, 'BlocksWorld':100,
                                'SystemAdministrator':500, 'PST':1000, 'Pendulum_InvertedBalance': 9 } # Values used in tutorial
    iFDDOnlineThreshold         = 5 #{'Pendulum':.001, 'BlocksWorld':.05, 'SystemAdministrator':10} 
    BatchDiscoveryThreshold     = 1e-4 if not 'BatchDiscoveryThreshold' in globals() else BatchDiscoveryThreshold  # Minimum relevance required for representation expansion techniques to add a feature 
    #BEBFNormThreshold           = #CONTROL:{'BlocksWorld':0.005, 'Pendulum_InvertedBalance':0.20}  # If the maximum norm of the td_errors is less than this value, representation expansion halts until the next LSPI iteration (if any).
    iFDD_CACHED                 = 1 # Results will remain IDENTICAL, but often faster
    Max_Batch_Feature_Discovery = 1 # Maximum Number of Features discovered on each iteration in the batch mode of iFDD
    BEBF_svm_epsilon            = {'BlocksWorld':0.0005,'Pendulum_InvertedBalance':0.1} # See BEBF; essentially the region in which no penalty is applied for training
    FourierOrder                = 3     # 
    iFDD_Sparsify               = 0     # Should be on for online and off for batch methods. Sparsify the output feature vectors at iFDD? [wont make a difference for 2 dimensional spaces.
    iFDD_Plus                   = 1     # True: relevance = abs(TD_Error)/norm(feature), False: relevance = sum(abs(TD_error)) [ICML 11]  
    OMPTD_BAG_SIZE              = 1000
    # Policy ----------------------
    EPSILON                 = .2 # EGreedy Often is .1 CHANGE ME <<<
    #Agent ----------------------
    alpha_decay_mode        = 'boyan' # Decay rate parameter; See Agent.py initialization for more information
    initial_alpha           = .1
    boyan_N0                = 100
    LAMBDA                  = 0
    LSPI_iterations         = 5 if not 'LSPI_iterations' in globals() else LSPI_iterations  #Maximum Number of LSPI Iterations
    LSPI_windowSize         = LEARNING_STEPS/PERFORMANCE_CHECKS
    LSPI_WEIGHT_DIFF_TOL    = 1e-3 # Minimum Weight Difference required to keep the LSPI loop going
    RE_LSPI_iterations      = 100
    LSPI_return_best_policy = False # Track the best policy through LSPI iterations using Monte-Carlo simulation. It uses extra samples to evaluate the policy
    
    #Policy Evaluation Parameters:
    PolicyEvaluation_test_samples   = 10000 # Number of samples used for calculating the accuracy of a value function
    PolicyEvaluation_MC_samples     = 100  # Number of Monte-Carlo samples used to estimate Q(s,a) of the policy 
    PolicyEvaluation_LOAD_PATH      = RL_PYTHON_ROOT + '/Policies/FixedPolicyEvaluations' # Directory path to load the base of policy evaluations (True Q(s,a)) 
    #Load Main Setting and override any parameter
    #    if os.path.exists('mainSetting.py'):
    #        print 'Incorporating mainSetting.py'
    #        from mainSetting import *
    # Python does not allow import * in functions... so this can not work this way

    # DOMAIN
    #=================
    #domain          = ChainMDP(10, logger = logger)
    #domain          = PitMaze(RL_PYTHON_ROOT+'/'+MAZE, noise = NOISE, logger = logger)
    #domain          = Pendulum_InvertedBalance(logger = logger);
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE, logger = logger)
    #domain          = MountainCar(noise = NOISE,logger = logger)
    #domain          = SystemAdministrator(networkmapname=RL_PYTHON_ROOT+'/'+NETWORKNMAP,logger = logger)
    #domain          = PST(NUM_UAV = 3, motionNoise = 0,logger = logger)
    domain          = IntruderMonitoring(RL_PYTHON_ROOT+'/'+INTRUDERMAP,logger)
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
    #representation  = BEBF(domain,logger, batchThreshold=BatchDiscoveryThreshold, svm_epsilon=BEBF_svm_epsilon[className(domain)])
    representation  = iFDD(domain,logger,iFDDOnlineThreshold,initial_rep,sparsify = iFDD_Sparsify,discretization = DISCRITIZATION,useCache=iFDD_CACHED,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, iFDDPlus = iFDD_Plus)
    #representation  = OMPTD(domain,logger, initial_representation = initial_rep, discretization = DISCRITIZATION,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, bagSize = OMPTD_BAG_SIZE, sparsify = iFDD_Sparsify)
    
    # POLICY
    #================
    #policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    policy          = FixedPolicy(representation,logger)
    
    # LEARNING AGENT
    #================
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    #agent           = LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize, LSPI_return_best_policy)
    #agent           = RE_LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations)
    #agent           = RE_LSPI_SARSA(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations,initial_alpha,LAMBDA,alpha_decay_mode, boyan_N0)
    #agent           =  Q_LEARNING(representation,policy,domain,logger)
    agent           = PolicyEvaluation(representation,policy,domain,logger,LSPI_windowSize, PolicyEvaluation_test_samples,PolicyEvaluation_MC_samples,PolicyEvaluation_LOAD_PATH)
    
    experiment      = OnlineExperiment(agent,domain,logger,exp_naming = EXPERIMENT_NAMING, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)
    experiment.run()
    experiment.save()
    
    #domain.showLearning(representation)
    if SHOW_FINAL_PLOT:
        if module_exists('matplotlib'):
            pl.ioff(); pl.show()
        else:
            logger.log('could not show final plot; no matplotlib')

if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         main(jobID = -1,PROJECT_PATH = 'Results/Temp',SHOW_FINAL_PLOT = 1, MAKE_EXP_NAME = 1)
     else: # Batch Mode through command line
         main(int(sys.argv[1]),sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
