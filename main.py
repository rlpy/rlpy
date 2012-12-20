#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################

from Tools import *
from Domains import *
from Agents import *
from Representations import *
from Policies import *
from Experiments import *
#from pandas.tests.test_series import CheckNameIntegration

def main(jobID=-1,              # Used as an indicator for each run of the algorithm
         PROJECT_PATH ='',      # Path to store the results. Notice that a directory is automatically generated within this directory based on the selection of domain,agent,representation, 
         SHOW_FINAL_PLOT=0):    # Draw the final plot when the run is finished? Automatically set to False if jobID == -1

    # Etc
    #----------------------
    PERFORMANCE_CHECKS  = 10
    LEARNING_STEPS      = 10000
    EXPERIMENT_NAMING   = ['domain','agent','representation']
    #EXPERIMENT_NAMING   = ['domain','representation','max_steps','representation.batchThreshold']
    RUN_IN_BATCH        = jobID != -1
    SHOW_ALL            = 0 and not RUN_IN_BATCH
    SHOW_PERFORMANCE    = 1 and not RUN_IN_BATCH
    PLOT_PERFORMANCE    = 1 and not RUN_IN_BATCH
    LOG_INTERVAL        = 1 
    JOB_ID              = 1 if jobID == -1 else jobID
    PROJECT_PATH        = 'Results/TempProject' if PROJECT_PATH == '' else PROJECT_PATH
    DEBUG               = 0
    logger              = Logger()
    MAX_ITERATIONS      = 10
    # Domain ----------------------
    #MAZE                = '/Domains/PitmazeMaps/1x3.txt'
    MAZE                = '/Domains/PitmazeMaps/4x5.txt'
    INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/4x4_1A_1I.txt'
    #MAZE                = '/Domains/PitMazeMaps/11x11-Rooms.txt'
    NOISE               = 0.3   # Noise parameters used for some of the domains such as the pitmaze
    BLOCKS              = 4     # Number of blocks for the BlocksWorld domain
    # Representation ----------------------
    #RBFS                        = 9
    DISCRITIZATION              = 20    # Number of bins used to discritize each continuous dimension. Used for some representations 
    RBFS                        = {'PitMaze':10, 'CartPole':20, 'BlocksWorld':100,
                                'NetworkAdmin':500, 'PST':1000} # Values used in tutorial
    iFDD_Threshold              = 0.1  
    #iFDD_Threshold              = .001  # Good for Inverted Pendulum
    #iFDD_Threshold              = .05 # Good for bloackWorld #10 good for NetworkAdmin
    iFDD_Sparsify               = 1     # Sparsify the output feature vectors at iFDD? 
    FeatureExpandThreshold      = .2    # Minimum relevance required for representation expansion techniques to add a feature 
    iFDD_CACHED                 = 1     # Results will remiain IDENTICAL, but often faster
    Max_Batch_Feature_Discovery = 1     # Maximum Number of Features discovered on each iteration in the batch mode of iFDD
    FourierOrder                = 3     # 
    # Policy ----------------------
    EPSILON                 = .1 # EGreedy
    #Agent ----------------------
    initial_alpha           = .1
    LAMBDA                  = 0
    LSPI_iterations         = 10
    LSPI_windowSize         = LEARNING_STEPS/PERFORMANCE_CHECKS
    LSPI_WEIGHT_DIFF_TOL    = 1e-3 # Minimum Weight Difference required to keep the LSPI loop going
    RE_LSPI_iterations      = 100
    
    #domain          = ChainMDP(10, logger = logger)
    #domain          = PitMaze(MAZE, noise = NOISE, logger = logger)
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE, logger = logger)
    #domain          = MountainCar(noise = NOISE,logger = logger)
    #domain          = NetworkAdmin(networkmapname='/Domains/NetworkAdminMaps/5Machines.txt',maptype='eachNeighbor',numNodes=5,logger = logger)
    #domain          = PST(NUM_UAV = 2, motionNoise = 0,logger = logger)
    #domain          = IntruderMonitoring(INTRUDERMAP,logger)
    #domain          = Pendulum_InvertedBalance(logger = logger);
    domain          = Pendulum_SwingUp(logger = logger);
    #domain          = CartPole_InvertedBalance(logger = logger);
    #domain          = CartPole_SwingUp(logger = logger);
    
    #representation  = Tabular(domain,logger,discretization = DISCRITIZATION) # Optional parameter discretization, for continuous domains
    #representation  = IncrementalTabular(domain,logger)
    representation  = iFDD(domain,logger,iFDD_Threshold,sparsify = iFDD_Sparsify,discretization = DISCRITIZATION,useCache=iFDD_CACHED,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = FeatureExpandThreshold)
    #representation  = IndependentDiscretization(domain,logger, discretization = DISCRITIZATION)
    #representation  = RBF(domain,logger, rbfs = RBFS['PitMaze'])
    #representation  = Fourier(domain,logger,order=FourierOrder)
    
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    
    agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA)
    #agent           = LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize)
    #agent           = RE_LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations)
    #agent           =  Q_LEARNING(representation,policy,domain,logger)
    experiment      = OnlineExperiment(agent,domain,logger,exp_naming = EXPERIMENT_NAMING, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)
    
    if DEBUG:
        domain.printAll()
        representation.printAll()
        policy.printAll()
        agent.printAll()
    
    experiment.run()
    experiment.save()
    #domain.showLearning(representation)
    if SHOW_FINAL_PLOT: pl.ioff(); pl.show()

if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         main(jobID = -1,PROJECT_PATH = 'Results/Test_Project',SHOW_FINAL_PLOT = True)
     else: # Batch Mode through command line
         main(int(sys.argv[1]),sys.argv[2])
