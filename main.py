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

def main(jobID=-1, OUT_PATH =-1, SHOW_FINAL_PLOT=0):

    # Etc
    #----------------------
    PERFORMANCE_CHECKS  = 10
    LEARNING_STEPS      = 10000
    RUN_IN_BATCH        = jobID != -1
    SHOW_ALL            = 0 and not RUN_IN_BATCH
    SHOW_PERFORMANCE    = 1 and not RUN_IN_BATCH
    PLOT_PERFORMANCE    = 1 and not RUN_IN_BATCH
    LOG_INTERVAL        = 1 
    RESULT_FILE         = 'result.txt'
    STDOUT_FILE         = 'out.txt'
    JOB_ID              = 1 if jobID == -1 else jobID
    OUT_PATH            = 'Results/Temp' if OUT_PATH == -1 else OUT_PATH
    DEBUG               = 0
    logger              = Logger('%s/%d-%s'%(OUT_PATH,JOB_ID,STDOUT_FILE))
    MAX_ITERATIONS      = 10
    # Domain ----------------------
    MAZE                = '/Domains/PitmazeMaps/4x5.txt'
    INTRUDERMAP         = '/Domains/IntruderMonitoringMaps/4x4_1A_1I.txt'
    #MAZE                = '/Domains/PitMazeMaps/11x11-Rooms.txt'
    NOISE               = 0.3
    BLOCKS              = 4 # For BlocksWorld
    # Representation ----------------------
    #RBFS                    = 9
    RBFS                    = {'PitMaze':10, 'CartPole':20, 'BlocksWorld':100,
                                'NetworkAdmin':500, 'PST':1000} # Values used in tutorial
    iFDD_Threshold          = .05 # Good for bloackWorld #10 good for NetworkAdmin
    iFDD_BatchThreshold     = .001 
    iFDD_CACHED             = 1
    iFDDMaxBatchDicovery    = 1
    # Policy ----------------------
    EPSILON             = .1 # EGreedy
    #Agent ----------------------
    initial_alpha           = 1
    LAMBDA                  = 0
    LSPI_iterations         = 5
    LSPI_windowSize         = LEARNING_STEPS/PERFORMANCE_CHECKS
    iFDD_LSPI_iterations    = 10
    
    domain          = ChainMDP(10, logger = logger)
    #domain          = PitMaze(MAZE, noise = NOISE, logger = logger)
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE, logger = logger)
    #domain          = MountainCar(noise = NOISE,logger = logger)
    #domain          = NetworkAdmin(networkmapname='/Domains/NetworkAdminMaps/5Machines.txt',maptype='eachNeighbor',numNodes=5,logger = logger)
    #domain          = PST(NUM_UAV = 2, motionNoise = 0,logger = logger)
    #domain          = IntruderMonitoring(INTRUDERMAP,logger)
    #domain           = Pendulum_InvertedBalance(logger = logger);
    #domain           = Pendulum_SwingUp(logger = logger);
    #domain           = CartPole_InvertedBalance(logger = logger);
    #domain           = CartPole_SwingUp(logger = logger);
    
    representation  = Tabular(domain,logger,discretization = 20) # Optional parameter discretization, for continuous domains
    #representation  = IncrementalTabular(domain,logger)
    #representation  = iFDD(domain,logger,iFDD_Threshold,useCache=iFDD_CACHED,maxBatchDicovery = iFDDMaxBatchDicovery, batchThreshold = iFDD_BatchThreshold)
    #representation  = IndependentDiscretization(domain,logger)
    #representation  = RBF(domain,logger, rbfs = RBFS['PitMaze'])
    
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA)
    #agent           = LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize)
    #agent           = RE_LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,iFDD_LSPI_iterations)
    agent           =  Q_LEARNING(representation,policy,domain,logger)
    
    experiment      = OnlineExperiment(agent,domain,logger,id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,output_path = OUT_PATH, output_filename = RESULT_FILE, plot_performance =  PLOT_PERFORMANCE)
    
    if DEBUG:
        domain.printAll()
        representation.printAll()
        policy.printAll()
        agent.printAll() 
    
    experiment.run()
    experiment.save()
    #domain.showLearning(representation)
    logger.done()
    if SHOW_FINAL_PLOT: pl.ioff(); pl.show()

if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         main(jobID = -1,OUT_PATH = 'Results/Temp',SHOW_FINAL_PLOT = True)
     else: # Batch Mode through command line
         main(int(sys.argv[1]),sys.argv[2])
     
     
     
     
     