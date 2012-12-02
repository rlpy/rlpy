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

def main(jobID=-1, SHOW_FINAL_PLOT=-1):

    # Etc
    #----------------------
    PERFORMANCE_CHECKS  = 1
    LEARNING_STEPS      = 10000
    SHOW_ALL            = 0
    SHOW_PERFORMANCE    = 0
    LOG_INTERVAL        = 1 
    RESULT_FILE         = 'result.txt'
    JOB_ID              = 1 if jobID == -1 else jobID
    SHOW_FINAL_PLOT     = 1 if SHOW_FINAL_PLOT == -1 else SHOW_FINAL_PLOT 
    DEBUG               = 0
    logger              = Logger('Results/%d/out.txt'%(JOB_ID))
    # Domain ----------------------
    MAZE                = '/Domains/PitMazeMaps/4x5.txt'
    #MAZE                = '/Domains/PitMazeMaps/11x11-Rooms.txt'
    NOISE               = .3
    BLOCKS              = 6 # For BlocksWorld
    # Representation ----------------------
    RBFS                    = 9
    iFDD_Threshold          = .05 # Good for bloackWorld
    iFDD_BatchThreshold     = .001 
    iFDD_CACHED             = 1
    iFDDMaxBatchDicovery    = 1
    # Policy ----------------------
    EPSILON             = .1 # EGreedy
    #Agent ----------------------
    initial_alpha           = .1
    LAMBDA                  = 0#.95
    LSPI_iterations         = 5
    LSPI_windowSize         = LEARNING_STEPS/PERFORMANCE_CHECKS
    iFDD_LSPI_iterations    = 10
    
    domain          = ChainMDP(logger,10)
    #domain          = PitMaze(logger,MAZE, noise = NOISE)
    #domain          = BlocksWorld(logger,blocks=BLOCKS,noise = NOISE)
    #domain          = MountainCar(logger,noise = NOISE)
    #domain          = NetworkAdmin(logger)
    #domain          = PST(logger) 
    
    #representation  = Tabular(domain,logger)
    #representation  = IncrementalTabular(domain,logger)
    representation  = iFDD(domain,logger,iFDD_Threshold,useCache=iFDD_CACHED,maxBatchDicovery = iFDDMaxBatchDicovery, batchThreshold = iFDD_BatchThreshold)
    #representation  = IndependentDiscretization(domain,logger)
    #representation  = RBF(domain,logger, rbfs = RBFS)
    
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    
    agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA)
    #agent           = LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize)
    #agent           = iFDD_LSPI(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,iFDD_LSPI_iterations)
    
    experiment      = OnlineExperiment(agent,domain,logger,id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL, output_filename = RESULT_FILE)
    
    if DEBUG:
        domain.printAll()
        representation.printAll()
        policy.printAll()
        agent.printAll() 
    
    experiment.run()
    experiment.save()
    logger.done()
    if SHOW_FINAL_PLOT: pl.show()

if __name__ == '__main__':
     if len(sys.argv) == 1:
         main()
     else:
         main(int(sys.argv[1]),0)
     
     
     
     
     