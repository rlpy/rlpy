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
    LEARNING_STEPS      = 5000
    SHOW_ALL            = 0
    SHOW_PERFORMANCE    = 0
    LOG_INTERVAL        = 1 
    RESULT_FILE         = 'result.txt'
    JOB_ID              = 1 if jobID == -1 else jobID
    SHOW_FINAL_PLOT     = 1 if SHOW_FINAL_PLOT == -1 else SHOW_FINAL_PLOT 
    DEBUG               = 0
    # Domain ----------------------
    MAZE                = '/Domains/PitMazeMaps/4x5.txt'
    #MAZE                = '/Domains/PitMazeMaps/11x11-Rooms.txt'
    NOISE               = .3
    BLOCKS              = 4 # For BlocksWorld
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
    
    #domain          = ChainMDP(10)
    domain          = PitMaze(MAZE, noise = NOISE)
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE)
    #domain          = MountainCar(noise = NOISE)
    #domain          = NetworkAdmin()
    #domain          = PST() 
    
    representation  = Tabular(domain)
    #representation  = IncrementalTabular(domain)
    #representation  = iFDD(domain,iFDD_Threshold,useCache=iFDD_CACHED,maxBatchDicovery = iFDDMaxBatchDicovery, batchThreshold = iFDD_BatchThreshold)
    #representation  = IndependentDiscretization(domain)
    #representation  = RBF(domain, rbfs = RBFS)
    
    policy          = eGreedy(representation, epsilon = EPSILON)
    #policy          = UniformRandom(representation)
    
    agent           = SARSA(representation,policy,domain,initial_alpha,LAMBDA)
    #agent           = LSPI(representation,policy,domain,LSPI_iterations,LSPI_windowSize)
    #agent           = iFDD_LSPI(representation,policy,domain,LSPI_iterations,LSPI_windowSize,iFDD_LSPI_iterations)
    
    experiment      = OnlineExperiment(agent,domain,id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL)
    
    if DEBUG:
        domain.printAll()
        representation.printAll()
        policy.printAll()
        agent.printAll() 
    
    experiment.run()
    experiment.save(RESULT_FILE)
    if SHOW_FINAL_PLOT: pl.show()

if __name__ == '__main__':
     main()