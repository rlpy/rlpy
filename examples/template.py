"""Main script for running different combinations of Agent, Domain, and Experiment."""

from Tools import *
from Domains import *
from Agents import *
from Representations import *
from Policies import *
from Experiments import *
from MDPSolvers import *
try:
    from ROS.ROS_RCCar import ROS_RCCar
except Exception:
    print "ROS is not installed => ROS_RCCar can not be used."
    pass


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"

visualize_steps = False # show each steps
visualize_learning = True # show visualizations of the learning progress, e.g. value function
visualize_performance = False # show performance runs

def make_experiment(id=1, path="./Results/Temp"):
    logger              = Logger()
    MAX_ITERATIONS = 10
    max_steps = 10000
    performanceChecks = 20
    vis_learning_freq = 5000
    checks_per_policy = 1

    # Domain Arguments----------------------
    #MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    #MAZE                = '/Domains/GridWorldMaps/2x3.txt'
    #MAZE                = '/Domains/GridWorldMaps/4x5.txt'
    MAZE                = '/Domains/GridWorldMaps/10x10-12ftml.txt'
    #MAZE                = '/Domains/GridWorldMaps/11x11-Rooms.txt'
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
    NOISE               = .3   # Noise parameters used for some of the domains such as the GridWorld
    BLOCKS              = 6     # Number of blocks for the BlocksWorld domain
    # Representation ----------------------
    DISCRITIZATION              = 9 # CHANGE ME TO 20 # Number of bins used to discritize each continuous dimension. Used for some representations, Suggestion: 30 for Acrobot, 20 for other domains
    RBFS                        = 200  #{'GridWorld':10, 'CartPole':20, 'BlocksWorld':100, 'SystemAdministrator':500, 'PST':500, 'InfCartPoleBalance': 20 } # Values used in tutorial RBF was 1000 though but it takes 13 hours time to run
    iFDDOnlineThreshold         =	1e7 # Edited by makexp.py script
    BatchDiscoveryThreshold     =	0.05 # Edited by makexp.py script
    #BEBFNormThreshold           = #CONTROL:{'BlocksWorld':0.005, 'InfCartPoleBalance':0.20}  # If the maximum norm of the td_errors is less than this value, representation expansion halts until the next LSPI iteration (if any).
    iFDD_CACHED                 = 1 # Results will remain IDENTICAL, but often faster
    Max_Batch_Feature_Discovery = 1 # Maximum Number of Features discovered on each iteration in the batch mode of iFDD
    BEBF_svm_epsilon            = .1 #{'BlocksWorld':0.0005,'InfCartPoleBalance':0.1} # See BEBF; essentially the region in which no penalty is applied for training
    FourierOrder                = 3     #
    iFDD_Sparsify               = 1     # Should be on for online and off for batch methods. Sparsify the output feature vectors at iFDD? [wont make a difference for 2 dimensional spaces.
    iFDD_Plus                   = 1     # True: relevance = abs(TD_Error)/norm(feature), False: relevance = sum(abs(TD_error)) [ICML 11]
    OMPTD_BAG_SIZE              = 0
    # Policy ----------------------
    EPSILON                 = 0.1 # EGreedy Often is .1 CHANGE ME if I am not .1<<<
    #Agent ----------------------
    alpha_decay_mode        = 'boyan' # Boyan works better than dabney in some large domains such as pst. Decay rate parameter; See Agent.py initialization for more information
    initial_alpha           =	.1 # Edited by makexp.py script
    boyan_N0                =	100000 # Edited by makexp.py script
    BetaCoef                = 1e-6# In the Greedy_GQ Algorithm the second learning rate, Beta, is assumed to be Alpha * THIS CONSTANT
    LAMBDA                  = 0.
    LSPI_iterations         = 5 if not 'LSPI_iterations' in globals() else LSPI_iterations  #Maximum Number of LSPI Iterations
    LSPI_windowSize         = max_steps / performanceChecks
    LSPI_WEIGHT_DIFF_TOL    = 1e-3 # Minimum Weight Difference required to keep the LSPI loop going
    RE_LSPI_iterations      = 50
    LSPI_return_best_policy = False # Track the best policy through LSPI iterations using Monte-Carlo simulation. It uses extra samples to evaluate the policy
    LSPI_use_sparse         = True # Represet large matrices such as A and all_phi with sparse matrices.

    #Policy Evaluation Parameters: ----------------------
    PolicyEvaluation_test_samples   = 10000 # Number of samples used for calculating the accuracy of a value function
    PolicyEvaluation_MC_samples     = 100  # Number of Monte-Carlo samples used to estimate Q(s,a) of the policy
    PolicyEvaluation_LOAD_PATH      = 'Policies/FixedPolicyEvaluations' # Directory path to load the base of policy evaluations (True Q(s,a))

    # MDP Solver Parameters: ----------------------
    NS_SAMPLES                  = 100   # Number of Samples used to estimate the expected r+\gamma*V(s') given a state and action.
    PLANNING_TIME               = 60*60*2 #2 Hours = Max Amount of time given to each planning algorithm (MDP Solver) to think in seconds
    CONVERGENCE_THRESHOLD       = 1e-3  # Parameter used to define convergence in the planner
    MAX_PE_ITERATIONS           = 10    # Maximum number of sweeps used for PE

    # DOMAIN
    #=================
    #domain          = ChainMDP(10, logger = logger)
    domain          = GridWorld('./'+MAZE, noise = NOISE, logger = logger)
    #domain          = HelicopterHover(logger=logger)
    #domain          = Acrobot(logger=logger)
    #domain          = InfCartPoleBalance(logger = logger);
    #domain          = MountainCar(noise = NOISE,logger = logger)
    #domain          = BlocksWorld(blocks=BLOCKS,noise = NOISE, logger = logger)
    #domain          = PuddleWorld(logger=logger)
    #domain          = ROS_RCCar(logger=logger)
    #domain          = SystemAdministrator(networkmapname='./'+NETWORKNMAP,logger = logger)
    #domain          = Acrobot(logger = logger)
    #domain          = PST(NUM_UAV = 4, motionNoise = 0,logger = logger)
    #domain          = IntruderMonitoring('./'+INTRUDERMAP,logger)
    #domain          = InfCartPoleSwingUp(logger = logger)
    #domain          = CartPole_InvertedBalance(logger = logger)
    #domain          = CartPole_SwingUp(logger = logger)
    #domain          = FiftyChain(logger = logger)
    #domain          = RCCar(logger = logger)
    #domain           = Pinball(logger, 500, width=500, height=500, configuration='Domains/PinballConfigs/pinball_medium.cfg')
    #domain           = HIVTreatment(logger=logger)
    
    # REPRESENTATION
    #================
    initial_rep     = IndependentDiscretizationCompactBinary(domain,logger, discretization = DISCRITIZATION)
    #initial_rep     = IndependentDiscretization(domain,logger, discretization = DISCRITIZATION)

    #representation  =  initial_rep
    #representation  = IndependentDiscretizationCompactBinary(domain,logger, discretization = DISCRITIZATION)
    #representation  = IndependentDiscretization(domain,logger, discretization = DISCRITIZATION)
    #representation  = Tabular(domain,logger,discretization = DISCRITIZATION) # Optional parameter discretization, for continuous domains
    #representation  = IncrementalTabular(domain,logger)
    representation  = iFDD(domain,logger,iFDDOnlineThreshold,initial_rep,sparsify = iFDD_Sparsify,discretization = DISCRITIZATION,useCache=iFDD_CACHED,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, iFDDPlus = iFDD_Plus)
    #representation  = RBF(domain,logger, rbfs = RBFS, id = JOB_ID)
    #representation  = Fourier(domain,logger,order=FourierOrder)
    #representation  = BEBF(domain,logger, batchThreshold=BatchDiscoveryThreshold, svm_epsilon=BEBF_svm_epsilon)
    #representation  = OMPTD(domain,logger, initial_representation = initial_rep, discretization = DISCRITIZATION,maxBatchDicovery = Max_Batch_Feature_Discovery, batchThreshold = BatchDiscoveryThreshold, bagSize = OMPTD_BAG_SIZE, sparsify = iFDD_Sparsify)

    #tile_matrix = array(mat("""
    #72 72 84 84             ;
    #1 18 18 18; 18 1 18 18; 18 18 1 18; 18 18 18 1;
    #1 1 12 12; 1 12 12 1; 1 12 1 12; 12 1 1 12; 12 1 12 1; 12 12 1 1;
    #1 1 1 18; 1 1 18 1; 1 18 1 1; 18 1 1 1"""), dtype="float")
    #tile_matrix[tile_matrix == 1] = 0.5
    #representation = TileCoding(logger=logger, domain=domain,
    #        resolution_matrix=tile_matrix, safety="none",
    #        num_tilings=[12, 3, 3, 3, 3]+[2]*6+[3]*4
    #                            ,memory=15000)

    # POLICY
    #================

    #policy_representation  = RBF(domain,logger, rbfs = RBFS, id = JOB_ID)
    #policy_representation = Tabular(domain, logger, discretization=DISCRITIZATION)
    #policy = GibbsPolicy(policy_representation, logger)
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    #policy          = UniformRandom(representation,logger)
    #policy          = FixedPolicy(representation,logger) # Use this together with the PolicyEvaluation agent

    # LEARNING AGENT
    #================
    #agent = NaturalActorCritic(representation, policy, domain, logger,
    #                           forgetting_rate=.8, min_steps_between_updates=500,
    #                           max_steps_between_updates=5000,
    #                           lam=LAMBDA, alpha=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    #agent           = Q_LEARNING(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    agent           = Greedy_GQ(representation, policy, domain,logger, initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0, BetaCoef)
    #agent           = LSPI(representation,policy,domain,logger,LEARNING_STEPS, LEARNING_STEPS/PERFORMANCE_CHECKS, LSPI_iterations, epsilon = LSPI_WEIGHT_DIFF_TOL, return_best_policy = LSPI_return_best_policy,re_iterations = RE_LSPI_iterations, use_sparse = LSPI_use_sparse)
    #agent           = LSPI_SARSA(representation,policy,domain,logger,LSPI_iterations,LSPI_windowSize,LSPI_WEIGHT_DIFF_TOL,RE_LSPI_iterations,initial_alpha,LAMBDA,alpha_decay_mode, boyan_N0)
    #agent           = PolicyEvaluation(representation,policy,domain,logger,LEARNING_STEPS, PolicyEvaluation_test_samples,PolicyEvaluation_MC_samples,PolicyEvaluation_LOAD_PATH, re_iterations = RE_LSPI_iterations); PERFORMANCE_CHECKS  = 1 # Because policy evaluation in one run, create the whole state matrix, having multiple checks make the program confused.

    # MDP_Solver
    #================
    #MDPsolver = ValueIteration(JOB_ID,representation,domain,logger, ns_samples= NS_SAMPLES, project_path = PROJECT_PATH, show = visualize_learning, convergence_threshold = CONVERGENCE_THRESHOLD, planning_time = PLANNING_TIME)
    #MDPsolver = PolicyIteration(JOB_ID,representation,domain,logger, ns_samples= NS_SAMPLES, project_path = PROJECT_PATH, show = visualize_learning, convergence_threshold = CONVERGENCE_THRESHOLD, planning_time = PLANNING_TIME, max_PE_iterations = MAX_PE_ITERATIONS)
    #MDPsolver = TrajectoryBasedValueIteration(JOB_ID,representation,domain,logger, ns_samples= NS_SAMPLES, project_path = PROJECT_PATH, show = visualize_learning, convergence_threshold = CONVERGENCE_THRESHOLD, epsilon = EPSILON, planning_time = PLANNING_TIME)
    #MDPsolver = TrajectoryBasedPolicyIteration(JOB_ID,representation,domain,logger, ns_samples= NS_SAMPLES, project_path = PROJECT_PATH, show = visualize_learning, convergence_threshold = CONVERGENCE_THRESHOLD, epsilon = EPSILON, planning_time = PLANNING_TIME, max_PE_iterations = MAX_PE_ITERATIONS)

    # beware: very very ugly...
    if "MDPsolver" in locals():
        return MDPsolver
    experiment = Experiment(**locals())
    return experiment
#===============================================================
if __name__ == '__main__':
    something = make_experiment(1) #use ID 1 by default
    if isinstance(something, Experiment):
        experiment = something
        #from Tools.run import run_profiled
        #run_profiled(make_experiment)
        experiment = make_experiment(1)
        experiment.run(visualize_steps=visualize_steps,
                       visualize_learning=visualize_learning,
                       visualize_performance=visualize_performance)
        experiment.plot()
        experiment.save()
    else:
        # MDP Solver
        something.solve()
        if visualize_performance:
            pl.ioff()
            pl.show()
