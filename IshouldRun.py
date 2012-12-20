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

PERFORMANCE_CHECKS  = 10
LEARNING_STEPS      = 10000
SHOW_ALL            = 0
SHOW_PERFORMANCE    = 1
PLOT_PERFORMANCE    = 1
LOG_INTERVAL        = 1 
JOB_ID              = 1
PROJECT_PATH        = 'Results/TempProject' 
logger              = Logger()
MAZE                = '/Domains/PitmazeMaps/4x5.txt'
NOISE               = 0.3
EPSILON             = .1 # Used for EGreedy
initial_alpha       = .1

domain          = PitMaze(MAZE, noise = NOISE, logger = logger)
representation  = Tabular(domain,logger,discretization = 20) # Optional parameter discretization, for continuous domains
policy          = eGreedy(representation,logger, epsilon = EPSILON)
agent           = SARSA(representation,policy,domain,logger,initial_alpha)
experiment      = OnlineExperiment(agent,domain,logger, id = JOB_ID, max_steps = LEARNING_STEPS,show_all= SHOW_ALL, performanceChecks = PERFORMANCE_CHECKS, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL,project_path = PROJECT_PATH, plot_performance =  PLOT_PERFORMANCE)

experiment.run()
pl.ioff(); pl.show()

