from Tools import *
from Domains import *
from Agents import *
from Representations import *
from Policies import *
from Experiments import *

# Etc
#----------------------
DEBUG               = 0
SHOW_ALL            = 0
SHOW_PERFORMANCE    = 1
LOG_INTERVAL        = 1 
RESULT_FILE         = 'result.txt'
learn_step          = 10000

domain          = PitMaze('/Domains/PitMazeMaps/4by5.txt')
representation  = Tabular(domain)
policy          = eGreedy(representation)
agent           = SARSA(representation,policy,domain)
experiment      = OnlineExperiment(agent,domain,max_steps = learn_step,show_all= SHOW_ALL, show_performance = SHOW_PERFORMANCE, log_interval = LOG_INTERVAL)

experiment.run()
experiment.save(RESULT_FILE)
pl.show()
