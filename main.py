######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
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
SHOW_PERFORMANCE    = 0 
RESULT_FILE         = 'result.txt'
# Domain
#----------------------
MAZE                = '/Domains/PitMazeMaps/4by5.txt'
NOISE               = 0
# Representation
#----------------------
RBFS = 9
# Policy
#----------------------
EPSILON             = .1 # EGreedy
#Agent
#----------------------
initial_alpha       = .1
learn_step          = 10000

domain          = PitMaze(MAZE, noise = NOISE)
representation  = Tabular(domain)
#representation  = IndependentDiscretization(domain)
# srepresentation  = RBF(domain, rbfs = RBFS)
policy          = eGreedy(representation, epsilon = EPSILON)
agent           = SARSA(representation,policy,domain,initial_alpha)
experiment      = OnlineExperiment(agent,domain,max_steps = learn_step,show_all= SHOW_ALL, show_performance = SHOW_PERFORMANCE)

if DEBUG:
    domain.printAll()
    representation.printAll()
    policy.printAll()
    agent.printAll() 

experiment.run()
experiment.save(RESULT_FILE)
pl.show()
#domain = Domains.PitMaze.PitMaze()
#domain = PitMaze()
#domain.show(1,1)

#domain.show()
#np.test('full')
#import scipy
#scipy.test()
    
#agent = SARSA;
#domain = BlocksWorld;
#experiment = Experiment(agent,domain)
#experiment.run();

 