#!/usr/bin/env python
######################################################
# Robert H Klein, Aug. 14, 2013                      #
######################################################  

from Tools import *
from Domains import *
from Agents import *
from Representations import *
from Policies import *
from Experiments import *

#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.' 
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

visualize_steps = False # show each steps
visualize_learning = True # show visualizations of the learning progress, e.g. value function
visualize_performance = True # show performance runs

def make_experiment(id=1, path="./Results/Temp"):
    logger              = Logger()
    MAX_ITERATIONS = 10
    max_steps = 2000
    performanceChecks = 10
    vis_learning_freq = 200
    checks_per_policy = 1
    
    # Domain Arguments----------------------
    #MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    MAZE                = '/Domains/GridWorldMaps/4x5.txt'
    NOISE               = 0.3   # Noise parameters used for some of the domains such as the GridWorld
    
    # Representation ----------------------
    DISCRITIZATION              = 20    # Number of bins used to discretize each continuous dimension. Used for some representations 
    
    # Policy ----------------------
    EPSILON                 = .2 # EGreedy Often is .1 CHANGE ME <<<
    
    #Agent ----------------------
    alpha_decay_mode        = 'boyan' # Decay rate parameter; See Agent.py initialization for more information
    initial_alpha           = .1
    boyan_N0                = 100
    LAMBDA                  = 0
    
    # DOMAIN
    #=================
    domain          = GridWorld(RL_PYTHON_ROOT+'/'+MAZE, noise = NOISE, logger = logger)
    #domain          = Pendulum_InvertedBalance(episodeCap = 300, logger = logger);
    
    # REPRESENTATION
    #================
    representation  = Tabular(domain,logger,discretization = DISCRITIZATION) # Optional parameter discretization, for continuous domains
    
    # POLICY
    #================
    policy          = eGreedy(representation,logger, epsilon = EPSILON)
    
    # LEARNING AGENT
    #================
    agent           = Q_LEARNING(representation,policy,domain,logger,initial_alpha,LAMBDA, alpha_decay_mode, boyan_N0)
    
    experiment = Experiment(**locals())
    return experiment
    
if __name__ == '__main__':
    path = './Results/I-Should-Run'
    something = make_experiment(1, path=path) #use ID 1 by default
    if isinstance(something, Experiment):
        experiment = something
        #from Tools.run import run_profiled
        #run_profiled(make_experiment)
        experiment = make_experiment(1, path=path)
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
