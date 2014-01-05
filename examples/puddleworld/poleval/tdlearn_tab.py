"""
Cart-pole balancing with independent discretization
"""
from Tools import Logger
from Domains import PuddleWorld
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Experiments import Experiment
from hyperopt import hp
import numpy as np
from Domains import PuddleWorld
from Policies import BasicPuddlePolicy
from ValueEstimators.Forest import Forest
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment

param_space = {'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/tab",
                    boyan_N0=3243.266410,
                    initial_alpha=0.191, #0.6633,
                    discretization=16.,
                    lambda_=0.81140): #1953):
    logger = Logger()
    max_steps = 1000000
    num_policy_checks = 20

    domain = PuddleWorld(logger=logger)
    representation = Tabular(domain, logger=logger, discretization=discretization)
    pol = BasicPuddlePolicy(domain, None)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)

    #policy = eGreedy(representation, logger, epsilon=0.1)
    #agent = Q_Learning(representation, policy, domain, logger
    #                   ,lambda_=lambda_, initial_alpha=initial_alpha,
    #                   alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    #experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_learning=False)
    experiment.plot(y="rmse")
