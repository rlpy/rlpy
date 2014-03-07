"""
Cart-pole balancing with iFDD+
"""

import sys, os
from Tools import Logger
from Domains import PuddleWorld
from Agents import Q_Learning
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 40, 1),
               'discover_threshold': hp.loguniform("discover_threshold", np.log(1e-2), np.log(1e1)),
               'kappa': hp.loguniform("kappa", np.log(1e-10), np.log(1e-3)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold = 8.63917,
                    kappa=1e-5,
                    lambda_=0.42,
                    boyan_N0 = 202.,
                    initial_alpha = .7442,
                    discretization=18.):
    logger = Logger()
    max_steps = 30000
    num_policy_checks = 15
    checks_per_policy = 100
    sparsify = True

    domain = PuddleWorld(logger=logger)

    initial_rep = QIndependentDiscretization(domain, logger, discretization=discretization)
    representation = QiFDDK(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True, lazy=False,
                          kappa=kappa, lambda_=lambda_)
    policy = eGreedy(representation, logger, epsilon=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha=1.,
    #                        lambda_=0., alpha_decay_mode="boyan", boyan_N0=100)
    agent = Q_Learning(representation, policy, domain, logger,
                       lambda_=lambda_, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
