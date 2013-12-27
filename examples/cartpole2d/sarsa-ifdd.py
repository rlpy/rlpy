"""
Cart-pole balancing with iFDD+
"""

import sys, os
from Tools import Logger
from Domains import InfCartPoleBalance
from Agents import Greedy_GQ, SARSA, Q_Learning
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 40, 1),
               'discover_threshold': hp.loguniform("discover_threshold", np.log(1e-2), np.log(1e1)),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold=0.03613232738,
                    lambda_=0.,
                    boyan_N0=12335.665,
                    initial_alpha=0.037282,
                    discretization=6.):
    logger = Logger()
    max_steps = 50000
    num_policy_checks = 20
    checks_per_policy = 10
    sparsify = True
    kappa = 1e-7
    domain = InfCartPoleBalance(logger=logger)

    initial_rep = QIndependentDiscretization(domain, logger, discretization=discretization)
    representation = QiFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=1-kappa)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = SARSA(representation, policy, domain, logger,
                       lambda_=lambda_,
                       initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
