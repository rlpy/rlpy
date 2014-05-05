"""
Cart-pole balancing with iFDD+
"""

import sys
import os
from rlpy.Tools import Logger
from rlpy.Domains import InfCartPoleBalance
from rlpy.Agents import Greedy_GQ, SARSA, Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 40, 1),
               'discover_threshold':
               hp.loguniform(
                   "discover_threshold",
                   np.log(1e-2),
                   np.log(1e1)),
               #'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=0.013461679,
        lambda_=0.,
        boyan_N0=484.78006,
        initial_alpha=0.5651405,
        discretization=23.):
    logger = Logger()
    max_steps = 50000
    num_policy_checks = 20
    checks_per_policy = 10
    sparsify = True
    kappa = 1e-7
    domain = InfCartPoleBalance(logger=logger)

    initial_rep = IndependentDiscretization(
        domain,
        logger,
        discretization=discretization)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=1. - kappa)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Greedy_GQ(representation, policy, domain, logger,
                      lambda_=lambda_,
                      BetaCoef=1e-6,
                      initial_alpha=initial_alpha,
                      alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
