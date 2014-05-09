"""
Cart-pole balancing with iFDD+
"""

import sys
import os
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
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=0.0148120884,
        lambda_=0.,
        boyan_N0=460.3858,
        initial_alpha=0.8014120,
        discretization=25.):
    max_steps = 50000
    num_policy_checks = 20
    checks_per_policy = 10
    sparsify = True
    kappa = 1e-7
    domain = InfCartPoleBalance()

    initial_rep = IndependentDiscretization(
        domain,
        discretization=discretization)
    representation = iFDD(domain, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=1 - kappa)
    policy = eGreedy(representation, epsilon=0.1)
    agent = Q_Learning(representation, policy, domain,
                       lambda_=lambda_,
                       initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
