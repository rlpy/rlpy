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
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=1240.89223,
        initial_alpha=0.0063744503,
        discretization=8.):
    max_steps = 50000
    num_policy_checks = 20
    checks_per_policy = 10
    sparsify = True

    domain = InfCartPoleBalance()

    representation = IndependentDiscretization(
        domain,
        discretization=discretization)
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
