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
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(1e-3), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=21.492871,
        initial_learn_rate=0.0385705,
        discretization=6.):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 50000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 10

    domain = InfCartPoleBalance()
    opt["domain"] = domain

    representation = IndependentDiscretization(
        domain,
        discretization=discretization)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Greedy_GQ(policy, representation,
                      discount_factor=domain.discount_factor,
                      lambda_=lambda_,
                      BetaCoef=1e-6,
                      initial_learn_rate=initial_learn_rate,
                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
