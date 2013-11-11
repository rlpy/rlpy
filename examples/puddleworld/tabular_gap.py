"""
Cart-pole balancing with continuous / Kernelized iFDD
"""

from Tools import Logger
from Domains.PuddleWorld import PuddleGapWorld, PuddleWorld
from Agents import SARSA, Q_Learning
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp
from Representations import FastKiFDD

param_space = {#'kernel_resolution': hp.loguniform("kernel_resolution", np.log(3), np.log(100)),
               'discretization': hp.uniform("discretization", 10, 50),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    boyan_N0 = 389.56,
                    lambda_ = 0.52738,
                    initial_alpha = .424409,
                    discretization=30):
    logger = Logger()
    max_steps = 400000
    num_policy_checks = 10
    checks_per_policy = 100

    domain = PuddleGapWorld(logger=logger)
    representation = Tabular(domain, logger, discretization=discretization)
    policy = eGreedy(representation, logger, epsilon=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha=1.,
    #                        lambda_=0., alpha_decay_mode="boyan", boyan_N0=100)
    agent = Q_Learning(representation, policy, domain, logger
                       ,lambda_=lambda_, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.save()
