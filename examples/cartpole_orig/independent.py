"""
Cart-pole balancing with independent discretization
"""
from Tools import Logger
from Domains.FiniteTrackCartPole import FiniteCartPoleBalanceOriginal, FiniteCartPoleBalanceModern
from Agents import SARSA, Q_LEARNING
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 3, 5, 1),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    boyan_N0=235,
                    initial_alpha=.05,
                    discretization=5):
    logger = Logger()
    max_steps = 30000
    num_policy_checks = 20
    checks_per_policy = 1

    domain = FiniteCartPoleBalanceOriginal(logger=logger, good_reward=0.)
    representation = IndependentDiscretization(domain, discretization=discretization, logger=logger)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Q_LEARNING(representation, policy, domain, logger
                       ,lambda_=0.9, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
    #experiment.save()
