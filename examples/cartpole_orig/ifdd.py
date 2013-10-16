"""
Cart-pole balancing with continuous / Kernelized iFDD
"""
from Tools import Logger
from Domains.CartPole import FiniteCartPoleBalanceOriginal, FiniteCartPoleBalanceModern
from Agents import SARSA, Q_LEARNING
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 50, 1),
               'discover_threshold': hp.loguniform("discover_threshold", np.log(1e-1), np.log(1e3)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold = 77.,
                    boyan_N0 = 11,
                    lambda_ = 0.9,
                    initial_alpha = .05,
                    discretization=47):
    logger = Logger()
    max_steps = 30000
    num_policy_checks = 20
    checks_per_policy = 10
    sparsify = 1

    domain = FiniteCartPoleBalanceOriginal(logger=logger, good_reward = 0.)
    initial_rep = IndependentDiscretization(domain, logger, discretization=discretization)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=True)
    policy = eGreedy(representation, logger, epsilon=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha=initial_alpha,
    #                        lambda_=.0, alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    agent = Q_LEARNING(representation, policy, domain, logger
                       ,lambda_=lambda_, initial_alpha=initial_alpha,
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
