"""
Cart-pole balancing with continuous / Kernelized iFDD
"""
from Tools import Logger
from Domains import HIVTreatment
from Agents import SARSA, Q_LEARNING
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 50, 1),
               'discover_threshold': hp.loguniform("discover_threshold",
                   np.log(1e0), np.log(1e8)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold = 107091,
                    lambda_=0.245,
                    boyan_N0 = 514,
                    initial_alpha = .327,
                    discretization=18):
    logger = Logger()
    max_steps = 150000
    num_policy_checks = 30
    checks_per_policy = 1
    sparsify = 1
    domain = HIVTreatment(logger=logger)
    initial_rep = QIndependentDiscretization(domain, logger, discretization=discretization)
    representation = QiFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=True)
    #representation.PRINT_MAX_RELEVANCE = True
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
    experiment.run()
    #experiment.plot()
    #experiment.save()
