"""
Cart-pole balancing with continuous / Kernelized iFDD
"""

from Tools import Logger
from Domains import Pendulum_InvertedBalance
from Agents import SARSA, Q_LEARNING
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'kernel_resolution': hp.loguniform("kernel_resolution", np.log(3), np.log(100)),
               'discover_threshold': hp.loguniform("discover_threshold", np.log(1e-2), np.log(1e1)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-3), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold = .01356,
                    boyan_N0 = 235.,
                    lambda_ = 0.6596,
                    initial_alpha = .993,
                    kernel_resolution=45.016):
    logger = Logger()
    max_steps = 10000
    num_policy_checks = 20
    checks_per_policy = 10
    active_threshold = 0.01
    max_base_feat_sim = 0.5
    sparsify = 1

    domain = Pendulum_InvertedBalance(logger=logger)
    kernel_width = (domain.statespace_limits[:,1] - domain.statespace_limits[:,0]) / kernel_resolution
    representation = FastKiFDD(domain, sparsify=sparsify,
                                    kernel=gaussian_kernel,
                                    kernel_args=[kernel_width],
                                    active_threshold=active_threshold,
                                    logger=logger, discover_threshold=discover_threshold,
                                    normalization=True,
                                    max_active_base_feat=10, max_base_feat_sim=max_base_feat_sim)
    policy = eGreedy(representation, logger, epsilon=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha=1.,
    #                        lambda_=0., alpha_decay_mode="boyan", boyan_N0=100)
    agent = Q_LEARNING(representation, policy, domain, logger
                       ,lambda_=lambda_, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run()
    experiment.save()
