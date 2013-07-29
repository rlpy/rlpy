"""
Cart-pole balancing with continuous / Kernelized iFDD
"""
from Tools import Logger
from Domains.HIVTreatment import HIVTreatment
from Agents import Q_Learning
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'kernel_resolution': hp.loguniform("kernel_resolution", np.log(5), np.log(50)),
               'discover_threshold': hp.loguniform("discover_threshold", np.log(1e-1), np.log(1e2)),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
                    discover_threshold = 1e5,
                    boyan_N0 = 1000,
                    initial_alpha = 1,
                    kernel_resolution=20.14):
    logger = Logger()
    max_steps = 150000
    num_policy_checks = 30
    checks_per_policy = 1
    active_threshold = 0.01
    max_base_feat_sim = 0.5
    sparsify = 3

    domain = HIVTreatment(logger=logger)
    # domain = CartPoleBalanceModern(logger=logger)
    kernel_width = (domain.statespace_limits[:,1] - domain.statespace_limits[:,0]) \
                   / kernel_resolution

    representation = KernelizediFDD(domain, sparsify=sparsify,
                                    kernel=linf_triangle_kernel,
                                    kernel_args=[kernel_width],
                                    active_threshold=active_threshold,
                                    logger=logger,
                                    discover_threshold=discover_threshold,
                                    normalization=True,
                                    max_active_base_feat=10,
                                    max_base_feat_sim=max_base_feat_sim)
    policy = eGreedy(representation, logger, epsilon=0.1)
    #agent           = SARSA(representation,policy,domain,logger,initial_alpha=initial_alpha,
    #                        lambda_=.0, alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    agent = Q_Learning(representation, policy, domain, logger
                       ,lambda_=0.5, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    #experiment.plot()
    #experiment.save()
