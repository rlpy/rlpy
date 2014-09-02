"""
Cart-pole balancing with continuous / Kernelized iFDD
"""

from rlpy.Domains.PuddleWorld import PuddleGapWorld
from rlpy.Agents import SARSA, Q_LEARNING
from rlpy.Representations import *
from rlpy.Policies import eGreedy, UniformRandom
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp
from rlpy.Representations import KernelizediFDD

param_space = {
    'kernel_resolution':
    hp.loguniform("kernel_resolution", np.log(3), np.log(100)),
    'discover_threshold':
    hp.loguniform(
        "discover_threshold",
        np.log(1e-2),
        np.log(1e1)),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(1e-3), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=1e-20,
        boyan_N0=1589.56,
        lambda_=0.52738,
        initial_learn_rate=.0124409,
        kernel_resolution=8.5):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 100
    active_threshold = 0.01
    max_base_feat_sim = 0.7
    sparsify = 10

    domain = PuddleGapWorld()
    opt["domain"] = domain
    kernel_width = (
        domain.statespace_limits[:,
                                 1] - domain.statespace_limits[:,
                                                               0]) / kernel_resolution
    representation = KernelizediFDD(domain, sparsify=sparsify,
                               kernel=gaussian_kernel,
                               kernel_args=[kernel_width],
                               active_threshold=active_threshold,
                               discover_threshold=discover_threshold,
                               normalization=False,
                               max_active_base_feat=100, max_base_feat_sim=max_base_feat_sim)
    #  policy = UniformRandom(representation, )
    policy = eGreedy(representation, epsilon=0.1)
    # agent           = SARSA(representation,policy,domain,initial_learn_rate=1.,
    # lambda_=0., learn_rate_decay_mode="boyan", boyan_N0=100)
    opt["agent"] = Q_LEARNING(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan_const", boyan_N0=boyan_N0)
    stat_bins_per_state_dim = 22
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_performance=-1, visualize_learning=True)
    experiment.save()
