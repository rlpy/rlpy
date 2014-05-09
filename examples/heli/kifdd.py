from rlpy.Domains import HelicopterHover
from rlpy.Agents import Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp
from rlpy.Representations import FastKiFDD

param_space = {
    'kernel_resolution':
    hp.loguniform("kernel_resolution", np.log(5), np.log(50)),
    'discover_threshold':
    hp.loguniform(
        "discover_threshold",
        np.log(1e4),
        np.log(1e8)),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=1e6,
        boyan_N0=5e5,
        lambda_=0.5,
        initial_alpha=0.9,
        kernel_resolution=10):
    max_steps = 10000
    num_policy_checks = 30
    checks_per_policy = 1
    active_threshold = 0.01
    max_base_feat_sim = 0.5
    sparsify = 1

    domain = HelicopterHover()
    # domain = FiniteCartPoleBalanceModern()
    kernel_width = (domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]) \
        / kernel_resolution
    representation = FastKiFDD(domain, sparsify=sparsify,
                               kernel=linf_triangle_kernel,
                               kernel_args=[kernel_width],
                               active_threshold=active_threshold,
                               discover_threshold=discover_threshold,
                               normalization=True,
                               max_active_base_feat=10,
                               max_base_feat_sim=max_base_feat_sim)
    policy = eGreedy(representation, epsilon=0.1)
    # lambda_=.0, alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    agent = Q_Learning(
        representation, policy, domain, lambda_=lambda_, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    run_profiled(make_experiment)
    #experiment = make_experiment(1)
    # experiment.run(visualize_learning=True)
    # experiment.plot()
    # experiment.save()
