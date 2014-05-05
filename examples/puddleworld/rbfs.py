"""
Cart-pole balancing with independent discretization
"""
from rlpy.Tools import Logger
from rlpy.Domains import PuddleWorld
from rlpy.Agents import Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=13444.,
        initial_alpha=0.6633,
        resolution=21.,
        num_rbfs=96.,
        lambda_=.1953):
    logger = Logger()
    max_steps = 40000
    num_policy_checks = 20
    checks_per_policy = 100

    domain = PuddleWorld(logger=logger)
    representation = RBF(domain, num_rbfs=int(num_rbfs), logger=logger,
                         resolution_max=resolution, resolution_min=resolution,
                         const_feature=False, normalize=True, seed=id)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Q_Learning(
        representation, policy, domain, logger, lambda_=lambda_, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
