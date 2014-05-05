"""
Cart-pole balancing with independent discretization
"""
from rlpy.Tools import Logger
from rlpy.Domains import Pacman
from rlpy.Agents import Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 3, 50, 1),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.9,
        boyan_N0=22.36,
        initial_alpha=.068,
        discretization=9):
    logger = Logger()
    max_steps = 150000
    num_policy_checks = 30
    checks_per_policy = 1

    domain = Pacman(logger=logger)
    representation = IncrementalTabular(
        domain,
        discretization=discretization,
        logger=logger)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Q_Learning(
        representation, policy, domain, logger, lambda_=0.9, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    #from Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_steps=True)
    experiment.plot()
    # experiment.save()
