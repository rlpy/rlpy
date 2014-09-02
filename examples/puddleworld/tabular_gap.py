"""
Cart-pole balancing with continuous / Kernelized iFDD
"""

from rlpy.Domains.PuddleWorld import PuddleGapWorld, PuddleWorld
from rlpy.Agents import SARSA, Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {
    # 'kernel_resolution': hp.loguniform("kernel_resolution", np.log(3), np.log(100)),
    'discretization': hp.uniform("discretization", 10, 50),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(1e-3), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=389.56,
        lambda_=0.52738,
        initial_learn_rate=.424409,
        discretization=30):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 400000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 100

    domain = PuddleGapWorld()
    opt["domain"] = domain
    representation = Tabular(domain, discretization=discretization)
    policy = eGreedy(representation, epsilon=0.1)
    # agent           = SARSA(representation,policy,domain,initial_learn_rate=1.,
    # lambda_=0., learn_rate_decay_mode="boyan", boyan_N0=100)
    opt["agent"] = Q_Learning(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.save()
