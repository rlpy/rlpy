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

param_space = {
    "num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
    'resolution': hp.quniform("resolution", 3, 30, 1),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'lambda_': hp.uniform("lambda_", 0., 1.),
    'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=2120,
        initial_alpha=.26,
        lambda_=0.9,
        resolution=8, num_rbfs=4958):
    logger = Logger()
    max_steps = 30000
    num_policy_checks = 20
    checks_per_policy = 10

    domain = FiniteCartPoleBalanceOriginal(logger=logger, good_reward=0.)
    representation = RBF(domain, num_rbfs=int(num_rbfs), logger=logger,
                         resolution_max=resolution, resolution_min=resolution,
                         const_feature=False, normalize=True, seed=id)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Q_LEARNING(
        representation, policy, domain, logger, lambda_=lambda_, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
    # experiment.save()
