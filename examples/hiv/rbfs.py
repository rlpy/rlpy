"""
Cart-pole balancing with independent discretization
"""
from Tools import Logger
from Domains import HIVTreatment
from Agents import SARSA, Q_LEARNING
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
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
        boyan_N0=136,
        lambda_=0.0985,
        initial_alpha=0.090564,
        resolution=13., num_rbfs=9019):
    logger = Logger()
    max_steps = 150000
    num_policy_checks = 30
    checks_per_policy = 1

    domain = HIVTreatment(logger=logger)
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
