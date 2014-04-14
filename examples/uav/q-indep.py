from Tools import Logger
from Domains import PST
from Agents import Q_Learning
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {  # 'discretization': hp.quniform("discretization", 5, 50, 1),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=3571.6541,
        initial_alpha=0.62267772):
    logger = Logger()
    max_steps = 500000
    num_policy_checks = 30
    checks_per_policy = 10
    beta_coef = 1e-6
    domain = PST(NUM_UAV=4, motionNoise=0, logger=logger)
    representation = IndependentDiscretization(domain, logger)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Q_Learning(representation, policy, domain, logger,
                       lambda_=lambda_, initial_alpha=initial_alpha,
                       alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    run_profiled(make_experiment)
    #experiment = make_experiment(1)
    # experiment.run()
    # experiment.plot()
    # experiment.save()
