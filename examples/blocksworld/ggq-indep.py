from Tools import Logger
from Domains import BlocksWorld
from Agents import Greedy_GQ
from Representations import *
from Policies import eGreedy
from Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=10.09,
        initial_alpha=.47):
    logger = Logger()
    max_steps = 100000
    num_policy_checks = 20
    checks_per_policy = 5
    sparsify = 1
    ifddeps = 1e-7
    domain = BlocksWorld(blocks=6, noise=0.3, logger=logger)
    representation = IndependentDiscretization(domain, logger)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Greedy_GQ(
        representation, policy, domain, logger, lambda_=lambda_, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    # experiment.plot()
    # experiment.save()
