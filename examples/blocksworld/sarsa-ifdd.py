from rlpy.Tools import Logger
from rlpy.Domains import BlocksWorld
from rlpy.Agents import SARSA
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discover_threshold': hp.loguniform("discover_threshold",
                                                   np.log(1e-3), np.log(1e2)),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=0.023476,
        lambda_=0.,
        boyan_N0=20.84362,
        initial_alpha=0.3356222674):
    logger = Logger()
    max_steps = 100000
    num_policy_checks = 20
    checks_per_policy = 1
    sparsify = 1
    ifddeps = 1e-7
    domain = BlocksWorld(blocks=6, noise=0.3, logger=logger)
    initial_rep = IndependentDiscretization(domain, logger)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          useCache=True,
                          iFDDPlus=1 - ifddeps)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = SARSA(
        representation, policy, domain, logger, lambda_=lambda_, initial_alpha=initial_alpha,
        alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    # experiment.plot()
    # experiment.save()
