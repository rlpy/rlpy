from rlpy.Tools import Logger
from rlpy.Domains import PST
from rlpy.Agents import Greedy_GQ
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {  # 'discretization': hp.quniform("discretization", 5, 50, 1),
    'discover_threshold': hp.loguniform("discover_threshold",
                                        np.log(5e1), np.log(1e4)),
    #'lambda_': hp.uniform("lambda_", 0., 1.),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=738.59,
        lambda_=.8927,
        boyan_N0=2804.,
        initial_alpha=.2706):
    logger = Logger()
    max_steps = 1000000
    num_policy_checks = 30
    checks_per_policy = 10
    sparsify = 1
    ifddeps = 1e-7
    beta_coef = 1e-6
    domain = PST(NUM_UAV=4, motionNoise=0, logger=logger)
    initial_rep = IndependentDiscretization(domain, logger)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          # discretization=discretization,
                          useCache=True,
                          iFDDPlus=1 - ifddeps)
    policy = eGreedy(representation, logger, epsilon=0.1)
    agent = Greedy_GQ(representation, policy, domain, logger,
                      BetaCoef=beta_coef,
                      lambda_=lambda_, initial_alpha=initial_alpha,
                      alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    run_profiled(make_experiment)
    #experiment = make_experiment(1)
    # experiment.run()
    # experiment.plot()
    # experiment.save()
