from rlpy.Domains import PST
from rlpy.Agents import Greedy_GQ
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {  # 'discretization': hp.quniform("discretization", 5, 50, 1),
    'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
    'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=3019.313,
        initial_learn_rate=0.965830):
    max_steps = 500000
    num_policy_checks = 30
    checks_per_policy = 10
    beta_coef = 1e-6
    domain = PST(NUM_UAV=4, motionNoise=0)
    representation = IndependentDiscretization(domain)
    policy = eGreedy(representation, epsilon=0.1)
    agent = Greedy_GQ(domain, policy, representation,
                      BetaCoef=beta_coef,
                      lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    run_profiled(make_experiment)
    #experiment = make_experiment(1)
    # experiment.run()
    # experiment.plot()
    # experiment.save()
