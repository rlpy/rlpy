"""
Fifty-State Chain with tabular representation, Q-Learning
"""
from rlpy.Domains.FiftyChain import FiftyChain
from rlpy.Agents import Q_Learning
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=120,
        initial_learn_rate=.06,
        discretization=50):
    max_steps = 40000
    num_policy_checks = 10
    checks_per_policy = 1

    domain = FiftyChain()
    representation = Tabular(
        domain)
    policy = eGreedy(representation, epsilon=0.1)
    agent = Q_Learning(
        domain, policy, representation, lambda_=0.9, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False, visualize_performance=3,
                   visualize_learning=True)
    experiment.plot()
    # experiment.save()
