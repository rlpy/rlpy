"""
System Administrator with incremental tabular representation, Q-Learning
"""
from rlpy.Domains.SystemAdministrator import SystemAdministrator
from rlpy.Agents import Q_Learning
from rlpy.Representations import IncrementalTabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=120,
        initial_learn_rate=.06,
        discretization=50):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 1

    domain = SystemAdministrator()
    opt["domain"] = domain
    representation = IncrementalTabular(domain)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Q_Learning(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=0.9, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False, visualize_performance=10,
                   visualize_learning=True)
    experiment.plot()
    # experiment.save()
