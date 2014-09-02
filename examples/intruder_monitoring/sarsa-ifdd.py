"""
Intruder Monitoring with iFDD representation, SARSA
"""
from rlpy.Domains.IntruderMonitoring import IntruderMonitoring
from rlpy.Agents import SARSA
from rlpy.Representations import iFDD, IndependentDiscretization
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discover_threshold': hp.loguniform("discover_threshold",
                                                   np.log(1e-3), np.log(1e2)),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=1.0,
        lambda_=0.,
        boyan_N0=20.1,
        initial_learn_rate=0.330):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 1
    sparsify = 1
    ifddeps = 1e-7
    domain = IntruderMonitoring()
    opt["domain"] = domain
    initial_rep = IndependentDiscretization(domain)
    representation = iFDD(domain, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          useCache=True,
                          iFDDPlus=1 - ifddeps)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = SARSA(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False, visualize_performance=10,
                   visualize_learning=True)
    # experiment.plot()
    # experiment.save()
