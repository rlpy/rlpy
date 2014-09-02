from rlpy.Domains import BlocksWorld
from rlpy.Agents import Greedy_GQ
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discover_threshold': hp.loguniform("discover_threshold",
                                                   np.log(1e-3), np.log(1e2)),
               #'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=0.012695,
        lambda_=0.2,
        boyan_N0=80.798,
        initial_learn_rate=0.402807):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 1
    sparsify = 1
    domain = BlocksWorld(blocks=6, noise=0.3)
    opt["domain"] = domain
    initial_rep = IndependentDiscretization(domain)
    representation = iFDDK(domain, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          useCache=True, lazy=True,
                          lambda_=lambda_)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Greedy_GQ(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    # experiment.plot()
    # experiment.save()
