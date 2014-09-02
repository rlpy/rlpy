from rlpy.Domains import BlocksWorld
from rlpy.Agents import Greedy_GQ
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        lambda_=0.,
        boyan_N0=10.09,
        initial_learn_rate=.47):
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 5
    sparsify = 1
    ifddeps = 1e-7
    domain = BlocksWorld(blocks=6, noise=0.3)
    opt["domain"] = domain
    representation = IndependentDiscretization(domain)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Greedy_GQ(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    # experiment.plot()
    # experiment.save()
