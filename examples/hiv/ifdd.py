"""
Cart-pole balancing with continuous / Kernelized iFDD
"""
from rlpy.Domains import HIVTreatment
from rlpy.Agents import SARSA, Q_LEARNING
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 50, 1),
               'discover_threshold': hp.loguniform("discover_threshold",
                                                   np.log(1e0), np.log(1e8)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        discover_threshold=107091,
        lambda_=0.245,
        boyan_N0=514,
        initial_learn_rate=.327,
        discretization=18):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 150000
    opt["num_policy_checks"] = 30
    opt["checks_per_policy"] = 1
    sparsify = 1
    domain = HIVTreatment()
    opt["domain"] = domain
    initial_rep = IndependentDiscretization(
        domain,
        discretization=discretization)
    representation = iFDD(domain, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          discretization=discretization,
                          useCache=True,
                          iFDDPlus=True)
    #representation.PRINT_MAX_RELEVANCE = True
    policy = eGreedy(representation, epsilon=0.1)
    # agent           = SARSA(representation,policy,domain,initial_learn_rate=initial_learn_rate,
    # lambda_=.0, learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    opt["agent"] = Q_LEARNING(
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
