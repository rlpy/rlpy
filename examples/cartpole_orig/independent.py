"""
Cart-pole balancing with independent discretization
"""
from rlpy.Domains.FiniteTrackCartPole import FiniteCartPoleBalanceOriginal, FiniteCartPoleBalanceModern
from rlpy.Agents import SARSA, Q_LEARNING
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 3, 5, 1),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        exp_id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=235,
        initial_learn_rate=.05,
        discretization=5):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 30000
    opt["num_policy_checks"] = 20
    opt["checks_per_policy"] = 1

    domain = FiniteCartPoleBalanceOriginal(good_reward=0.)
    opt["domain"] = domain
    representation = IndependentDiscretization(
        domain,
        discretization=discretization)
    policy = eGreedy(representation, epsilon=0.1)
    opt["agent"] = Q_LEARNING(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=0.9, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
    # experiment.save()
