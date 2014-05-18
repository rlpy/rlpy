from rlpy.Domains import HIVTreatment
from rlpy.Agents import SARSA, Q_LEARNING
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import numpy as np
from hyperopt import hp

param_space = {'resolution': hp.quniform("resolution", 3, 30, 1),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_learn_rate': hp.loguniform("initial_learn_rate", np.log(5e-2), np.log(1))}


def make_experiment(
        id=1, path="./Results/Temp/{domain}/{agent}/{representation}/",
        boyan_N0=136,
        lambda_=0.0985,
        initial_learn_rate=0.090564,
        resolution=13., num_rbfs=9019):
    max_steps = 150000
    num_policy_checks = 30
    checks_per_policy = 1

    domain = HIVTreatment()
    representation = NonparametricLocalBases(domain,
                                             kernel=linf_triangle_kernel,
                                             resolution=resolution,
                                             normalization=True)
    policy = eGreedy(representation, epsilon=0.1)
    agent = Q_LEARNING(
        domain, policy, representation, lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    from rlpy.Tools.run import run_profiled
    # run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run(visualize_learning=True)
    experiment.plot()
    # experiment.save()
