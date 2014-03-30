"""
Cart-pole balancing with independent discretization
"""
from Tools import Logger
from Domains import PuddleWorld
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Experiments import Experiment
from hyperopt import hp
import numpy as np
from Domains import FiniteCartPoleSwingUpFriction
from Policies.FixedPolicy import GoodCartPoleSwingupPolicy
from ValueEstimators.Forest import Forest
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment

param_space = {'lambda_': hp.uniform("lambda_", 0., 1.),
               'discretization': hp.quniform("discretization", 5, 40, 1),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/tab",
                    boyan_N0=1378.86965,
                    initial_alpha=0.0932996, #0.6633,
                    discretization=5,
                    lambda_=0.6433850): #1953):
    logger = Logger()
    max_steps = 100000
    domain = FiniteCartPoleSwingUpFriction(logger=logger)
    pol = GoodCartPoleSwingupPolicy(domain, logger)
    representation = IncrementalTabular(domain, logger=logger, discretization=discretization)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)

    experiment.num_eval_points_per_dim=20
    experiment.num_traj_V = 100
    experiment.num_traj_stationary = 1000
    #policy = eGreedy(representation, logger, epsilon=0.1)
    #agent = Q_Learning(representation, policy, domain, logger
    #                   ,lambda_=lambda_, initial_alpha=initial_alpha,
    #                   alpha_decay_mode="boyan", boyan_N0=boyan_N0)
    #experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_learning=False)
    experiment.plot(y="rmse")
