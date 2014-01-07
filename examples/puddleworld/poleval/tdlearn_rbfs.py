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
from Domains import PuddleWorld
from Policies import BasicPuddlePolicy
from ValueEstimators.Forest import Forest
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment

param_space = {"num_rbfs": hp.qloguniform("num_rbfs", np.log(1e1), np.log(1e4), 1),
               'resolution': hp.quniform("resolution", 3, 30, 1),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/rbfs",
                    boyan_N0=5781.602,
                    initial_alpha=0.3414, #0.6633,
                    resolution=21.,
                    num_rbfs=6988.,
                    lambda_=0.388028): #1953):
    logger = Logger()
    max_steps = 1000000
    num_policy_checks = 20
    checks_per_policy = 100

    domain = PuddleWorld(logger=logger)
    representation = RBF(domain, num_rbfs=int(num_rbfs), logger=logger,
                         resolution_max=resolution, resolution_min=resolution,
                         const_feature=False, normalize=True, seed=id)
    pol = BasicPuddlePolicy(domain, None)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha)
    experiment = PolicyEvaluationExperiment(estimator, domain, pol,
                                            max_steps=max_steps, num_checks=20, 
                                            log_interval=10, path=path, id=id)

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
