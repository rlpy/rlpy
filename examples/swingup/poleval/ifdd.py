from Tools import Logger
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Domains import FiniteCartPoleSwingUpFriction
from Policies.FixedPolicy import GoodCartPoleSwingupPolicy
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment
from Policies import OptimalBlocksWorldPolicy
import numpy as np
from hyperopt import hp

param_space = {'discretization': hp.quniform("discretization", 5, 40, 1),
               'discover_threshold': hp.loguniform("discover_threshold",
                   np.log(1e-3), np.log(1e2)),
               #'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/ifdd/",
                    discretization=21,
                    discover_threshold=1.3763480,
                    lambda_=0.0,
                    boyan_N0=33697.309,
                    initial_alpha=0.41957):
    logger = Logger()
    max_steps = 100000
    sparsify = 1
    domain = FiniteCartPoleSwingUpFriction(logger=logger)
    pol = GoodCartPoleSwingupPolicy(domain, logger)

    initial_rep = IndependentDiscretization(domain, logger)
    representation = iFDD(domain, logger, discover_threshold, initial_rep,
                          sparsify=sparsify,
                          useCache=True,
                          iFDDPlus=1)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)
    experiment.num_eval_points_per_dim=20
    experiment.num_traj_V = 100
    experiment.num_traj_stationary = 1000
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    #experiment.plot()
    #experiment.save()
