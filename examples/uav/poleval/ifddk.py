from Tools import Logger
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Domains import PST
from Tools import __rlpy_location__
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment
from Policies import StoredPolicy
import numpy as np
from hyperopt import hp

param_space = {
               'discover_threshold': hp.loguniform("discover_threshold",
                   np.log(1e1), np.log(1e4)),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'kappa': hp.loguniform("kappa", np.log(1e-10), np.log(1e-3)),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(1e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/ifdd/",
                    discover_threshold=17.9708712, #0.42878655,
                    lambda_=0.57053162,
                    kappa=7.6e-9,
                    boyan_N0=78454.41838,
                    initial_alpha=0.16240166):
    logger = Logger()
    max_steps = 500000
    sparsify = 1
    domain = PST(NUM_UAV=4, motionNoise=0, logger=logger)
    pol = StoredPolicy(filename="__rlpy_location__/Policies/PST_4UAV_mediocre_policy_nocache.pck")
    initial_rep = IndependentDiscretization(domain, logger)
    representation = iFDDK(domain, logger, discover_threshold, initial_rep,
                           sparsify=sparsify, lambda_=lambda_,
                           useCache=True, lazy=True, kappa=kappa)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)
    experiment.num_eval_points_per_dim=20
    experiment.num_traj_V = 300
    experiment.num_traj_stationary = 300
    return experiment

if __name__ == '__main__':
    from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment = make_experiment(1)
    experiment.run()
    #experiment.plot()
    #experiment.save()
