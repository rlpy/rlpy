from Tools import Logger
from Domains import PuddleWorld
from Agents import Q_Learning
from Representations import *
from Experiments import Experiment
from hyperopt import hp
import numpy as np
from Domains import FiniteCartPoleSwingUpFriction
from Policies.FixedPolicy import GoodCartPoleSwingupPolicy
from ValueEstimators.Forest import Forest
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment

param_space = {"learn_rate_coef": hp.loguniform("learn_rate_coef", np.log(1e-2), np.log(1e0)),
               'm': hp.qloguniform("m", np.log(1e0), np.log(1e2), 1),
               'grow_coef': hp.loguniform("grow_coef", np.log(1e1), np.log(1e2)),
               'learn_rate_exp': hp.uniform("learn_rate_exp", -0.1, -.01),
               'lambda_': hp.uniform("lambda_", 0., 1.),
               'grow_exp': hp.loguniform("grow_exp", np.log(1.00000001),
                   np.log(1.4))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/forest1",
                    learn_rate_coef=0.03870882,
                    learn_rate_exp=-0.011265, #13444,  #-0.04,
                    grow_coef=10.1152766,
                    grow_exp=1.2217416, #0001,
                    p_structure=0.05,
                    m=5,
                    lambda_=0.994068):
    logger = Logger()
    max_steps = 500000

    domain = FiniteCartPoleSwingUpFriction(logger=logger)
    pol = GoodCartPoleSwingupPolicy(domain, logger)
    forest = Forest(domain, None, num_trees=10, m=m, p_structure=p_structure,
                    grow_coef=grow_coef, grow_exp=grow_exp, learn_rate_mode="exp",
                    learn_rate_coef=learn_rate_coef,
                    learn_rate_exp=learn_rate_exp, lambda_=lambda_)
    experiment = PolicyEvaluationExperiment(forest, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, id=id, log_interval=10)
    experiment.num_eval_points_per_dim=20
    experiment.num_traj_V = 100
    experiment.num_traj_stationary = 1000
    #num = domain.num_V_batches(pol, experiment.num_traj_V, experiment.num_eval_points_per_dim)
    #domain.precompute_V_batch_cache(experiment.num_eval_points_per_dim, pol, max_len_traj=200,
    #                                num_traj=experiment.num_traj_V, batch=id)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    #from Tools.run import run_profiled
    #run_profiled(make_experiment)
    experiment.run(visualize_learning=False)
    experiment.plot(y="rmse")

    #tree = experiment.estimator.trees[0]
    #import matplotlib.pyplot as plt
    #plt.figure("Cuts")
    #tree.plot_2d_cuts()
