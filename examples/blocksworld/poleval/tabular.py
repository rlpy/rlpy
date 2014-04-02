"""
Cart-pole balancing with independent discretization
"""
from Tools import Logger
from Domains import BlocksWorld
from ValueEstimators.TDLearning import TDLearning
from Representations import *
from Experiments import Experiment
from hyperopt import hp
import numpy as np
from Domains import PuddleWorld
from Policies import OptimalBlocksWorldPolicy
from ValueEstimators.Forest import Forest
from Experiments.PolicyEvaluationExperiment import PolicyEvaluationExperiment

param_space = {'lambda_': hp.uniform("lambda_", 0., 1.),
               'boyan_N0': hp.loguniform("boyan_N0", np.log(1e1), np.log(1e5)),
               'initial_alpha': hp.loguniform("initial_alpha", np.log(5e-2), np.log(1))}


def make_experiment(id=1, path="./Results/Temp/{domain}/poleval/tab",
                    boyan_N0=60.477342,
                    initial_alpha=0.133447,
                    lambda_=0.6613):
    logger = Logger()
    max_steps = 1000000

    domain = BlocksWorld(blocks=6, gamma=0.95, logger=logger, noise=0.1)
    representation = Tabular(domain, logger=logger)
    pol = OptimalBlocksWorldPolicy(domain, random_action_prob=0.3)
    estimator = TDLearning(representation=representation, lambda_=lambda_,
                           boyan_N0=boyan_N0, initial_alpha=initial_alpha, alpha_decay_mode="boyan")
    experiment = PolicyEvaluationExperiment(estimator, domain, pol, max_steps=max_steps, num_checks=20,
                                            path=path, log_interval=10, id=id)

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
    """
    import matplotlib.pyplot as plt
    plt.figure()
    V_true, s_test, s_term, s_distr = experiment.domain.V(experiment.target_policy)
    V_pred = np.zeros_like(V_true)
    for i in xrange(s_test.shape[1]):
        V_pred[i] = experiment.estimator.predict(s_test[:,i], s_term[i])
    plt.plot(V_true, label="true")
    plt.plot(V_pred, label="prediction")
    plt.plot(s_distr / s_distr.sum(), label="distr")
    plt.legend()
    """
