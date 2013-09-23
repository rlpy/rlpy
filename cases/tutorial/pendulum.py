#!/usr/bin/env python
__author__ = "Robert H. Klein"
from Domains import Pendulum_InvertedBalance
from Tools import Logger
from Agents import Q_Learning
from Representations import Tabular
from Policies import eGreedy
from Experiments import Experiment


def make_experiment(id=1, path="./Results/Temp"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    logger = Logger()

    domain = Pendulum_InvertedBalance(episodeCap=300, logger=logger)

    representation  = Tabular(domain, logger, discretization=20)
    policy = eGreedy(representation, logger, epsilon=0.2)
    agent = Q_Learning(representation=representation, policy=policy,
                       domain=domain, logger=logger,
                       initial_alpha=0.1,
                       alpha_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    max_steps = 8000
    num_policy_checks = 5
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    path = './Results/tutorial_pendulum'
    experiment = make_experiment(1, path=path)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show performance runs?
                   visualize_performance=True)  # show value function?
    experiment.plot()
    experiment.save()
