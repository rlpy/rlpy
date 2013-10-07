#!/usr/bin/env python
"""
Getting Started Tutorial for RLPy
=================================

This file contains a very basic example of a RL experiment:
A simple Grid-World.
"""
__author__ = "Robert H. Klein"
from Domains import GridWorld
from Tools import Logger
from Agents import Q_Learning
from Representations import Tabular
from Policies import eGreedy
from Experiments import Experiment
import os


def make_experiment(id=1, path="./Results/Temp/Tutorial1/"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    logger = Logger()

    ## Domain:
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3, logger=logger)

    ## Representation
    representation  = Tabular(domain, logger, discretization=20)

    ## Policy
    policy = eGreedy(representation, logger, epsilon=0.2)

    ## Agent
    agent = Q_Learning(representation=representation, policy=policy,
                       domain=domain, logger=logger,
                       initial_alpha=0.1,
                       alpha_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)

    max_steps = 2000
    num_policy_checks = 10
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show performance runs?
                   visualize_performance=True)  # show value function?
    experiment.plot()
    experiment.save()
