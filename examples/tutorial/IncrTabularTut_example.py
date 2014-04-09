#!/usr/bin/env python
"""
Representation Tutorial for RLPy
================================

Assumes you have created the IncrTabularTut.py agent according to the tutorial and
placed it in the Representations/ directory.
Tests the Representation on the GridWorld domain usin SARSA
"""
__author__ = "Robert H. Klein"
from Domains import GridWorld
from Tools import Logger
from Agents import SARSA
from Representations import IncrTabularTut
from Policies import eGreedy
from Experiments import Experiment
import os


def make_experiment(id=1, path="./Results/Tutorial/gridworld-IncrTabularTut"):
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
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = IncrTabularTut(domain, logger)

    ## Policy
    policy = eGreedy(representation, logger, epsilon=0.2)

    ## Agent
    agent = SARSA(representation=representation, policy=policy,
                       domain=domain, logger=logger,
                       learning_rate=0.1)
    checks_per_policy = 100
    max_steps = 2000
    num_policy_checks = 10
    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()
