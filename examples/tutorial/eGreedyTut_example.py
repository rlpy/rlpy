#!/usr/bin/env python
"""
Policy Tutorial for RLPy
=================================

Assumes you have created the eGreedyTut.py agent according to the tutorial and
placed it in the current working directory.
Tests the policy on the GridWorld domain, with the policy and value function
visualized.
"""
__author__ = "Robert H. Klein"
from rlpy.Domains import GridWorld
from rlpy.Agents import SARSA
from rlpy.Representations import Tabular
from eGreedyTut import eGreedyTut
from rlpy.Experiments import Experiment
import os


def make_experiment(exp_id=1, path="./Results/Tutorial/gridworld-eGreedyTut"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path

    ## Domain:
    maze = '4x5.txt'
    domain = GridWorld(maze, noise=0.3)
    opt["domain"] = domain

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain, discretization=20)

    ## Policy
    policy = eGreedyTut(representation, epsilon=0.2)

    ## Agent
    opt["agent"] = SARSA(representation=representation, policy=policy,
                  discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1)
    opt["checks_per_policy"] = 100
    opt["max_steps"] = 2000
    opt["num_policy_checks"] = 10
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()
