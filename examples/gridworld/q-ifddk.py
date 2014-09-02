#!/usr/bin/env python

__author__ = "William Dabney"

from rlpy.Domains import GridWorld
from rlpy.Agents import Q_Learning
from rlpy.Representations import iFDDK, IndependentDiscretization
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os


def make_experiment(exp_id=1, path="./Results/Temp"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    # Experiment variables
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 100000
    opt["num_policy_checks"] = 10

    # Logging

    # Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3)
    opt["domain"] = domain

    # Representation
    discover_threshold = 1.
    lambda_ = 0.3
    initial_learn_rate = 0.11
    boyan_N0 = 100

    initial_rep = IndependentDiscretization(domain)
    representation = iFDDK(domain, discover_threshold, initial_rep,
                          sparsify=True,
                          useCache=True, lazy=True,
                          lambda_=lambda_)

    # Policy
    policy = eGreedy(representation, epsilon=0.1)

    # Agent
    opt["agent"] = Q_Learning(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)

    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show performance runs?
                   visualize_performance=True)  # show value function?
    experiment.plot()
    experiment.save()
