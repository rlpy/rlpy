#!/usr/bin/env python

__author__ = "William Dabney"

from rlpy.Domains import GridWorld
from rlpy.Agents import LSPI
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os


def make_experiment(id=1, path="./Results/Temp"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    # Experiment variables
    max_steps = 10000
    num_policy_checks = 10

    # Logging

    # Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3)

    # Representation
    representation = Tabular(domain, discretization=20)

    # Policy
    policy = eGreedy(representation, epsilon=0.1)

    # Agent
    agent = LSPI(representation, policy, domain,
                 max_steps, max_steps / num_policy_checks)

    experiment = Experiment(**locals())
    return experiment

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show performance runs?
                   visualize_performance=True)  # show value function?
    experiment.plot()
    experiment.save()
