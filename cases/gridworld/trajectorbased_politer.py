#!/usr/bin/env python

__author__ = "William Dabney"

from Domains import GridWorld
from Tools import Logger
from MDPSolvers import TrajectoryBasedPolicyIteration
from Representations import Tabular
from Policies import GibbsPolicy
from Experiments import Experiment


def make_experiment(id=1, path="./Results/Temp"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    ## Logging
    logger = Logger()

    ## Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    maze = './Domains/GridWorldMaps/4x5.txt'
    domain = GridWorld(maze, noise=0.3, logger=logger)

    ## Representation
    representation  = Tabular(domain, logger, discretization=20)

    ## Agent
    agent = TrajectoryBasedPolicyIteration(id, representation, domain, logger, project_path=path, show= True)

    return agent

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.solve()

