#!/usr/bin/env python

__author__ = "William Dabney"

from Domains import GridWorld
from Tools import Logger
from MDPSolvers import ValueIteration
from Representations import QTabular
from Policies import GibbsPolicy
from Experiments import MDPSolverExperiment
import os

def make_experiment(id=1, path="./Results/Temp", show=False):
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
    maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    domain = GridWorld(maze, noise=0.3, logger=logger)

    ## Representation
    representation  = QTabular(domain, logger, discretization=20)

    ## Agent
    agent = ValueIteration(id, representation, domain, logger, project_path=path, show=show)

    return MDPSolverExperiment(agent, domain)

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.run(show=True)

