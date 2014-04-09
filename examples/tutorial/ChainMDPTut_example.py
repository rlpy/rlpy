#!/usr/bin/env python
"""
Domain Tutorial for RLPy
=================================

Assumes you have created the ChainMDPTut.py domain according to the
tutorial and placed it in the Domains/ directory.
Tests the agent using SARSA with a tabular representation.
"""
__author__ = "Robert H. Klein"
from Domains import ChainMDPTut
from Tools import Logger
from Agents import SARSA
from Representations import Tabular
from Policies import eGreedy
from Experiments import Experiment
import os


def make_experiment(id=1, path="./Results/Tutorial/ChainMDPTut-SARSA"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    logger = Logger()

    ## Domain:
    chainSize = 50
    domain = ChainMDPTut(chainSize=chainSize, logger=logger)

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain, logger)

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
