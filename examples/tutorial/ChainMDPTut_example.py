#!/usr/bin/env python
"""
Domain Tutorial for RLPy
=================================

Assumes you have created the ChainMDPTut.py domain according to the
tutorial and placed it in the current working directory.
Tests the agent using SARSA with a tabular representation.
"""
__author__ = "Robert H. Klein"
from rlpy.Agents import SARSA
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
from ChainMDPTut import ChainMDPTut
import os
import logging


def make_experiment(exp_id=1, path="./Results/Tutorial/ChainMDPTut-SARSA"):
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
    chainSize = 50
    domain = ChainMDPTut(chainSize=chainSize)
    opt["domain"] = domain

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    opt["agent"] = SARSA(policy=policy, representation=representation, discount_factor=domain.discount_factor, initial_learn_rate=0.1)
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
