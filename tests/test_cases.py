"""Nosetests for testing the use cases contained in the cases directory."""
from __future__ import print_function

import numpy as np
from nose.tools import ok_, eq_
import glob
import logging
from rlpy.Tools.run import read_setting_content

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


def test_all_100_step():
    for fn in glob.glob("examples/**/*.py"):
        if fn in glob.glob("examples/tutorial/*"):
            continue; # This is a tutorial file, just skip
        if "run" in fn or "plot" in fn:
            continue
        yield check_running, fn, 100


def check_running(filename, steps):
    content = read_setting_content(filename)
    local = {}
    exec(content, local)
    make_experiment = local["make_experiment"]
    exp = make_experiment(exp_id=1, path="./Results/Temp/nosetests")
    exp.max_steps = steps
    exp.config_logging = False
    exp.num_policy_checks = 2
    exp.checks_per_policy = 1
    exp.run()


def test_tutorial():
    content = read_setting_content("examples/tutorial/gridworld.py")
    local = {}
    exec(content, local)
    make_experiment = local["make_experiment"]
    exp = make_experiment(exp_id=1, path="./Results/Temp/nosetests")
    exp.config_logging = False
    exp.run()
    print("Final Return", exp.result["return"][-1])
    assert exp.result["return"][-1] > 0.4
