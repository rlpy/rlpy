"""Nosetests for testing the use cases contained in the cases directory."""

import numpy as np
from nose.tools import ok_, eq_
import glob
from Tools.run import read_setting_content

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"

def test_all_100_step():
    for fn in glob.glob("cases/**/*.py"):
        yield check_running, fn, 100

def test_main_files():
    yield check_running, "main.py", 100

def check_running(filename, steps):
    content = read_setting_content(filename)
    local = {}
    exec content in local
    make_experiment = local["make_experiment"]
    exp = make_experiment(id=1, path="./Results/Temp/nosetests")
    exp.max_steps = steps
    exp.num_policy_checks = 2
    exp.checks_per_policy = 1
    exp.run()
