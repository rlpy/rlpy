from rlpy.Representations import Tabular
from rlpy.Domains import FiftyChain
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import logging


def _make_experiment(exp_id=1, path="./Results/Tmp/test_FiftyChain"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    ## Domain:
    domain = FiftyChain()

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    agent = SARSA(representation=representation, policy=policy,
                  discount_factor=domain.discount_factor,
                       learn_rate=0.1)
    checks_per_policy = 3
    max_steps = 50
    num_policy_checks = 3
    experiment = Experiment(**locals())
    return experiment


from .helpers import check_seed_vis
def test_seed():
    check_seed_vis(_make_experiment)

def test_errs():
    """ Ensure that we can call custom methods without error """

    domain = FiftyChain()

    # [[Test storeOptimalPolicy()]]
    domain.storeOptimalPolicy()

    # [[Test L_inf_distance_to_V_star]]
    exp = _make_experiment(exp_id=1)
    exp.config_logging = False
    exp.run(visualize_steps=False,
            visualize_learning=False,
            visualize_performance=0)
    distToVStar = exp.domain.L_inf_distance_to_V_star(exp.agent.representation)
    assert distToVStar is not np.NAN # must use `is not` because of np.NaN vs np.NAN vs...
    assert distToVStar is not np.Inf

def test_transitions():
    """
    Ensure that actions result in expected state transition behavior.
    Note that if the agent attempts to leave the edge
    (select LEFT from s0 or RIGHT from s49) then the state should not change.
    NOTE: assume p_action_failure is only noise term.

    """
    # [[initialize domain]]
    domain = FiftyChain()
    domain.p_action_failure = 0.0 # eliminate stochasticity
    dummyS = domain.s0()
    domain.state = 2 # state s2
    left = domain.LEFT
    right = domain.RIGHT
    goals = domain.GOAL_STATES

    # Check basic step
    r,ns,terminal,possibleA = domain.step(left)
    assert ns == 1 and terminal == False
    assert np.all(possibleA == np.array([left, right])) # all actions available
    if ns in goals: assert r > 0
    else: assert r <= 0

    # Check another basic step
    r,ns,terminal,possibleA = domain.step(left)
    assert ns == 0 and terminal == False
    assert np.all(possibleA == np.array([left, right])) # all actions available
    if ns in goals: assert r > 0
    else: assert r <= 0

    # Ensure state does not change or wrap around and that all actions
    # remain availableon corner case, per domain spec
    r,ns,terminal,possibleA = domain.step(left)
    assert ns == 0 and terminal == False
    assert np.all(possibleA == np.array([left, right])) # all actions available
    if ns in goals: assert r > 0
    else: assert r <= 0

    # A final basic step
    r,ns,terminal,possibleA = domain.step(right)
    assert ns == 1 and terminal == False
    assert np.all(possibleA == np.array([left, right])) # all actions available
    if ns in goals: assert r > 0
    else: assert r <= 0

if __name__ == '__main__':
    test_seed()
