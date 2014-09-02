from rlpy.Representations import IncrementalTabular
from rlpy.Domains import SystemAdministrator
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from rlpy.Tools import __rlpy_location__

from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment

from .helpers import check_seed_vis
import os

def _make_experiment(exp_id=1, path="./Results/Tmp/test_SystemAdministrator"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    ## Domain:
    domain = SystemAdministrator()

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = IncrementalTabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.2)

    ## Agent
    agent = SARSA(representation=representation, policy=policy,
                  discount_factor=domain.discount_factor,
                       learn_rate=0.1)
    checks_per_policy = 2
    max_steps = 20
    num_policy_checks = 2
    experiment = Experiment(**locals())
    return experiment

def _checkSameExperimentResults(exp1, exp2):
    """ Returns False if experiments gave same results, true if they match. """
    if not np.array_equiv(exp1.result["learning_steps"], exp2.result["learning_steps"]):
        # Same number of steps before failure (where applicable)
        return False
    if not np.array_equiv(exp1.result["return"], exp2.result["return"]):
        # Same return on each test episode
        return False
    if not np.array_equiv(exp1.result["steps"], exp2.result["steps"]):
        # Same number of steps taken on each training episode
        return False
    return True


def test_seed():
    check_seed_vis(_make_experiment)

def test_errs():
    """ Ensure that we can call custom methods without error """

    default_map_dir = os.path.join(
        __rlpy_location__,
        "Domains",
        "SystemAdministratorMaps")
    domain = SystemAdministrator(networkmapname=os.path.join(
                default_map_dir, "20MachTutorial.txt"))

    # loadNetwork() is called by __init__
    # setNeighbors() is run by loadNetwork

def test_transitions():
    """
    Ensure that actions result in expected state transition behavior.
    """
    # [[manually set state, manually turn off stochasticity ie deterministic,
    # and observe transitions, reward, etc.]]
    default_map_dir = os.path.join(
        __rlpy_location__,
        "Domains",
        "SystemAdministratorMaps")
    domain = SystemAdministrator(networkmapname=os.path.join(
                default_map_dir, "5Machines.txt"))
    dummyS = domain.s0()
    up = domain.RUNNING # shorthand
    down = domain.BROKEN # shorthand

    state = np.array([up for dummy in xrange(0, domain.state_space_dims)])
    domain.state = state.copy()
    a = 5 # =n on this 5-machine map, ie no action
    ns = state.copy()

    # Test that no penalty is applied for a non-reboot action
    r, ns, t, pA = domain.step(a)
    numWorking = len(np.where(ns == up)[0])
    if domain.IS_RING and domain.state[0] == self.RUNNING:
        r = r-1 # remove the correctin for rings / symmetry
    assert r == numWorking

    # Test that penalty is applied for reboot
    r, ns, t, pA = domain.step(0) # restart computer 0
    numWorking = len(np.where(ns == up)[0])
    if domain.IS_RING and domain.state[0] == self.RUNNING:
        r = r-1 # remove the correctin for rings / symmetry
    assert r == numWorking + domain.REBOOT_REWARD


    while np.all(ns == up):
        r, ns, t, pA = domain.step(a)
    # now at least 1 machine has failed

    domain.P_SELF_REPAIR = 0.0
    domain.P_REBOOT_REPAIR = 0.0

    # Test that machine remains down when no reboot taken
    fMachine = np.where(ns == down)[0][0]
    r, ns, t, pA = domain.step(fMachine)
    assert ns[fMachine] == down

    # Test that machine becomes up when reboot taken
    domain.P_REBOOT_REPAIR = 1.0
    r, ns, t, pA = domain.step(fMachine)
    assert ns[fMachine] == up

