from rlpy.Representations import Tabular
from rlpy.Domains import SystemAdministrator
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

def test_seed():
    """ Ensure that providing the same random seed yields same result """
    # [[initialize and run experiment without visual]]
    # [[initialize and run experiment with visual]]
    # [[assert get same results]]
    
    # TODO get from fiftychain

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
    domain.state = state
    a = 5 # =n on this 5-machine map, ie no action
    ns = state
    
    # Test that no penalty is applied for a non-reboot action
    r, ns, t, pA = domain.step(a)
    numWorking = len(np.where(ns == up))
    assert r == numWorking
    
    # Test that penalty is applied for reboot
    r, ns, t, pA = domain.step(0) # restart computer 0
    numWorking = len(np.where(ns == up))
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
    