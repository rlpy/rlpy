from rlpy.Representations import IncrementalTabular
from rlpy.Domains import PST
from rlpy.Domains.PST import UAVLocation, ActuatorState, SensorState, UAVAction
from rlpy.Domains.PST import StateStruct
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from rlpy.Tools import __rlpy_location__
from rlpy.Tools import vec2id, id2vec
import os

from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import logging

def _make_experiment(exp_id=1, path="./Results/Tmp/test_PST"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    ## Domain:
    NUM_UAV=3
    domain = PST(NUM_UAV=NUM_UAV)

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = IncrementalTabular(domain)

    ## Policy
    policy = eGreedy(representation, epsilon=0.1)

    ## Agent
    agent = SARSA(representation=representation, policy=policy,
                  discount_factor=domain.discount_factor,
                       learn_rate=0.1)
    checks_per_policy = 3
    max_steps = 50
    num_policy_checks = 3
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
    """ Ensure that providing the same random seed yields same result """
    # [[initialize and run experiment without visual]]
    expNoVis = _make_experiment(exp_id=1)
    expNoVis.run(visualize_steps=False,
            visualize_learning=False,
            visualize_performance=0)
    
    # [[initialize and run experiment with visual]]
    expVis1 = _make_experiment(exp_id=1)
    expVis1.run(visualize_steps=True,
            visualize_learning=False,
            visualize_performance=1)
    
    expVis2 = _make_experiment(exp_id=1)
    expVis2.run(visualize_steps=False,
            visualize_learning=True,
            visualize_performance=1)
    
    # [[assert get same results]]
    assert _checkSameExperimentResults(expNoVis, expVis1)
    assert _checkSameExperimentResults(expNoVis, expVis2)

def test_errs():
    """ Ensure that we can call custom methods without error """
    
    domain = PST(NUM_UAV=2)
    dummyState = domain.s0()
    
    # state2Struct
    rlpy_state = [1,2,9,3,1,0,1,1]
    internState = domain.state2Struct(rlpy_state)
    assert np.all(internState.locations == [1,2])
    assert np.all(internState.fuel == [9,3])
    assert np.all(internState.actuator == [1,0])
    assert np.all(internState.sensor == [1,1])
    
    # struct2State
    locs = np.array([1,2])
    fuel = np.array([9,3])
    act = np.array([1,0])
    sens = np.array([1,1])
    sStruct = StateStruct(locs, fuel, act, sens)
    assert np.all(domain.struct2State(sStruct) == [1,2,9,3,1,0,1,1])
    
    # properties2StateVec
    locs = np.array([1,2])
    fuel = np.array([9,3])
    act = np.array([1,0])
    sens = np.array([1,1])
    assert np.all(domain.properties2StateVec(locs, fuel, act, sens) == [1,2,9,3,1,0,1,1])

def test_transitions():
    """
    Ensure that actions result in expected state transition behavior.
    Test:
        1) Actuator and sensor failure, associated lack of reward
        2) Refuel
        3) Repair
        4) Presence of reward iff a UAV is in COMMS *and* SURVEIL
        5) UAV Crash because of lack of fuel
        
    """
    NUM_UAV = 2
    nPosActions = 3 # = UAVAction.SIZE
    actionLimits = nPosActions * np.ones(NUM_UAV, dtype='int')
    
    # Test p=1 actuator failure when not at base
    domain = PST(NUM_UAV=NUM_UAV)
    dummyS = domain.s0()
    
    domain.P_ACT_FAIL = 0.0
    domain.P_SENSOR_FAIL = 1.0
    
    locs = np.array([UAVLocation.COMMS, UAVLocation.COMMS])
    fuel = np.array([10,10])
    act = np.array([ActuatorState.RUNNING, ActuatorState.RUNNING])
    sens = np.array([SensorState.RUNNING, SensorState.RUNNING])
    actionVec = np.array([UAVAction.LOITER, UAVAction.LOITER])
    a = vec2id(actionVec, actionLimits)
    domain.state = domain.properties2StateVec(locs, fuel, act, sens)
    r, ns, t, possA = domain.step(a)
    # Assert that only change was reduction in fuel and failure of sensor
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel-1, \
                                                   act, np.array([0,0])))
    
    # Test location change movement
    actionVec = np.array([UAVAction.ADVANCE, UAVAction.ADVANCE])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs+1, fuel-2, \
                                                   act, np.array([0,0])))
    
    # Test p=1 sensor failure when not at base
    domain.FUEL_BURN_REWARD_COEFF = 0.0
    domain.MOVE_REWARD_COEFF = 0.0
    domain.P_ACT_FAIL = 1.0
    actionVec = np.array([UAVAction.RETREAT, UAVAction.LOITER])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs + [0,1], fuel-3, \
                                                   np.array([0,0]), np.array([0,0])))
    
    # Test that no reward was received since the sensor is broken
    assert r == 0
    
    # Test Refuel
    # After action below will be in locs + [-1,1], or REFUEL and SURVEIL 
    # respectively, with 4 fuel units consumed.  Must LOITER to refill fuel though
    actionVec = np.array([UAVAction.RETREAT, UAVAction.RETREAT])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    locs = np.array([UAVLocation.REFUEL, UAVLocation.COMMS])
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel-4, \
                                                   np.array([0,0]), np.array([0,0])))
    # Refuel occurs after loitering
    actionVec = np.array([UAVAction.LOITER, UAVAction.RETREAT])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    fuel = np.array([10,5])
    locs = np.array([UAVLocation.REFUEL, UAVLocation.REFUEL])
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel, \
                                                   np.array([0,0]), np.array([0,0])))
    
    # Test repair [note uav2 was never refueled since never loitered]
    actionVec = np.array([UAVAction.RETREAT, UAVAction.RETREAT])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs-1, fuel-1, \
                                                   np.array([0,0]), np.array([0,0])))
    
    # Repair only occurs after loiter [no fuel burned for BASE/REFUEL loiter
    actionVec = np.array([UAVAction.LOITER, UAVAction.LOITER])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs-1, fuel-1, \
                                                   np.array([1,1]), np.array([1,1])))
    
    # Test comms but no surveillance
    domain.P_ACT_FAIL = 0.0
    domain.P_SENSOR_FAIL = 0.0
    actionVec = np.array([UAVAction.ADVANCE, UAVAction.ADVANCE])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel-2, \
                                                   np.array([1,1]), np.array([1,1])))
    actionVec = np.array([UAVAction.ADVANCE, UAVAction.ADVANCE])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs+1, fuel-3, \
                                                   np.array([1,1]), np.array([1,1])))
    assert r == 0 # no reward because only have comms, no surveil
    
    # add 2 units of extra fuel to each and move
    domain.state = domain.properties2StateVec(locs+1, fuel-1, \
                                                   np.array([1,1]), np.array([1,1]))
    
    # Test surveillance but no comms
    actionVec = np.array([UAVAction.ADVANCE, UAVAction.ADVANCE])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs+2, fuel-2, \
                                                   np.array([1,1]), np.array([1,1])))
    assert r == 0 # no reward because have only surveil, no comms
    
    # Test comms and surveillance
    actionVec = np.array([UAVAction.RETREAT, UAVAction.LOITER])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    locs = np.array([UAVLocation.COMMS, UAVLocation.SURVEIL])
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel-3, \
                                                   np.array([1,1]), np.array([1,1])))
    assert r == 0
    # reward based on "s", not "ns", pickup reward here
    actionVec = np.array([UAVAction.LOITER, UAVAction.LOITER])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    locs = np.array([UAVLocation.COMMS, UAVLocation.SURVEIL])
    assert np.array_equiv(ns, domain.properties2StateVec(locs, fuel-4, \
                                                   np.array([1,1]), np.array([1,1])))
    assert r == domain.SURVEIL_REWARD
    
    # Test crash
    # Since reward based on "s" not "ns", also pickup reward from prev step
    actionVec = np.array([UAVAction.RETREAT, UAVAction.RETREAT])
    a = vec2id(actionVec, actionLimits)
    r, ns, t, possA = domain.step(a)
    assert np.array_equiv(ns, domain.properties2StateVec(locs-1, fuel-5, \
                                                   np.array([1,1]), np.array([1,1])))
    assert t == True
    assert r == domain.CRASH_REWARD + domain.SURVEIL_REWARD
    