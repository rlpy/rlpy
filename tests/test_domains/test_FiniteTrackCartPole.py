from rlpy.Representations import IncrementalTabular
from rlpy.Domains.FiniteTrackCartPole import FiniteTrackCartPole, \
FiniteCartPoleBalance, FiniteCartPoleSwingUp, FiniteCartPoleBalanceModern
from rlpy.Agents.TDControlAgent import SARSA
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import logging

def _make_experiment(domain, exp_id=1,
                     path="./Results/Tmp/test_FiniteTrackCartPole"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param domain: the domain object to be used in the experiment
    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = IncrementalTabular(domain)

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

def _checkSameExperimentResults(exp1, exp2):
    """ Returns False if experiments gave same results, true if they match. """
    if not np.all(exp1.result["learning_steps"] == exp2.result["learning_steps"]):
        # Same number of steps before failure (where applicable)
        print 'LEARNING STEPS DIFFERENT'
        print exp1.result["learning_steps"]
        print exp2.result["learning_steps"]
        return False
    if not np.all(exp1.result["return"] == exp2.result["return"]):
        # Same return on each test episode
        print 'RETURN DIFFERENT'
        print exp1.result["return"]
        print exp2.result["return"]
        return False
    if not np.all(exp1.result["steps"] == exp2.result["steps"]):
        # Same number of steps taken on each training episode
        print 'STEPS DIFFERENT'
        print exp1.result["steps"]
        print exp2.result["steps"]
        return False
    return True

def test_seed_balance():
    """ Ensure that providing the same random seed yields same result """
    
    domain = FiniteCartPoleBalance()
    # [[initialize and run experiment without visual]]
    expNoVis = _make_experiment(domain=domain, exp_id=1)
    expNoVis.run(visualize_steps=False,
            visualize_learning=False,
            visualize_performance=0)
    
    # [[initialize and run experiment with visual]]
    expVis1 = _make_experiment(domain=domain, exp_id=1)
    expVis1.run(visualize_steps=True,
            visualize_learning=False,
            visualize_performance=1)
    
    expVis2 = _make_experiment(domain=domain, exp_id=1)
    expVis2.run(visualize_steps=False,
            visualize_learning=True,
            visualize_performance=1)
    
    # [[assert get same results]]
    assert _checkSameExperimentResults(expNoVis, expVis1)
    assert _checkSameExperimentResults(expNoVis, expVis2)
    
def test_seed_swingup():
    domain = FiniteCartPoleSwingUp()
    # [[initialize and run experiment without visual]]
    expNoVis = _make_experiment(domain=domain, exp_id=1)
    expNoVis.run(visualize_steps=False,
            visualize_learning=False,
            visualize_performance=0)
    
    # [[initialize and run experiment with visual]]
    expVis1 = _make_experiment(domain=domain, exp_id=1)
    expVis1.run(visualize_steps=True,
            visualize_learning=False,
            visualize_performance=1)
    
    #FIXME - Experiment line 315 cannot make deepcopy of matplotlib object
#     expVis2 = _make_experiment(domain=domain, exp_id=1)
#     expVis2.run(visualize_steps=False,
#             visualize_learning=True,
#             visualize_performance=1)
    
    # [[assert get same results]]
    assert _checkSameExperimentResults(expNoVis, expVis1)
#     assert _checkSameExperimentResults(expNoVis, expVis2)

def test_physicality():
    """
    Test coordinate system [vertical is 0]
        1) gravity acts in proper direction based on origin
        2) force actions behave as expected in that frame
    """
    # Apply a bunch of non-force actions, ensure that monotonically accelerate

    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2
    domain = FiniteCartPoleBalanceModern()
    domain.force_noise_max = 0 # no stochasticity in applied force
    
    # Positive angle (right)
    s = np.array([1.0 * np.pi/180.0, 0.0, 0.0, 0.0]) # pendulum slightly right
    domain.state = s
    
    for i in np.arange(5): # do for 5 steps and ensure works
        energ = (s)
        domain.step(NO_FORCE)
        assert np.all(domain.state[0:2] > s[0:2]) # angle and angular velocity increase
        # no energy should enter or leave system under no force action
        assert np.abs(_cartPoleEnergy(s) - _cartPoleEnergy(domain.state)) < 0.01
        s = domain.state
    
    # Negative angle (left)
    s = np.array([-1.0 * np.pi/180.0, 0.0, 0.0, 0.0]) # pendulum slightly right
    domain.state = s
    
    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.all(domain.state[0:2] < s[0:2]) # angle and angular velocity increase
        # no energy should enter or leave system under no force action
        assert np.abs(_cartPoleEnergy(s) - _cartPoleEnergy(domain.state)) < 0.01
        s = domain.state
    
    
    # Start vertical, ensure that force increases angular velocity in direction
    # Negative force on cart, yielding positive rotation
    s = np.array([0.0, 0.0, 0.0, 0.0])
    domain.state = s
    
    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(LEFT_FORCE)
        assert np.all(domain.state[0:2] > s[0:2]) # angle and angular velocity increase
        s = domain.state
        
    # Positive force on cart, yielding negative rotation
    s = np.array([0.0, 0.0, 0.0, 0.0])
    domain.state = s
    
    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(RIGHT_FORCE)
        assert np.all(domain.state[0:2] < s[0:2]) # angle and angular velocity increase
        s = domain.state
    # Ensure that reward racks up while in region
    
def test_physicality_hanging():
    """
    Test that energy does not spontaneously enter system
    """
    # Apply a bunch of non-force actions, ensure that monotonically accelerate

#     LEFT_FORCE = 0, RIGHT_FORCE = 1
    LEFT_FORCE = 0
    NO_FORCE = 1
    RIGHT_FORCE = 2
    domain = FiniteCartPoleBalance()
    domain.force_noise_max = 0 # no stochasticity in applied force
    
    # Positive angle (right)
    s = np.array([179.6 * np.pi/180.0, 0.0, -2.0, 0.0]) # pendulum hanging down
    domain.state = s
    
    for i in np.arange(5): # do for 5 steps and ensure works
        domain.step(NO_FORCE)
        assert np.abs(domain.state[0]) <=179.5 # angle does not increase
        assert np.abs(domain.state[1]) <= 0.1 # angular rate does not increase
        # no energy should enter or leave system under no force action
        assert np.abs(_cartPoleEnergy(s) - _cartPoleEnergy(domain.state)) < 0.01
        s = domain.state
        s = domain.state
    
    # Ensure that running out of x bounds causes experiment to terminate
    assert domain.isTerminal(s=np.array([0.0, 0.0, 2.5, 0.0]))
    assert domain.isTerminal(s=np.array([0.0, 0.0, -2.5, 0.0]))
    assert domain.isTerminal(s=np.array([0.0, 0.0, 0.0, 2.5]))
    assert domain.isTerminal(s=np.array([0.0, 0.0, 0.0, -2.5]))
    
def _cartPole_energy(s):
    """
    energy equation:
    http://robotics.itee.uq.edu.au/~metr4202/tpl/t10-Week12-pendulum.pdf
    """
    cartEnergy = 0.5 * domain.MASS_CART * s[3] ** 2
    pendEnergy = 0.5*domain.MASS_PEND * ( (s[2] + domain.LENGTH*sin(s[0])) ** 2 \
                                        + (domain.LENGTH*cos(s[0])) ** 2 )
    pendEnergy = pendEnergy - domain.MASS_PEND*9.81*domain.LENGTH*cos(s[0])
    
    return cartEnergy + pendEnergy