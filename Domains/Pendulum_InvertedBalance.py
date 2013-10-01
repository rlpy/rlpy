"""Pendulum inverted balance domain."""

from Tools import *
from Pendulum import Pendulum, StateIndex


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Alborz Geramifard", "Robert H. Klein"]


#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
#
# ---OBJECTIVE--- \n
# Reward is 0 everywhere, -1 if theta falls below horizontal. \n
# This is also the terminal condition for an episode. \n
#
# theta:    [-pi/2, pi/2] \n
# thetaDot: [-2,2]    Limits imposed per L & P 2003 \n
#
# Pendulum starts unmoving, straight up, theta = 0.
# (see Pendulum parent class for coordinate definitions). \n
#
# (See 1Link implementation by Lagoudakis & Parr, 2003)
#####################################################################

class Pendulum_InvertedBalance(Pendulum):

    # Domain constants per 1Link implementation by Lagoudakis & Parr, 2003.
	## Newtons, N - Torque values available as actions
    AVAIL_FORCE         = array([-50,0,50])
	## kilograms, kg - Mass of the pendulum arm
    MASS_PEND           = 2.0
	## kilograms, kg - Mass of cart
    MASS_CART           = 8.0
	## meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH              = 1.0
	## m/s^2 - gravitational constant
    ACCEL_G             = 9.8
	## Time between steps
    dt                  = 0.1
	## Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max     = 10

	## Reward received when the pendulum falls below the horizontal
    FELL_REWARD         = -1
	## Limit on theta (Note that this may affect your representation's discretization)
    ANGLE_LIMITS        = [-pi/2.0, pi/2.0]
	## Limits on pendulum rate, per 1Link of Lagoudakis & Parr
    ANGULAR_RATE_LIMITS = [-2., 2.] # NOTE that L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
                                # saturates them frequently when falling; more realistic to use 2*pi.
	## Max number of steps per trajectory (default 3000)
    episodeCap          = None
	## Based on Parr 2003 and ICML 11 (RL-Matlab)
    gamma               = .95

    # For Visual Stuff
    MAX_RETURN = 0
    MIN_RETURN = -1

    def __init__(self, episodeCap = 3000, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        self.episodeCap = episodeCap
        super(Pendulum_InvertedBalance,self).__init__(logger)

    def s0(self):
        # Returns the initial state, pendulum vertical
        # Initial state is uniformly randomed between [-.2,.2] for both dimensions
        self.state = (self.random_state.rand(2)*2-1)*0.2
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a):
        # Return the reward earned for this state-action pair
        # On this domain, reward of -1 is given for failure, |angle| exceeding pi/2
        return self.FELL_REWARD if self.isTerminal() else 0

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return not (-pi/2.0 < s[StateIndex.THETA] < pi/2.0) # per L & P 2003
