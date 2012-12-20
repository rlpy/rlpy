import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
from Pendulum import *

#####################################################################
# Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
# (See 1Link implementation by Lagoudakis & Parr, 2003)
#
# ---OBJECTIVE---
# Reward is 0 everywhere, -1 if theta falls below horizontal.
# This is also the terminal condition for an episode.
#
# theta:    [-pi/2, pi/2]
# thetaDot: [-2,2]    Limits imposed per L & P 2003
#
# Pendulum starts unmoving, straight up, theta = 0.
# (see Pendulum parent class for coordinate definitions).
#
#####################################################################

## @author: Robert H. Klein
class Pendulum_InvertedBalance(Pendulum):
    
    # Domain constants per 1Link implementation by Lagoudakis & Parr, 2003.
    AVAIL_FORCE         = array([-50,0,50]) # Newtons, N - Torque values available as actions
    MASS_PEND           = 2.0   # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = 8.0   # kilograms, kg - Mass of cart
    LENGTH              = 1.0   # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = 9.8   # m/s^2 - gravitational constant
    dt                  = 0.1   # Time between steps
    force_noise_max     = 10    # Newtons, N - Maximum noise possible, uniformly distributed

    FELL_REWARD         = -1    # Reward received when the pendulum falls below the horizontal
    ANGLE_LIMITS        = [-pi/2.0, pi/2.0] # Limit on theta (Note that this may affect your representation's discretization)
    ANGULAR_RATE_LIMITS = [-2, 2] # Limits on pendulum rate, per 1Link of Lagoudakis & Parr
                                # NOTE that L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
                                # saturates them frequently when falling; more realistic to use 2*pi.
    episodeCap          = 3000    # Max number of steps per trajectory

    # For Visual Stuff
    MAX_RETURN = 0
    MIN_RETURN = -1

    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(Pendulum_InvertedBalance,self).__init__(logger)
    def s0(self):    
        # Returns the initial state, pendulum vertical
        return array([0,0])
    def _getReward(self, s, a):
        # Return the reward earned for this state-action pair
        # On this domain, reward of -1 is given for failure, |angle| exceeding pi/2
        return self.FELL_REWARD if self.isTerminal(s) else 0
    def isTerminal(self,s):
        return not (-pi/2.0 < s[StateIndex.THETA] < pi/2.0) # per L & P 2003
    
if __name__ == '__main__':
    random.seed(0)
    p = Pendulum_InvertedBalance();
    p.test(1000)
