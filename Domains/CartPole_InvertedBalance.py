import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *
from CartPole import *

#####################################################################
# Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
# (See CartPole implementation in the RL Community,
# http://library.rl-community.org/wiki/CartPole)
#
# NOTE the different reward scheme from Pendulum_InvertedBalance
#
# ---OBJECTIVE---
# Reward 1 is received on each timestep spent within the goal region,
# zero elsewhere.
# This is also the terminal condition.
#
# RL Community has the following bounds for failure:
# theta: [-12, 12] degrees  -->  [-pi/15, pi/15]
# x: [-2.4, 2.4] meters
#
# Pendulum starts straight up, theta = 0, with the
# cart at x = 0.
# (see CartPole parent class for coordinate definitions).
#
# Note that the reduction in angle limits affects the resolution
# of the discretization of the continuous space.
#
# Note that if unbounded x is desired, set x (and/or xdot) limits to
# be [float("-inf"), float("inf")]
#
#####################################################################

## @author: Robert H. Klein
class CartPole_InvertedBalance(CartPole):
    
    # Domain constants per RL Community / RL_Glue CartPole implementation.
    # (http://code.google.com/p/rl-library/wiki/CartpoleJava)
    AVAIL_FORCE         = array([-10,0,10]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    MASS_PEND           = 0.1   # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = 1.0   # kilograms, kg - Mass of cart
    LENGTH              = 1.0   # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = 9.8   # m/s^2 - gravitational constant
    dt                  = 0.02  # Time between steps
    force_noise_max     = 1     # Newtons, N - Maximum noise possible, uniformly distributed
    
    GOAL_REWARD         = 1               # Reward received on each step the pendulum is in the goal region
    ANGLE_LIMITS        = [-pi/15, pi/15] # rad - Limits on pendulum angle per RL Community CartPole (NOTE we wrap the angle at 2*pi)
    ANGULAR_RATE_LIMITS = [-6.0, 6.0]     # Limits on pendulum rate [per RL Community CartPole]
    POSITON_LIMITS 		= [-2.4, 2.4] 	  # m - Limits on cart position [Per RL Community CartPole]
    VELOCITY_LIMITS 	= [-6.0, 6.0] 	  # m/s - Limits on cart velocity [per RL Community CartPole]   

    episodeCap          = 3000      # Max number of steps per trajectory
    
    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        super(CartPole_InvertedBalance,self).__init__(logger)
    
    def s0(self):    
        # Returns the initial state, pendulum vertical
        return array([0,0,0,0])
    
    ## Return the reward earned for this state-action pair
    # On this domain, reward of 1 is given for each step spent within goal region.
    # There is no specific penalty for failure.
    def _getReward(self, s, a):
        return self.GOAL_REWARD if -pi/6 < s[StateIndex.THETA] < pi/6 else 0
    
    def isTerminal(self,s):
        return (not (-pi/15 < s[StateIndex.THETA] < pi/15) or \
                not (-2.4    < s[StateIndex.X]     < 2.4))


if __name__ == '__main__':
    random.seed(0)
    p = CartPole_InvertedBalance();
    p.test(1000)
    