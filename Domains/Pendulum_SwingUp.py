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
# Reward is 1 within the goal region, 0 elsewhere.
# There is no terminal condition aside from episodeCap.
#
# Pendulum starts straight down, theta = pi
# (see Pendulum parent class for coordinate definitions).
#
# The objective is to get and then keep the pendulum in the goal
# region for as long as possible, with +1 reward for
# each step in which this condition is met; the expected
# optimum then is to swing the pendulum vertically and
# hold it there, collapsing the problem to Pendulum_InvertedBalance
# but with much tighter bounds on the goal region.
#
#####################################################################

## @author: Robert H. Klein
class Pendulum_SwingUp(Pendulum):
    
    # Domain constants [temporary, per Lagoudakis & Parr 2003, for InvertedBalance Task]
    AVAIL_FORCE         = array([-50,0,50]) # Newtons, N - Torque values available as actions
    MASS_PEND           = 2.0   # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = 8.0   # kilograms, kg - Mass of cart
    LENGTH              = 1.0   # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = 9.8   # m/s^2 - gravitational constant
    dt                  = 0.1   # Time between steps
    force_noise_max     = 10    # Newtons, N - Maximum noise possible, uniformly distributed
    
    GOAL_REWARD         = 1             # Reward received on each step the pendulum is in the goal region
    GOAL_LIMITS         = [-pi/6, pi/6] # Goal region for reward [temporary values]
    ANGLE_LIMITS        = [-pi, pi]     # Limit on theta
    ANGULAR_RATE_LIMITS = [-3*pi, 3*pi] # Limits on pendulum rate [temporary values, copied from InvertedBalance task of RL Community]
    episodeCap          = 3000          # Max number of steps per trajectory
    
    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(Pendulum_SwingUp,self).__init__(logger)  
    def s0(self):    
        """Returns the initial state: pendulum straight up and unmoving."""
        return array([pi,0])
    def _getReward(self, s, a):
        """Return the reward earned for this state-action pair.
        On this domain, reward of 1 is given for success, which occurs when |theta| < pi/6"""
        return self.GOAL_REWARD if self.GOAL_LIMITS[0] < s[StateIndex.THETA] < self.GOAL_LIMITS[1] else 0
    def isTerminal(self,s):
        return False
    
if __name__ == '__main__':
    random.seed(0)
    p = Pendulum_SwingUp();
    p.test(1000)
