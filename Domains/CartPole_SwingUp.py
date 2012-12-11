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
# ---OBJECTIVE---
# Reward 1 is received on each timestep spent within the goal region,
# zero elsewhere.
# The episode terminates if x leaves its bounds, [-2.4, 2.4]
#
# Pendulum starts straight down, theta = pi, with the
# cart at x = 0.
# (see CartPole parent class for coordinate definitions).
#
# The objective is to get and then keep the pendulum in the goal
# region for as long as possible, with +1 reward for
# each step in which this condition is met; the expected
# optimum then is to swing the pendulum vertically and
# hold it there, collapsing the problem to Pendulum_InvertedBalance
# but with much tighter bounds on the goal region.
#
#####################################################################

## @todo: THIS DOMAIN is still in-progress.
class CartPole_SwingUp(CartPole):
    
    # Domain constants [temporary, per RL Community / RL_Glue CartPole InvertedBalance Task]
    AVAIL_FORCE         = array([-10,0,10]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    MASS_PEND           = 0.1   # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = 1.0   # kilograms, kg - Mass of cart
    LENGTH              = 1.0   # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = 9.8   # m/s^2 - gravitational constant
    dt                  = 0.02  # Time between steps
    force_noise_max     = 1     # Newtons, N - Maximum noise possible, uniformly distributed
    
    GOAL_REWARD         = 1            # Reward received on each step the pendulum is in the goal region
    ANGLE_LIMITS        = [-pi, pi]    # Limit on theta (used for discretization)
    ANGULAR_RATE_LIMITS = [-6.0, 6.0]  # Limits on pendulum rate [temporary]
    POSITON_LIMITS      = [-2.4, 2.4]  # m - Limits on cart position [temporary]
    VELOCITY_LIMITS     = [-6.0, 6.0]  # m/s - Limits on cart velocity [temporary]  
    episodeCap          = 3000         # Max number of steps per trajectory
    
    def __init__(self, logger = None):
        print 'Caution: This domain is still in the implementation stages.'
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(CartPole_SwingUp,self).__init__(logger)
    
    def s0(self):    
        # Returns the initial state, pendulum vertical
        return array([pi,0,0,0])
    
    
    ## Return the reward earned for this state-action pair
    # On this domain, reward of 1 is given for each step spent within goal region.
    # There is no specific penalty for failure.
    def _getReward(self, s, a):
        return self.GOAL_REWARD if -pi/6 < s[StateIndex.THETA] < pi/6 else 0
    
    def isTerminal(self,s):
        return not (-2.4 < s[StateIndex.X] < 2.4)
    
if __name__ == '__main__':
    random.seed(0)
    p = CartPole_SwingUp();
    p.test(1000)
