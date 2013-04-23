import sys, os
#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *
from CartPole import *

#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
#
# ---OBJECTIVE--- \n
# Reward 1 is received on each timestep spent within the goal region,
# zero elsewhere.
# This is also the terminal condition.
#
# RL Community has the following bounds for failure: \n
# theta: [-12, 12] degrees  -->  [-pi/15, pi/15] \n
# x: [-2.4, 2.4] meters \n
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
# \note The reward scheme is different from that in Pendulum_InvertedBalance
#
# (See CartPole implementation in the RL Community,
# http://library.rl-community.org/wiki/CartPole)
#####################################################################

# @author: Robert H. Klein
class CartPole_InvertedBalance(CartPole):
    ANGLE_LIMITS        = [-pi/15, pi/15] # rad - Limits on pendulum angle per RL Community CartPole (NOTE we wrap the angle at 2*pi)
    
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
        return self.GOAL_REWARD if -pi/15 < s[StateIndex.THETA] < pi/15 else 0
    
    def isTerminal(self,s):
        return (not (-pi/15 < s[StateIndex.THETA] < pi/15) or \
                not (-2.4    < s[StateIndex.X]     < 2.4))


if __name__ == '__main__':
    random.seed(0)
    p = CartPole_InvertedBalance();
    p.test(1000)
    