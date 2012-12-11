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

class Pendulum_SwingUp(Pendulum):
    # Domain constants
    GOAL_REWARD         = 1            # Reward received on each step the pendulum is in the goal region
    ANGLE_LIMITS        = [-pi, pi] # Limit on theta (used for discretization)
    ANGULAR_RATE_LIMITS = [-6, 6]       # Limits on pendulum rate, per 1Link of Lagoudakis & Parr
                                # NOTE that those rate limits are actually unphysically slow; more realistic to use 2*pi
    episodeCap          = 3000          # Max number of steps per trajectory
    
    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(Pendulum_SwingUp,self).__init__(logger)
    
    def s0(self):    
        """Returns the initial state: pendulum straight up and unmoving."""
        return array([pi,0])
    
    def _getReward(self, s, a):
        """Return the reward earned for this state-action pair.
        On this domain, reward of -1 is given for failure, which occurs when |theta| exceeds pi/2"""
        return self.GOAL_REWARD if -pi/6 < s[StateIndex.THETA] < pi/6 else 0
    
    def isTerminal(self,s):
        return False
    
if __name__ == '__main__':
    random.seed(0)
    p = Pendulum_SwingUp();
    p.test(1000)
