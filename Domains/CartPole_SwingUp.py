#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.'
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *
from Domain import *
from CartPole import *

#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
#
# ---OBJECTIVE--- \n
# Reward is 1 within the goal region, 0 elsewhere.
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
# (See CartPole implementation in the RL Community,
# http://library.rl-community.org/wiki/CartPole)
#####################################################################

class CartPole_SwingUp(CartPole):
    ANGLE_LIMITS        = [-pi, pi]     # Limit on theta (used for discretization)
    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        super(CartPole_SwingUp,self).__init__(logger)

    def s0(self):
        # Returns the initial state, pendulum vertical
        return array([pi,0,0,0])


    ## Return the reward earned for this state-action pair
    # On this domain, reward of -1 is given for failure, |angle| exceeding pi/2
    def _getReward(self, s, a):
        return self.GOAL_REWARD if -pi/6 < s[StateIndex.THETA] < pi/6 else 0

    def isTerminal(self,s):
        return not (-2.4 < s[StateIndex.X] < 2.4)

class CartPole_SwingUpReal(CartPole_SwingUp):
	## Limits on pendulum rate [per RL Community CartPole]
    ANGULAR_RATE_LIMITS = [-3.0, 3.0]
	## Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1
	## m - Limits on cart position [Per RL Community CartPole]
    POSITON_LIMITS      = [-2.4, 2.4]
	## m/s - Limits on cart velocity [per RL Community CartPole]
    VELOCITY_LIMITS     = [-3.0, 3.0]

    MASS_CART = 0.5
    MASS_PEND = 0.5
    LENGTH = 0.6
    A = .5
    DT = 0.1
    episodeCap          = 400
    gamma = 0.95
    B = 0.1 # Friction between cart and ground

    ## Return the reward earned for this state-action pair
    # On this domain, reward of -1 is given for failure, |angle| exceeding pi/2
    def _getReward(self, s, a):
        if not (-2.4 < s[StateIndex.X] < 2.4):
            return -30
        pen_pos = array([s[StateIndex.X] + self.LENGTH * sin(s[StateIndex.THETA]),
                         self.LENGTH * cos(s[StateIndex.THETA])])
        diff = pen_pos - array([0, self.LENGTH])
        #diff[1] *= 1.5
        return exp(-.5* sum(diff**2) * self.A) - .5

    def _dsdt(self, s_aug, t):
        s = zeros((4))
        s[0] = s_aug[StateIndex.X]
        s[1] = s_aug[StateIndex.X_DOT]
        s[3] = pi - s_aug[StateIndex.THETA]
        s[2] = - s_aug[StateIndex.THETA_DOT]
        a = s_aug[4]
        ds = self._ode(s, t, a, self.MASS_PEND, self.LENGTH, self.MASS_CART, self.B)
        ds_aug = s_aug.copy()
        ds_aug[StateIndex.X] = ds[0]
        ds_aug[StateIndex.X_DOT] = ds[1]
        ds_aug[StateIndex.THETA_DOT] = ds[2]
        ds_aug[StateIndex.THETA] = ds[3]
        return ds_aug

    def _ode(self, s, t, a, m, l, M, b):
        """
        [x, dx, dtheta, theta]
        """

        #cdef double g, c3, s3
        s3 = sin(s[3])
        c3 = cos(s[3])
        g = 9.81
        ds = zeros(4)
        ds[0] = s[1]
        ds[1] = (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a - 4 * b * s[1])\
            / (4 * (M + m) - 3 * m * c3 ** 2)
        ds[2] = (-3 * m * l * s[2] ** 2 * s3*c3 - 6 * (M + m) * g * s3 - 6 * (a - b * s[1]) * c3)\
            / (4 * l * (m + M) - 3 * m * l * c3 ** 2)
        ds[3] = s[2]
        return ds

if __name__ == '__main__':
    random.seed(0)
    p = CartPole_SwingUp();
    p.test(1000)
