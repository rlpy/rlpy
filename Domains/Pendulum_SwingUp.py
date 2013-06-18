#See http://acl.mit.edu/RLPy for documentation and future code updates

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
from Pendulum import *

#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
# ---OBJECTIVE--- \n
# Reward is 1 within the goal region, 0 elsewhere. \n
# There is no terminal condition aside from episodeCap. \n
#
# Pendulum starts straight down, theta = pi 
# (see Pendulum parent class for coordinate definitions). \n
#
# The objective is to get and then keep the pendulum in the goal
# region for as long as possible, with +1 reward for
# each step in which this condition is met; the expected
# optimum then is to swing the pendulum vertically and
# hold it there, collapsing the problem to Pendulum_InvertedBalance
# but with much tighter bounds on the goal region. \n
#
# (See 1Link implementation by Lagoudakis & Parr, 2003)
#
#####################################################################

class Pendulum_SwingUp(Pendulum):
    
    # Domain constants [temporary, per Lagoudakis & Parr 2003, for InvertedBalance Task]
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
	
	## Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1             
	## Goal region for reward [temporary values]
    GOAL_LIMITS         = [-pi/6, pi/6] 
	## Limit on theta
    ANGLE_LIMITS        = [-pi, pi]     
	## Limits on pendulum rate [temporary values, copied from InvertedBalance task of RL Community]
    ANGULAR_RATE_LIMITS = [-3*pi, 3*pi] 
	## Max number of steps per trajectory
    episodeCap          = 300          
    # For Visual Stuff
    MAX_RETURN = episodeCap*GOAL_REWARD
    MIN_RETURN = 0
    gamma               = .9   # 
    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(Pendulum_SwingUp,self).__init__(logger)  
    def s0(self):    
        #Returns the initial state: pendulum straight up and unmoving.
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
