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
    ANGULAR_RATE_LIMITS = [-2*pi, 2*pi] # NOTE that L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
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
        s0 = (random.rand(2)*2-1)*0.2
        return s0 
            
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
