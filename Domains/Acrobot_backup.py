#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# TODO currently duplicate L1, self.L1 etc - integrate these.s
# TODO change variable names to match, eg torqueJoint, torque vs torque, etc


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

# REQURES matplotlib for rk4 integration
from scipy import integrate # For integrate.odeint (accurate, slow)

##########################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
##########################################################
# This is the acrobot domain described in Sutton + Barto
# Reinforcement Learning: An Introduction (Ch. 11)
# http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node110.html
#
# NOTE: Sutton and Barto set the parameters for I_1 and I_2 along with
# l_1, l_2, M1, and m_2; unfortunately this is not internally consistent
# with their dynamics, as I_1 and I_2 are both fixed given the other quantites
# and their assumptions about gravity acting halfway along the beam at l_c_1 = l_1 / 2, etc.
# We adopt their convention but acknowledge it as unphysical.
#
#
# State: [theta1, thetaDot1, theta2, thetaDot2]. \n
# Actions: [-1, 0, 1]
#
# theta1 = angular position of first (inner) link
# (relative to horizontal at right, 0 rad), and positive ccw. \n
# thetaDot1 = Angular rate of inner link.
# Same for theta2 and thetaDot2, only for the outer link.
#
##########################################################

class Acrobot(Domain):

    DEBUG               = 0     # Set to non-zero to enable print statements

    AVAIL_TORQUE         = array([-1,0,1])
	## kilograms, kg - Mass of the inner arm
    M1             = 1.0
	## kilograms, kg - Mass of the outer arm
    M2             = 1.0
	## meters, m - Physical length of the inner link (note the moment-arm lies at half this distance)
    L1             = 1.0
    ## meters, m - Physical length of the outer link (note the moment-arm lies at half this distance)
    L2             = 1.0
	## m/s^2 - gravitational constant
    ACCEL_G             = 9.8
	## Time between steps
    dt                  = .05
	## Newtons, N - Maximum noise possible, uniformly distributed
    torque_noise_max     = 0

    ANGLE_LIMITS        = [-pi, pi]
    ## Limits on acrobot rate, per 1Link of Lagoudakis & Parr
    ANGULAR_RATE_LIMITS1 = [-4*pi, 4*pi]
    ANGULAR_RATE_LIMITS2 = [-9*pi, 9*pi]
	## Max number of steps per trajectory
    episodeCap          = None

    I1        = 1.0 # Moment of Intertia of link 1
    I2        = 1.0 # Moment of Intertia of link 2

    ## 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)


    # Domain constants computed in __init__
    ## m - Length of the moment-arm to the center of mass, equal to half the link length
    LC1             = 0 # Note that some elsewhere refer to this simply as 'length' somewhat of a misnomer.
    LC2            = 0


    #Visual Stuff
    valueFunction_fig       = None
    policy_fig              = None
    MIN_RETURN              = None # Minimum return possible, used for graphical normalization, computed in init
    MAX_RETURN              = None # Minimum return possible, used for graphical normalization, computed in init
    circle_radius           = 0.05
    ARM_LENGTH              = 1.0
    PENDULUM_PIVOT_X        = 0 # X position is also fixed in this visualization
    PENDULUM_PIVOT_Y        = 0 # Y position of acrobot pivot
    acrobotLink1             = None
    acrobotLink2             = None
    actionArrow             = None
    domain_fig              = None
    Theta_discretization    = 20 #Used for visualizing the policy and the value function
    ThetaDot_discretization = 20 #Used for visualizing the policy and the value function

    GOAL_LIMITS         = [5*pi/6, 7*pi/6] # NOTE REWARD COORDINATE TRANSFORM BELOW

    gamma               = 1.0

    numSimSteps = 4

    def __init__(self, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        #print 'hello'
        sys.stdout.flush()


        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS1, self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS2])
        self.episodeCap = 600


        self.actions_num        = len(self.AVAIL_TORQUE)
        self.continuous_dims    = [StateIndex.THETA1, StateIndex.THETA_DOT1, StateIndex.THETA2, StateIndex.THETA_DOT2]

        self.LC1         = self.L1 / 2.0
        self.LC2        = self.L2 / 2.0
        self._L1Sq        = self.L1 ** 2
        self._L2Sq          = self.L2 ** 2
        self._LC1Sq         = self.LC1 ** 2
        self._LC2Sq         = self.LC2 ** 2

        self.xTicks         = linspace(0,self.Theta_discretization-1,5)
        self.xTicksLabels   = around(linspace(self.statespace_limits[0,0]*180/pi,self.statespace_limits[0,1]*180/pi,5)).astype(int)
        #self.xTicksLabels   = ["$-\\pi$","$-\\frac{\\pi}{2}$","$0$","$\\frac{\\pi}{2}$","$\\pi$"]
        self.yTicks         = [0,self.Theta_discretization/4.0,self.ThetaDot_discretization/2.0,self.ThetaDot_discretization*3/4.0,self.ThetaDot_discretization-1]
        self.yTicksLabels   = around(linspace(self.statespace_limits[1,0]*180/pi,self.statespace_limits[1,1]*180/pi,5)).astype(int)
        #self.yTicksLabels   = ["$-8\\pi$","$-4\\pi$","$0$","$4\\pi$","$8\\pi$"]
        self.DimNames       = ['Theta','Thetadot']
        if self.logger:
            self.logger.log("length1:\t\t%0.2f(m)" % self.L1)
            self.logger.log("length2:\t\t%0.2f(m)" % self.L2)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)

        #print 'initialized'

        super(Acrobot,self).__init__(logger)
    def showDomain(self,s,a = 0):
        # Plot the acrobot and its angles, along with an arc-arrow indicating the
        # direction of torque applied (not including noise!)
        if self.domain_fig == None: # Need to initialize the figure
            self.domain_fig = pl.subplot(1,3,1)
            self.acrobotLink1 = lines.Line2D([],[], linewidth = 3, color='black')
            self.acrobotLink2 = lines.Line2D([],[], linewidth = 3, color='black')

            self.domain_fig.add_line(self.acrobotLink1)
            self.domain_fig.add_line(self.acrobotLink2)
            # Allow room for acrobot to swing without getting cut off on graph
            viewableDistance = self.L1 + self.L2 + 0.5
            self.domain_fig.set_xlim(-viewableDistance, viewableDistance)
            self.domain_fig.set_ylim(-viewableDistance, viewableDistance)
            pl.axis('off')
            self.domain_fig.set_aspect('equal')
            pl.show()

        torqueAction = self.AVAIL_TORQUE[a]
        theta1 = s[StateIndex.THETA1] # Using continuous state
        theta2 = s[StateIndex.THETA2]

        # recall we define 0deg  up, 90 deg right
        torqueJointX = self.PENDULUM_PIVOT_X + self.L1 * sin(theta1)
        torqueJointY = self.PENDULUM_PIVOT_Y + self.L1 * (-cos(theta1))

        arm2X = torqueJointX + self.L2 * sin(theta2)
        arm2Y = torqueJointY + self.L2 * (-cos(theta2))

        # update acrobot links on figure
        self.acrobotLink1.set_data([self.PENDULUM_PIVOT_X, torqueJointX],[self.PENDULUM_PIVOT_Y, torqueJointY])
        self.acrobotLink2.set_data([torqueJointX, arm2X], [torqueJointY, arm2Y])

        if self.DEBUG: print 'Acrobot Position: (\theta 1, \theta2), X,Y position of joint',theta1, theta2, torqueJointX,torqueJointY
        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None

        if torqueAction == 0:
            pass # no torque
        else: # cw or ccw torque
            SHIFT = .5
            if (torqueAction > 0): # counterclockwise torque
                self.actionArrow = fromAtoB(SHIFT/2.0,.5*SHIFT,-SHIFT/2.0,-.5*SHIFT,'k',connectionstyle="arc3,rad=+1.2", ax =self.domain_fig)
            else:# clockwise torque
                self.actionArrow = fromAtoB(-SHIFT/2.0,.5*SHIFT,+SHIFT/2.0,-.5*SHIFT,'r',connectionstyle="arc3,rad=-1.2", ax =self.domain_fig)

        #sleep(self.dt*4)
        pl.draw()
    def showLearning(self,representation):
        pass

    def s0(self):
        # Returns the initial state, acrobot vertical down
        self.state = array([0, 0, 0, 0])
        return self.state.copy()

    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)
    def step(self,a):
    # Simulate one step of the acrpbot after taking torque action a

        torqueAction = self.AVAIL_TORQUE[a]

        # Add noise to the torque action
        if self.torque_noise_max > 0:
            torqueAction += random.uniform(-self.torque_noise_max, self.torque_noise_max)
        else: torqueNoise = 0

        # Now, augment the state with our torque action so it can be passed to _dsdt
        s_augmented = append(self.state, torqueAction)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt*4])
        ns = ns[-1] # only care about final timestep of integration returned by integrator
        ns = ns[0:4] # [theta, thetaDot]

         # wrap angle between -pi and pi (or whatever values assigned to ANGLE_LIMITS)
        ns[StateIndex.THETA1]        = bound(wrap(ns[StateIndex.THETA1],-pi, pi), self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT1]    = bound(ns[StateIndex.THETA_DOT1], self.ANGULAR_RATE_LIMITS1[0], self.ANGULAR_RATE_LIMITS1[1])
        ns[StateIndex.THETA2]        = bound(wrap(ns[StateIndex.THETA2],-pi, pi), self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT2]    = bound(ns[StateIndex.THETA_DOT2], self.ANGULAR_RATE_LIMITS2[0], self.ANGULAR_RATE_LIMITS2[1])
        terminal                    = self.isTerminal(ns)
        reward                      = self._getReward(ns,a)
        self.state = ns.copy()
        return reward, ns, terminal

    #
    # @param s_augmented: {The state at which to compute derivatives, augmented with the current action.
    # Specifically @code (theta,thetaDot,torqueAction) @endcode .}
    #
    # @return: {The derivatives of the state s_augmented, here (thetaDot, thetaDotDot).}
    #
    # Numerically integrate the differential equations forFrom Lagoudakis & Parr, 2003, described in class definition.
    def _dsdt(self, s_augmented, t):


        # This function is needed for ode integration.  It calculates and returns the
        # derivatives at a given state, s.  The last element of s_augmented is the
        # torque action taken, required to compute these derivatives.


        g = self.ACCEL_G

        theta1       = s_augmented[StateIndex.THETA1]
        thetaDot1    = s_augmented[StateIndex.THETA_DOT1]
        theta2       = s_augmented[StateIndex.THETA2]
        thetaDot2    = s_augmented[StateIndex.THETA_DOT2]
        torque       = s_augmented[StateIndex.TORQUE]

        d1     = self.M1*self._LC1Sq + self.M2*(self._L1Sq + self._LC2Sq + 2*self.L1*self.LC2 * cos(theta2)) + self.I1 + self.I2
        d2     = self.M2*(self._LC2Sq+self.L1*self.LC2*cos(theta2)) + self.I2

        phi2   = self.M2*self.LC2*g*cos(theta1+theta2-pi/2.0)
        phi1   = -self.M2*self.L1*self.LC2*thetaDot2*sin(theta2)*(thetaDot2-2*thetaDot1)+(self.M1*self.LC1+self.M2*self.L1)*g*cos(theta1-(pi/2.0))+phi2

        accel2 = (torque+phi1*(d2/d1)-self.M2*self.L1*self.LC2*thetaDot1*thetaDot1*sin(theta2)-phi2)

        thetaDotDot2 = accel2/(self.M2*self._LC2Sq+self.I2-(d2*d2/d1))
        thetaDotDot1 = -(d2*accel2+phi1)/d1

        return (thetaDot1, thetaDotDot1, thetaDot2, thetaDotDot2, 0) # final cell corresponds to action passed i


    ## @param s: state
    #  @param a: action
    #  @return: Reward earned for this state-action pair.
    def _getReward(self, s, a):
        theta1 = s[StateIndex.THETA1]
        theta2 = s[StateIndex.THETA2]
        if(theta1 < 0): theta1 += 2*pi
        if(theta2 < 0): theta2 += 2*pi

        # Return the reward earned for this state-action pair
        # Note that b/c of coordinate transform above, we trivially
        # satisfy lower bound
        if ((self.GOAL_LIMITS[0] < theta1 < self.GOAL_LIMITS[1])):
            if(self.GOAL_LIMITS[0] < theta2 < self.GOAL_LIMITS[1]):
                return 0
#             return -0.5
        return -1

## \cond DEV
# Flexible way to index states in the Pendulum Domain
#
# This class enumerates the different indices used when indexing the state. \n
# e.g. s[StateIndex.THETA] is guaranteed to return the angle state.

class StateIndex:
    THETA1, THETA_DOT1, THETA2, THETA_DOT2 = 0,1,2,3
    TORQUE = 4 # Used by the state augmented with input in dynamics calculations


# \endcond


if __name__ == '__main__':
    random.seed(0)
    #print 'hello?'
    p = Acrobot();
    p.test(1000)
