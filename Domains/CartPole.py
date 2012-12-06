import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

#from scipy import integrate # for integration of state
from matplotlib.mlab import rk4
from matplotlib import lines

#######################################################
# Robert Klein, Alborz Geramifard at MIT, Nov. 30 2012#
#######################################################
# State is angular position of pendulum on [-pi, pi),
# (relative to straight up at 0 rad),and positive cw.
# Positive force acts to the right on the cart
# Angular rate of pendulum, [-15*pi, 15*pi].
# State space is discretized into uniform intervals, default
# 20 intervals each, for a state space size of 20^2=400.
# Actions take the form of torque applied directly
# (STANDS IN CONTRAST TO THE 4-STATE SYSTEM CARTPOLE);
# [-1, 0, 1] are the default available actions.
#
# The objective is to keep the pendulum in the goal
# region for as long as possible; default goal
# region is [pi - pi/6, pi + pi/6], no reward is
# obtained anywhere else.
# Optional noise is added in the form of torques on the
# pendulum.
######################################################

# In comments below, "DPF" refers to Kemal's Java implementation

class StateIndex:
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3
    # X, X_DOT NOT YET COMPUTED - optionally in ds_dt, also return states for xDot and xDotDot,
    # integrate, and track them along with angular states.


class CartPole(Domain):
    DEBUG = 0 # Set to non-zero to enable print statements
    visualize = True # Whether or not to compute x states, for visualization
    
    AVAIL_FORCE = array([-50,0,50]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    
    GOAL_REGION = [-pi/6, pi/6] # radians, rad - Goal region [5pi/6, 7pi/6 per tutorial]
    GOAL_REWARD = 1 # Reward obtained for remaining in the goal region [1 per tutorial]
    # Implicit 0 reward everywhere else - it's all relative.
    ANGLE_LIMITS = [-pi, pi] # rad - Limits on pendulum angle (NOTE we wrap the angle at 2*pi)
    ANGULAR_RATE_LIMITS=  [-15*pi, 15*pi] # rad / s [+-15pi in tutorial]
    start_angle = 0 # rad - Starting angle of the pendulum
    start_rate = 0 # rad/s - Starting rate of the pendulum
    start_pos = 0 # m - Starting position of the cart
    start_vel = 0 # m/s - Starting velocity of the cart
    NUM_ANGLE_INTERVALS = 40 # Number of discrete states in the angular range [20 in tutorial]
    NUM_RATE_INTERVALS = 40 # Number of discrete states in the angular velocity [20 in tutorial]
    MASS_BOB = 2.0 # kilograms, kg - Mass of the bob at the end of the pendulum (assume zero arm mass) [2 per DPF]
    MASS_CART = 8.0 # kilograms, kg - Mass of cart [8 per DPF]
    LENGTH = 0.5 # meters, m - Length of the pendulum, meters [0.5 in DPF]
    ACCEL_G = 9.81 # m/s^2 - gravitational constant
    ROT_INERTIA = 0 # kg * m^2 - rotational inertia of the pendulum, computed in __init__
    
    dt = 0.10 # Time between steps [0.1 in DPF]
    
    force_noise_max = 10 # Newtons, N - Maximum noise possible, uniformly distributed [10 in tutorial]
    
    EPISODE_CAP = 300 # [200 in tutorial, 300 in DPF]
    
    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
    s_continuous = None # Current actual, continuous state.  NOTE that this contains values for x, xdot as well
    cur_force_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    
    # Plotting variables
    pendulumArm = None
    pendulumBob = None
    actionArrowBottom = None
    ACTION_ARROW_LENGTH = 0.4
    action_arrow_x_left = 0
    action_arrow_x_right = 0
    domainFig = None
    subplotAxes = None
    CIRCLE_RADIUS = 0.1
    PENDULUM_PIVOT_Y = 0 # Y position of pendulum pivot
    
    # are constrained by the format expected by ode functions.
    def __init__(self, start_angle = 0, start_rate = 0, dt = 0.10, force_noise_max = 10, visualize = visualize, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        self.states_num = self.NUM_ANGLE_INTERVALS * self.NUM_RATE_INTERVALS       # Number of states
        self.actions_num = len(self.AVAIL_FORCE)      # Number of Actions
        self.state_space_dims = 2 # Number of dimensions of the state space
        self.episodeCap = self.EPISODE_CAP       # The cap used to bound each episode (return to s0 after)
        self.start_angle = start_angle
        self.start_rate = start_rate
        self.dt = dt
        self.length = self.LENGTH
        self.rot_inertia = self.MASS_BOB * self.length ** 2
        self.force_noise_max = force_noise_max
        self.circle_radius = self.CIRCLE_RADIUS
        self.visualize = visualize
        if self.visualize: self.s_continuous = self.s0WithCartPosition()
        else: self.s_continuous = self.s0()
#        super(InvertedPendulum,self).__init__(logger)
        
    def resetLocalVariables(self):
        if self.visualize: self.s_continuous = array([self.start_angle, self.start_rate, self.start_pos, self.start_vel])
        else: self.s_continuous = array([self.start_angle, self.start_rate])
        
    def showDomain(self,s,a = 0):
        if self.visualize == False:
            pass
        # Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        # Pendulum rotation is centered at origin
        # Note that we show the actual, continuous state of the pendulum as recorded
        # internally by the domain, s_continuous - THE DISCRETIZED STATE PASSED IN, 's', IS IGNORED
        # To use discretized state instead, merely change the assignment of 'theta' below
        if self.domainFig is None:
            self.domainFig = pl.figure()
            pl.show()
        pl.clf()
        self.subplotAxes = self.domainFig.gca()
        forceAction = self.AVAIL_FORCE[a]
        curX = self.s_continuous[StateIndex.X]
        curXDot = self.s_continuous[StateIndex.X_DOT]
        
        # Allow room for pendulum to swing without getting cut off on graph
        viewableDistance = self.length + self.circle_radius + 0.5
        self.subplotAxes.set_xlim(curX -viewableDistance, curX + viewableDistance)
        self.subplotAxes.set_ylim(-viewableDistance, viewableDistance)
    #           self.subplotAxes.set_aspect('equal')
#           pl.axis('off')
        theta = self.s_continuous[StateIndex.THETA] # Using continuous state
        # theta = s[StateIndex.THETA] # using discretized state
        # recall we define 0 down, 90 deg right
        pendulumBobX = curX + self.length * sin(theta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.length * cos(theta)
        if self.DEBUG: print 'Pendulum Position: ',pendulumBobX,pendulumBobY
        # update pendulum arm on figure
        self.pendulumArm = lines.Line2D([curX, pendulumBobX],[self.PENDULUM_PIVOT_Y, pendulumBobY], linewidth = 3, color='black')
        if forceAction == 0: pass # no torque
        else: # cw or ccw torque
            self.action_arrow_x_left = curX - self.ACTION_ARROW_LENGTH/2
            self.action_arrow_x_right = self.PENDULUM_PIVOT_Y + self.ACTION_ARROW_LENGTH/2
            if forceAction > 0: # counterclockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_left, -self.length - 0.3, self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='red')
            else:# clockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_right, -self.length - 0.3, -self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='black')
            self.subplotAxes.add_patch(self.actionArrowBottom)
        self.pendulumBob = mpatches.Circle((pendulumBobX,pendulumBobY), radius = self.circle_radius, color = 'blue')
#            self.actionArrow = mpatches.FancyArrowPatch(posA=[arrowX,0],posB=[-arrowX,0],connectionstyle='arc3',arrowstyle='simple')
#            self.subplotAxes.add_patch(self.actionArrow)
        self.subplotAxes.add_patch(self.pendulumBob)
        self.subplotAxes.add_line(self.pendulumArm)
        pl.draw()
        sleep(self.dt)
        
    def showLearning(self,representation):
        pass

    def s0(self):   
        # ASSUME that any time s0 is called, we should reset the internal values of s_continuous    
        # Returns the initial state, [theta0, thetaDot0]
        self.resetLocalVariables()
        return array([self.start_angle, self.start_rate])
    
    def s0WithCartPosition(self):
        # Returns the initial state (4 states) [theta0, thetaDot0, x0, xDot0]
        self.s_continuous = array([self.start_angle, self.start_rate, self.start_pos, self.start_vel])
        return array([self.start_angle, self.start_rate, self.start_pos, self.start_vel])

    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(len(self.AVAIL_FORCE))

    def step(self,s,a):
        # Simulate one step of the pendulum after taking torque action a
        # Note that the discretized state passed in is IGNORED - only the state internal
        # to this domain, s_continuous, is used.
        # Note also that we store the current action and noise in member variables so they can
        # be accessed by the function _dsdt below, which has a pre-defined header
        # expected by integrate.odeint (so we cannot pass further parameters.)
        # Adapted from pybrain code for cartpole
        self.cur_action = self.AVAIL_FORCE[a] # store action so it can be retrieved by other methods;
        if self.force_noise_max > 0:  # Store random noise so it can be retrieved by other methods
            self.cur_force_noise = random.uniform(-self.force_noise_max, self.force_noise_max)
        else: self.cur_force_noise = 0
        # Unfortunate that scipy ode methods do not allow additional parameters to be passed.
        ns = s[:]
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        #self.s_continuous = ns_continuous[-1] # We only care about the state at the ''final timestep'', self.dt
        
        if self.visualize: ns_continuous = rk4(self._dsdtVis, self.s_continuous, [0, self.dt])
        else: ns_continuous = rk4(self._dsdt, self.s_continuous, [0, self.dt])
        self.s_continuous = ns_continuous[-1]
        
        # ns_continuous = integrate.quad(self._dsdt, 0, self.dt, args=(self.s_continuous))
         # wrap angle between 0 and 2pi
        self.s_continuous[StateIndex.THETA] = wrap(self.s_continuous[StateIndex.THETA],self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        self.s_continuous[StateIndex.THETA_DOT] = bound(self.s_continuous[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        theta = self.s_continuous[StateIndex.THETA]
        thetaDot = self.s_continuous[StateIndex.THETA_DOT]
        # collapse angles to their respective discretizations
        ns[StateIndex.THETA] = closestDiscretization(theta, self.NUM_ANGLE_INTERVALS, self.ANGLE_LIMITS)
        ns[StateIndex.THETA_DOT] = closestDiscretization(thetaDot, self.NUM_RATE_INTERVALS, self.ANGULAR_RATE_LIMITS)
        return self._earnedReward(ns), ns, self.NOT_TERMINATED
    
    # From pybrain environment 'cartpole'
    # Used by odeint to numerically integrate the differential equation
    def _dsdt(self, s_continuous, t):
    # def _dsdt(self,t, s_continuous):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # of the state, which can then 
        force = self.cur_action + self.cur_force_noise
        g = self.ACCEL_G
        l = self.length
        m_p = self.MASS_BOB
        alpha = 1.0 / (self.MASS_CART + m_p)
        theta = s_continuous[StateIndex.THETA]
        thetaDot = s_continuous[StateIndex.THETA_DOT]
        
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        thetaDotSq = thetaDot ** 2
        
        numer1 = g * sinTheta - alpha * m_p * l * thetaDotSq * sin(2 * theta) / 2.0
        numer2 = -alpha * cosTheta * force
        denom = 4.0 * l / 3.0  -  alpha * m_p * l * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = (numer1 + numer2) / denom
        return (thetaDot, thetaDotDot)
        # reverse signs from http://www.ece.ucsb.edu/courses/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
        # because of different conventions in Wang
#        xDotDot = -(l**2) / cos(theta) * thetaDotDot + g*sin(theta)
#        if self.visualize:
#            xDot = s_continuous[StateIndex.X_DOT]
#            xDotDot = alpha * force - m_p*l*alpha*sinTheta*thetaDotSq + m_p*l*alpha*cosTheta*thetaDot
 #           return (thetaDot, thetaDotDot, xDot, xDotDot)
#        else:

    # From pybrain environment 'cartpole'
    # Used by odeint to numerically integrate the differential equation
    def _dsdtVis(self, s_continuous, t):
    # def _dsdt(self,t, s_continuous):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # of the state, which can then 
        force = self.cur_action + self.cur_force_noise
        g = self.ACCEL_G
        l = self.length
        m_p = self.MASS_BOB
        alpha = 1.0 / (self.MASS_CART + m_p)
        theta = s_continuous[StateIndex.THETA]
        thetaDot = s_continuous[StateIndex.THETA_DOT]
        
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        thetaDotSq = thetaDot ** 2
        
        numer1 = g * sinTheta - alpha * m_p * l * thetaDotSq * sin(2 * theta) / 2.0
        numer2 = -alpha * cosTheta * force
        denom = 4.0 * l / 3.0  -  alpha * m_p * l * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = (numer1 + numer2) / denom
        # reverse signs from http://www.ece.ucsb.edu/courses/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
        # because of different conventions in Wang
#        xDotDot = -(l**2) / cos(theta) * thetaDotDot + g*sin(theta)
        xDot = s_continuous[StateIndex.X_DOT]
        xDotDot = alpha * force - m_p*l*alpha*sinTheta*thetaDotSq + m_p*l*alpha*cosTheta*thetaDot
        return (thetaDot, thetaDotDot, xDot, xDotDot)
        
    def _earnedReward(self, s):
        if(self.GOAL_REGION[0] < s[StateIndex.THETA] < self.GOAL_REGION[1]):
            return self.GOAL_REWARD
        else: return 0
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        return self.NOT_TERMINATED # Pendulum has no absorbing state
if __name__ == '__main__':
    random.seed(0)
    p = CartPole(start_angle = 0, start_rate = 0, dt = 0.10, force_noise_max = 10, visualize = True);
    p.test(1000)
