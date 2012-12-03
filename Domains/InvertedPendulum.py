import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

from scipy import integrate # for integration of state
from matplotlib import lines

#######################################################
# Robert Klein, Alborz Geramifard at MIT, Nov. 30 2012#
#######################################################
# State is angular position of pendulum on [0, 2*pi),
# (relative to straight down at 0 rad),and
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

class StateIndex:
    THETA, THETA_DOT = 0,1


class InvertedPendulum(Domain):
    AVAIL_TORQUES = [-10,0,10] # Newtons, N - Torque values available as actions
#    AVAIL_TORQUES = [-1,0,1] # Newtons, N - Torque values available as actions
    
    GOAL_REGION = [pi - pi/6, pi + pi/6] # radians, rad - Goal region,
    GOAL_REWARD = 1 # Reward obtained for remaining in the goal region
    # Implicit 0 reward everywhere else - it's all relative.
    ANGLE_LIMITS = [0, 2*pi] # rad - Limits on pendulum angle (NOTE we wrap the angle at 2*pi)
    ANGULAR_RATE_LIMITS=  [-15*pi, 15*pi] # rad / s
    start_angle = 0 # rad - Starting angle of the pendulum
    start_rate = 0 # rad/s - Starting rate of the pendulum
    NUM_ANGLE_INTERVALS = 20 # Number of discrete states in the angular range [20 in tutorial]
    NUM_RATE_INTERVALS = 20 # Number of discrete states in the angular velocity [20 in tutorial]
    MASS_BOB = 1 # kilograms, kg - Mass of the bob at the end of the pendulum (assume zero arm mass)
    LENGTH = 1.0 # meters, m - Length of the pendulum, meters
    ACCEL_G = 9.81 # m/s^2 - gravitational constant
    ROT_INERTIA = 0 # kg * m^2 - rotational inertia of the pendulum
    
    dt = 0 # Time between steps
    
    torque_noise_var = 0.1 # Noise std deviation (in N*m) of applied pendulum torque [0.1 in tutorial]
    
    EPISODE_CAP = 200 # [200 in tutorial]
    
    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
    s_continuous = None # Current actual, continuous state
    cur_torque_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    # Plotting variables
    pendulumArm = None
    pendulumBob = None
    actionArrowTop = None
    actionArrowBottom = None
    ACTION_ARROW_LENGTH = 0.4
    action_arrow_x_left = 0
    action_arrow_x_right = 0
    domainFig = None
    subplotAxes = None
    CIRCLE_RADIUS = 0.1
    PENDULUM_PIVOT = [0,0] # pivot point of pendulum
    
    # are constrained by the format expected by ode functions.
    def __init__(self, logger, start_angle = 0, start_rate = 0, dt = 0.1, torque_noise_var = 0.1):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.state_space_limits = [self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS]
        self.states_num = self.NUM_ANGLE_INTERVALS * self.NUM_RATE_INTERVALS       # Number of states
        self.actions_num = len(self.AVAIL_TORQUES)      # Number of Actions
        state_space_dims = 2 # Number of dimensions of the state space
        self.episodeCap = self.EPISODE_CAP       # The cap used to bound each episode (return to s0 after)
        self.start_angle = start_angle
        self.start_rate = start_rate
        self.dt = dt
        self.length = self.LENGTH
        self.rot_inertia = self.MASS_BOB * self.length ** 2
        self.torque_noise_var = torque_noise_var
        self.circle_radius = self.CIRCLE_RADIUS
        self.action_arrow_x_left = self.PENDULUM_PIVOT[0] - self.ACTION_ARROW_LENGTH/2
        self.action_arrow_x_right = self.PENDULUM_PIVOT[0] + self.ACTION_ARROW_LENGTH/2
        
        self.s_continuous = self.s0()
#        super(InvertedPendulum,self).__init__(logger)
        
    def showDomain(self,s,a = 0):
        # Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        # Pendulum rotation is centered at origin
        # Note that we show the actual, continuous state of the pendulum as recorded
        # internally by the domain, s_continuous - THE DISCRETIZED STATE PASSED IN, 's', IS IGNORED
        # To use discretized state instead, merely change the assignment of 'theta' below
        if self.domainFig == None: # Need to initialize the figure
            self.domainFig = pl.figure()
            self.subplotAxes = self.domainFig.gca()
            self.pendulumArm = lines.Line2D([],[], linewidth = 3, color='black')
            self.pendulumBob = mpatches.Circle((0,0), radius = self.circle_radius)
            
            self.subplotAxes.add_patch(self.pendulumBob)
            self.subplotAxes.add_line(self.pendulumArm)
            # Allow room for pendulum to swing without getting cut off on graph
            viewableDistance = self.length + self.circle_radius + 0.5
            self.subplotAxes.set_xlim(-viewableDistance, viewableDistance)
            self.subplotAxes.set_ylim(-viewableDistance, viewableDistance)
 #           self.subplotAxes.set_aspect('equal')
            pl.axis('off')
            pl.show(block=False)
        theta = self.s_continuous[StateIndex.THETA] # Using continuous state
        # theta = s[StateIndex.THETA] # using discretized state
        # recall we define 0 down, 90 deg right
        pendulumBobX = self.PENDULUM_PIVOT[0] + self.length * sin(theta)
        pendulumBobY = self.PENDULUM_PIVOT[0] - self.length * cos(theta)
#DEBUG        print pendulumBobX,pendulumBobY
        # update pendulum arm on figure
        self.pendulumArm.set_data([self.PENDULUM_PIVOT[0], pendulumBobX],[self.PENDULUM_PIVOT[1], pendulumBobY])
        if self.pendulumBob is not None:
            self.pendulumBob.remove()
            self.pendulumBob = None
        if self.actionArrowBottom is not None:
            self.actionArrowBottom.remove()
            self.actionArrowBottom = None
        if self.actionArrowTop is not None:
            self.actionArrowTop.remove()
            self.actionArrowTop = None
        #if self.actionArrow is not None: self.actionArrow.remove()
        if a == 0: pass # no torque
        else: # cw or ccw torque
            if a > 0: # counterclockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_left, -self.length - 0.3, self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='red')
                self.actionArrowTop = pl.Arrow(self.action_arrow_x_right, self.length + 0.3, -self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='red')
            else:# clockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_right, -self.length - 0.3, -self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='black')
                self.actionArrowTop = pl.Arrow(self.action_arrow_x_left, self.length + 0.3, self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='black')
            self.subplotAxes.add_patch(self.actionArrowBottom)
            self.subplotAxes.add_patch(self.actionArrowTop)
        self.pendulumBob = mpatches.Circle((pendulumBobX,pendulumBobY), radius = self.circle_radius, color = 'blue')
#            self.actionArrow = mpatches.FancyArrowPatch(posA=[arrowX,0],posB=[-arrowX,0],connectionstyle='arc3',arrowstyle='simple')
#            self.subplotAxes.add_patch(self.actionArrow)
        self.subplotAxes.add_patch(self.pendulumBob)
        pl.draw()
        sleep(self.dt)
        
    def showLearning(self,representation):
        pass
    def s0(self):       
        # Returns the initial state
        return [self.start_angle, self.start_rate]
    def possibleActions(self,s):
        return self.AVAIL_TORQUES
    def step(self,s,a):
        # Simulate one step of the pendulum after taking torque action a
        # Note that the discretized state passed in is IGNORED - only the state internal
        # to this domain, s_continuous, is used.
        # Note also that we store the current action and noise in member variables so they can
        # be accessed by the function _dsdt below, which has a pre-defined header
        # expected by integrate.odeint (so we cannot pass further parameters.)
        # Adapted from pybrain code for cartpole
        self.cur_action = a # store action so it can be retrieved by other methods;
        if self.torque_noise_var > 0:  # Store random noise so it can be retrieved by other methods
            self.cur_torque_noise = random.normal(scale = sqrt(self.torque_noise_var))
        else: self.cur_torque_noise = 0
        # Unfortunate that scipy ode methods do not allow additional parameters to be passed.
        ns = s[:]
        ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        self.s_continuous = ns_continuous[-1] # We only care about the state at the ''final timestep'', self.dt
#        print 'After -1',ns
         # wrap angle between 0 and 2pi
        theta = wrap(self.s_continuous[StateIndex.THETA],self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        thetaDot = self.s_continuous[StateIndex.THETA_DOT]
        # collapse angles to their respective discretizations
        ns[StateIndex.THETA] = closestDiscretization(theta, self.NUM_ANGLE_INTERVALS, self.ANGLE_LIMITS)
        ns[StateIndex.THETA_DOT] = closestDiscretization(thetaDot, self.NUM_RATE_INTERVALS, self.ANGULAR_RATE_LIMITS)
        return self._earnedReward(ns), ns, self.NOT_TERMINATED
    # From pybrain environment 'cartpole' 
    def _dsdt(self, s_continuous, t):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # of the state, which can then 
        torque = self.cur_action + self.cur_torque_noise
        g = self.ACCEL_G
        l = self.length
        theta = s_continuous[StateIndex.THETA]
        # -g/l sin(theta) + tau / I   [[mgl sin(theta) + tau) / (m * l^2) ]]
        thetaDotDot = -(g / l) * sin(theta) + (torque / self.rot_inertia)
#        print 'g,l,theta',g,l,theta
#        print 'thetaDot,thetaDotDot,torque',s[StateIndex.THETA_DOT],thetaDotDot,torque
        return (s_continuous[StateIndex.THETA_DOT], thetaDotDot)
    def _earnedReward(self, s):
        if(self.GOAL_REGION[0] < s[StateIndex.THETA] < self.GOAL_REGION[1]):
            return self.GOAL_REWARD
        else: return 0
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        return self.NOT_TERMINATED # Pendulum has no absorbing state
if __name__ == '__main__':
    random.seed(0)
    p = InvertedPendulum(logger = None, start_angle = 0, start_rate = 0, dt = 0.25, torque_noise_var = 1.0);
    p.test(500)
