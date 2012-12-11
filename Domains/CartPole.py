import sys, os
from matplotlib.patches import ConnectionStyle
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

#from scipy import integrate # for integration of state
from matplotlib.mlab import rk4
from matplotlib import lines

#######################################################
# Robert Klein, Alborz Geramifard at MIT, Dec. 6 2012#
#######################################################
# Objective is to hold the pendulum vertical, rather than
# having to swing it up to balance.
# Essentially a sub-problem of the InvertedPendulum Domain,
# except that the pendulum is never allowed to fall below
# the horizontal (or some specified angle).
# Additionally, limits in the motion of the cart are imposed.
# (Those these can be set to float("-inf") and float("inf") ).
# Note though, that Lagoudakis and Parr did not actually impose
# any such constraints.
#
# Per Lagoudakis and Parr LSPI 2003, derived from Wang 1996.
# (See mkpendulumonelink.m of "1Link" implementation by Parr)
# (See CartPole implementation in the RL Community,
# http://library.rl-community.org/wiki/CartPole)
# Used those domain implementations to define limits not 
# specifically mentioned in the papers.
#
# State is angular position of pendulum on [-pi/2, pi/2),
# (relative to straight up at 0 rad),and positive cw.
# Positive force acts to the right on the cart
# Angular rate of pendulum, [-2, 2] rad/s.
# Actions take the form of force applied to cart
# [-50, 0, 50] N force are the default available actions.
#
# The objective is to keep the pendulum in the goal
# region for as long as possible; default goal
# region is [-pi/6,+ pi/6], no reward is
# obtained anywhere else.
# Default has the pendulum upright, episode ends when
# pendulum falls below horizontal
# Optional noise is added in the form of force on cart.
######################################################

# Size of objects in visualization is computed automatically
# based on assigned mass.

# In comments below, "DPF" refers to Kemal's Java implementation

class StateIndex:
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3


class CartPole(Domain):
    DEBUG = 0 # Set to non-zero to enable print statements
    
    AVAIL_FORCE = array([-50,0,50]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    
    GOAL_REGION = [-pi/6, pi/6] # radians, rad - Goal region [5pi/6, 7pi/6 per tutorial]
    GOAL_REWARD = 1 # Reward obtained for remaining in the goal region [1 per tutorial]
    TERMINAL_REWARD = -1 # Reward (penalty) for allowing the pendulum to drop out of the angle_limits
    
    ANGLE_LIMITS = [-pi/2, pi/2] # rad - Limits on pendulum angle
    ANGULAR_RATE_LIMITS=  [-2, 2] # rad / s - Limits on pendulum rate
#    POSITON_LIMITS = [float("-inf"), float("inf")]
#    VELOCITY_LIMITS = [float("-inf"), float("inf")] 
    POSITON_LIMITS = [-2.0, 2.0] # m - Limits on cart position (unspecified) (Approx. -2,2 gives good results)
    VELOCITY_LIMITS = [-5.0, 5.0] # m/s - Limits on cart velocity
    start_angle = 0 # rad - Starting angle of the pendulum
    start_rate = 0 # rad/s - Starting rate of the pendulum
    start_pos = 0 # m - Starting position of the cart
    start_vel = 0 # m/s - Starting velocity of the cart
    HORIZONTAL_ANGLE = pi/2 # Horizontal angle, below absolute value of which terminate episode
    # Note that we implicitly terminate the episode of cart position x exceeds POSITION_LIMITS above
    MASS_PEND = 2.0 # kilograms, kg - Mass of the pendulum [2 per DPF]
    MASS_CART = 8.0 # kilograms, kg - Mass of cart [8 per DPF]
    LENGTH = 1.0 # meters, m - Length to the pendulum center of mass, meters [0.5 in DPF]
    ACCEL_G = 9.8 # m/s^2 - gravitational constant
    ROT_INERTIA = 0 # kg * m^2 - rotational inertia of the pendulum, computed in __init__
    
    dt = 0.10 # Time between steps [0.1 in DPF]
    
    force_noise_max = 10 # Newtons, N - Maximum noise possible, uniformly distributed [10 in tutorial]
    
    EPISODE_CAP = 300 # [200 in tutorial, 300 in DPF, 50 in Parr's Matlab]
    
    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
    cur_force_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    
    # Plotting variables
    pendulumArm = None
    cartBox = None
    actionArrowBottom = None
    ACTION_ARROW_LENGTH = 0.4
    action_arrow_x_left = 0
    action_arrow_x_right = 0
    domainFig = None
    domainFigAxes = None
    circle_radius = 0.05
    PENDULUM_PIVOT_Y = 0 # Y position of pendulum pivot
    RECT_WIDTH = 0.5 
    BLOB_WIDTH = 0.2
    RECT_HEIGHT = MASS_CART/20.0
    PEND_WIDTH = 0 # If this value is left as zero, it is computed automatically based on mass.
    GROUND_WIDTH = 2
    GROUND_HEIGHT = 1
    CARTBLOB_R = .3/4.0 # The radius of the blob on the cart
    # vertecies for the ground:
    GROUND_VERTS = array([
                          (POSITON_LIMITS[0],-RECT_HEIGHT/2.0),
                          (POSITON_LIMITS[0],RECT_HEIGHT/2.0),
                          (POSITON_LIMITS[0]-GROUND_WIDTH, RECT_HEIGHT/2.0),
                          (POSITON_LIMITS[0]-GROUND_WIDTH, RECT_HEIGHT/2.0-GROUND_HEIGHT),
                          (POSITON_LIMITS[1]+GROUND_WIDTH, RECT_HEIGHT/2.0-GROUND_HEIGHT),
                          (POSITON_LIMITS[1]+GROUND_WIDTH, RECT_HEIGHT/2.0),
                          (POSITON_LIMITS[1], RECT_HEIGHT/2.0),
                          (POSITON_LIMITS[1], -RECT_HEIGHT/2.0),
                          ])
    # are constrained by the format expected by ode functions.
    def __init__(self, start_angle = 0, start_rate = 0, dt = 0.10, force_noise_max = 10, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
#        self.states_num = inf       # Number of states
        self.actions_num = len(self.AVAIL_FORCE)      # Number of Actions
        self.state_space_dims = 4 # Number of dimensions of the state space
        self.continuous_dims = [StateIndex.THETA, StateIndex.THETA_DOT, StateIndex.X, StateIndex.X_DOT]
        self.episodeCap = self.EPISODE_CAP       # The cap used to bound each episode (return to s0 after)
        self.start_angle = start_angle
        self.start_rate = start_rate
#        self.start_pos = 0 # Optionally add these to init parameter list
#        self.start_vel = 0# Optionally add these to init parameter list
        self.dt = dt
        self.length = self.LENGTH
        self.moment_arm = self.length / 2.0
        self.rot_inertia = self.MASS_PEND * self.moment_arm ** 2
        self.force_noise_max = force_noise_max
        
        self.alpha = 1.0 / (self.MASS_CART + self.MASS_PEND)
        
        if self.RECT_HEIGHT == 0: # No rectangle height specified
            self.RECT_HEIGHT = self.MASS_CART / 20 # Arbitrary number, reasonable visualization
        if self.PEND_WIDTH == 0: # No rectangle height specified
            self.PEND_WIDTH = int(self.MASS_PEND / self.length)+1 # Arbitrary number, reasonable visualization
        
        if self.logger: 
            self.logger.log("length:\t\t%0.2f(m)" % self.length)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)
        super(CartPole,self).__init__(logger)
        
    def showDomain(self,s,a = 0):
        # Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        # Pendulum rotation is centered at origin
        
        if self.domainFig == None: # Need to initialize the figure
            self.domainFig = pl.figure()
            ax = self.domainFig.add_axes([0, 0, 1, 1], frameon=True, aspect=1.)
            self.pendulumArm = lines.Line2D([],[], linewidth = self.PEND_WIDTH, color='black')
            self.cartBox    = mpatches.Rectangle([0, self.PENDULUM_PIVOT_Y - self.RECT_HEIGHT/2.0], self.RECT_WIDTH, self.RECT_HEIGHT,alpha=.4)
            self.cartBlob   = mpatches.Rectangle([0, self.PENDULUM_PIVOT_Y - self.BLOB_WIDTH/2.0], self.BLOB_WIDTH, self.BLOB_WIDTH,alpha=.4) 
            ax.add_patch(self.cartBox)
            ax.add_line(self.pendulumArm)
            ax.add_patch(self.cartBlob)
            #Draw Ground
            path    = mpath.Path(self.GROUND_VERTS)
            patch   = mpatches.PathPatch(path,hatch="//")
            ax.add_patch(patch)

            # Allow room for pendulum to swing without getting cut off on graph
            viewableDistance = self.length + self.circle_radius + 0.5
            ax.set_xlim(self.POSITON_LIMITS[0] - viewableDistance, self.POSITON_LIMITS[1] + viewableDistance)
            ax.set_ylim(-viewableDistance, viewableDistance)
            #ax.set_aspect('equal')
            
            pl.show()
            
        forceAction = self.AVAIL_FORCE[a]
        curX = s[StateIndex.X]
        curXDot = s[StateIndex.X_DOT]
        curTheta = s[StateIndex.THETA]
        
        pendulumBobX = curX + self.length  * sin(curTheta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.length * cos(curTheta)

        if self.DEBUG: print 'Pendulum Position: ',pendulumBobX,pendulumBobY
        
        # update pendulum arm on figure
        self.pendulumArm.set_data([curX, pendulumBobX],[self.PENDULUM_PIVOT_Y, pendulumBobY])
        self.cartBox.set_x(curX - self.RECT_WIDTH/2.0)
        self.cartBlob.set_x(curX - self.BLOB_WIDTH/2.0)
        
        
        if self.actionArrowBottom is not None:
            self.actionArrowBottom.remove()
            self.actionArrowBottom = None
            
        if forceAction == 0: pass # no force
        else: # cw or ccw torque
            if forceAction > 0: # rightward force
                self.actionArrowBottom = fromAtoB(
                                                  curX - self.ACTION_ARROW_LENGTH - self.RECT_WIDTH/2.0, 0, 
                                                  curX - self.RECT_WIDTH/2.0,  0, 
                                                  'k',"arc3,rad=0",
                                                  0,0, 'simple'
                                                  )
            else:# leftward force
                self.actionArrowBottom = fromAtoB(
                                                  curX + self.ACTION_ARROW_LENGTH + self.RECT_WIDTH/2.0, 0,
                                                  curX + self.RECT_WIDTH/2.0, 0, 
                                                  'r',"arc3,rad=0",
                                                  0,0,'simple'
                                                  )
            
        pl.draw()
    def showLearning(self,representation):
        pass
    
    def s0(self):   
        # ASSUME that any time s0 is called, we should reset the internal values of s_continuous    
        # Returns the initial state, [theta0, thetaDot0]
        return array([self.start_angle, self.start_rate, self.start_pos, self.start_vel])
    
    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(len(self.AVAIL_FORCE))
    def step(self,s,a):
        # Simulate one step of the pendulum after taking force action a
        # Note that we store the current action and noise in member variables so they can
        # be accessed by the function _dsdt below, which has a pre-defined header
        # expected by integrate.odeint (so we cannot pass further parameters.)
        
        self.cur_action = self.AVAIL_FORCE[a] # store action so it can be retrieved by other methods;
        if self.force_noise_max > 0:  # Store random noise so it can be retrieved by other methods
            self.cur_force_noise = random.uniform(-self.force_noise_max, self.force_noise_max)
        else: self.cur_force_noise = 0
        # Unfortunate that scipy ode methods do not allow additional parameters to be passed.
        
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        #self.s_continuous = ns_continuous[-1] # We only care about the state at the ''final timestep'', self.dt
        
        ns = rk4(self._dsdt, s, [0, self.dt])
        ns = ns[-1] # only care about final timestep of integration returned by integrator


        ns[StateIndex.THETA_DOT] = bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        ns[StateIndex.X_DOT] = bound(ns[StateIndex.X_DOT], self.VELOCITY_LIMITS[0], self.VELOCITY_LIMITS[1])
        return self._earnedReward(ns), ns, self.isTerminal(ns)
    # From CartPole implementation described at top, from rlcommunity.org
    # Used by odeint to numerically integrate the differential equation
    def _dsdt(self, s_continuous, t):
    # def _dsdt(self,t, s_continuous):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # at a given state
        force = self.cur_action + self.cur_force_noise
        g = self.ACCEL_G
        l = self.moment_arm
        m_pendAlphaTimesL = self.MASS_PEND * self.alpha * l
        theta = s_continuous[StateIndex.THETA]
        thetaDot = s_continuous[StateIndex.THETA_DOT]
        xDot = s_continuous[StateIndex.X_DOT]
        
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        thetaDotSq = thetaDot ** 2
        
        term1 = force*self.alpha + m_pendAlphaTimesL * thetaDotSq * sinTheta
        numer = g * sinTheta - cosTheta * term1
        denom = 4.0 * l / 3.0  -  m_pendAlphaTimesL * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = numer / denom
        
        xDotDot = term1 - m_pendAlphaTimesL * thetaDotDot * cosTheta
        return (thetaDot, thetaDotDot, xDot, xDotDot)
    
    def _earnedReward(self, s):
        if(self.GOAL_REGION[0] < s[StateIndex.THETA] < self.GOAL_REGION[1]):
            return self.GOAL_REWARD
        else: return 0
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        return not(self.POSITON_LIMITS[0] < s[StateIndex.X] < self.POSITON_LIMITS[1]) or \
                abs(s[StateIndex.THETA]) >= self.HORIZONTAL_ANGLE
                
if __name__ == '__main__':
    random.seed(0)
    p = CartPole(start_angle = .01, start_rate = 0, dt = 0.10, force_noise_max = 10);
    p.test(1000)
