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
# Perhaps more aptly described as "Invert-The-Pendulum"

# State is [theta, thetaDot]:
# theta = angular position of pendulum on [-pi, pi) rad
# (relative to straight up at 0 rad),and positive cw.
# thetaDot = Angular rate of pendulum, [-15*pi, 15*pi] rad/s.
# Positive force acts to the right on the cart
#
# Actions take the form of force applied to cart;
# [-50, 0, 50] N force are the default available actions.
#
# The objective is to keep the pendulum in the goal
# region for as long as possible, with +1 reward for
# each step in which this condition is met; the expected
# optimum then is to swing the pendulum vertically and
# hold it there, collapsing the problem to CartPole
# but without any restrictions in x.
# Default goal region is [-pi/6,+ pi/6].
# No reward is obtained anywhere else.
# Optional noise is added in the form of force on cart.
# Note that THERE IS NO TERMINAL STATE FOR THIS TASK -
# the pendulum is allowed to swing in its full rotation.
# Typically, the pendulum starts down, and must be
# swung upright and held there for as long as possible.
######################################################

# In comments below, "DPF" refers to Kemal's Java implementation

class StateIndex:
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3
    # X, X_DOT NOT YET COMPUTED - optionally in ds_dt, also return states for xDot and xDotDot,
    # integrate, and track them along with angular states.


class InvertedPendulum(Domain):
    DEBUG = 0 # Set to non-zero to enable print statements
    
    AVAIL_FORCE = array([-50,0,50]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    
    GOAL_REGION = [-pi/6, pi/6] # radians, rad - Goal region [5pi/6, 7pi/6 per tutorial]
    GOAL_REWARD = 1 # Reward obtained for remaining in the goal region [1 per tutorial]
    # Implicit 0 reward everywhere else - it's all relative.
    ANGLE_LIMITS = [-pi, pi] # rad - Limits on pendulum angle (NOTE we wrap the angle at 2*pi)
    ANGULAR_RATE_LIMITS=  [-8*pi, 8*pi] # Limits on pendulum rate [+-15pi in tutorial]
    start_angle = 0 # rad - Starting angle of the pendulum
    start_rate = 0 # rad/s - Starting rate of the pendulum
    MASS_PEND = 2.0 # kilograms, kg - Mass of the bob at the end of the pendulum (assume zero arm mass) [2 per DPF]
    MASS_CART = 8.0 # kilograms, kg - Mass of cart [8 per DPF]
    LENGTH = 1.0 # meters, m - Length of the pendulum, meters [0.5 in DPF, center of mass]
    ACCEL_G = 9.81 # m/s^2 - gravitational constant
    ROT_INERTIA = 0 # kg * m^2 - rotational inertia of the pendulum, computed in __init__
    
    dt = 0.0 # Time between steps [0.1 in DPF]
    
    force_noise_max = 10 # Newtons, N - Maximum noise possible, uniformly distributed [10 in tutorial]
    
    EPISODE_CAP = 100 # [200 in tutorial, 300 in DPF]
    
    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
    cur_force_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    
    # Plotting variables
    pendulumArm = None
    pendulumBob = None
    actionArrowBottom = None
    ACTION_ARROW_LENGTH = 0.4
    action_arrow_x_left = 0
    action_arrow_x_right = 0
    domainFig = None
    domainFigAxes = None
    circle_radius = 0.1
    PENDULUM_PIVOT_X = 0 # X position is also fixed in this visualization
    PENDULUM_PIVOT_Y = 0 # Y position of pendulum pivot
    
    #
    valueFunction_fig = None
    valueFunction_axes = None
    valueFunction_im = None
    valueFunction_cbar = None
    valueFunction_canv = None
    policy_fig = None
    policy_axes = None
    policy_im = None
    policy_canv = None
    MIN_RETURN = 0 # Minimum return possible, used for graphical normalization, computed in init
    MAX_RETURN = 0
    
    # Place colormap here since also require a norm to discretize the colormap
    # MUST ADD MORE COLORS and expand 'boundary' [0,1,2] below when incorporating new actions.
    policy_cmap = col.ListedColormap(['black','white','red'])  # 3 available actions, think 'Red' = 'Rightward' (force), bLack = Leftward
    policy_cmap_norm = col.BoundaryNorm([-0.5,0.5,1.5,2.5], policy_cmap.N) # bounds surrounding the action integers [0,1,2]
    
    
    def __init__(self, start_angle = 0, start_rate = 0, dt = 0.10, force_noise_max = 10, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
#        self.states_num = inf       # Number of states# Number of states
        self.actions_num = len(self.AVAIL_FORCE)      # Number of Actions
        self.episodeCap = self.EPISODE_CAP       # The cap used to bound each episode (return to s0 after)
        self.start_angle = start_angle
        self.start_rate = start_rate
        self.dt = dt
        self.length = self.LENGTH
        self.moment_arm = self.length / 2.0
        self.rot_inertia = self.MASS_PEND * self.moment_arm ** 2
        self.force_noise_max = force_noise_max
        self.alpha = 1.0 / (self.MASS_CART + self.MASS_PEND)
        self.continuous_dims = [StateIndex.THETA, StateIndex.THETA_DOT, StateIndex.X, StateIndex.X_DOT]
        
        if self.logger: 
            self.logger.log("length:\t\t%0.2f(m)" % self.length)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)

        if not ((2*pi / dt > self.ANGULAR_RATE_LIMITS[1]) and (2*pi / dt > -self.ANGULAR_RATE_LIMITS[0])):
            print '''
            WARNING:
            # This has not been observed in practice, but conceivably
            # if the bound on angular velocity is large compared with
            # the time discretization, seemingly 'optimal' performance
            # might result from a stroboscopic-like effect.
            # For example, if dt = 1.0 sec, and the angular rate limits
            # exceed -2pi or 2pi respectively, then it is possible that
            # between consecutive timesteps, the pendulum will have
            # the same position, even though it really completed a
            # rotation, and thus we will find a solution that commands
            # the pendulum to spin with angular rate in multiples of
            # 2pi / dt.
            '''
            print 'Your selection, dt=',self.dt,'and limits',self.ANGULAR_RATE_LIMITS,'Are at risk.'
            print 'Reduce your timestep dt (to increase # timesteps) or reduce angular rate limits so that 2pi / dt > max(AngularRateLimit)'
            print 'Currently, 2pi / dt = ',2*pi/self.dt,', angular rate limits shown above.'
        #plotting - these are constants
        self.action_arrow_x_left = self.PENDULUM_PIVOT_X - self.ACTION_ARROW_LENGTH/2
        self.action_arrow_x_right = self.PENDULUM_PIVOT_Y + self.ACTION_ARROW_LENGTH/2
        
        self.MAX_RETURN = self.GOAL_REWARD * self.episodeCap / 4 # Divide by 2 to emphasize other features of the graph
        self.MIN_RETURN = 0
        
        super(InvertedPendulum,self).__init__(logger)       

        
    def showDomain(self,s,a = 0):
        # Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        # Pendulum rotation is centered at origin
        
        if self.domainFig == None: # Need to initialize the figure
            self.domainFig = pl.figure(2)
            self.domainFigAxes = self.domainFig.gca()
            self.pendulumArm = lines.Line2D([],[], linewidth = 3, color='black')
            self.pendulumBob = mpatches.Circle((0,0), radius = self.circle_radius)
            
            self.domainFigAxes.add_patch(self.pendulumBob)
            self.domainFigAxes.add_line(self.pendulumArm)
            # Allow room for pendulum to swing without getting cut off on graph
            viewableDistance = self.length + self.circle_radius + 0.5
            self.domainFigAxes.set_xlim(-viewableDistance, viewableDistance)
            self.domainFigAxes.set_ylim(-viewableDistance, viewableDistance)
 #           self.domainFigAxes.set_aspect('equal')
            pl.axis('off')
            pl.show()
            
        forceAction = self.AVAIL_FORCE[a]
        theta = s[StateIndex.THETA] # Using continuous state
        
        # recall we define 0deg  up, 90 deg right
        pendulumBobX = self.PENDULUM_PIVOT_X + self.length * sin(theta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.length * cos(theta)
        
        # update pendulum arm on figure
        self.pendulumArm.set_data([self.PENDULUM_PIVOT_X, pendulumBobX],[self.PENDULUM_PIVOT_Y, pendulumBobY])        
  
        if self.DEBUG: print 'Pendulum Position: ',pendulumBobX,pendulumBobY
        if self.pendulumBob is not None:
            self.pendulumBob.remove()
            self.pendulumBob = None
        if self.actionArrowBottom is not None:
            self.actionArrowBottom.remove()
            self.actionArrowBottom = None
        
        if forceAction == 0: pass # no torque
        else: # cw or ccw torque
            if forceAction > 0: # counterclockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_left, -self.length - 0.3, self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='red')
            else:# clockwise torque
                self.actionArrowBottom = pl.Arrow(self.action_arrow_x_right, -self.length - 0.3, -self.ACTION_ARROW_LENGTH, 0.0, width=0.2, color='black')
            self.domainFigAxes.add_patch(self.actionArrowBottom)
            
        self.pendulumBob = mpatches.Circle((pendulumBobX,pendulumBobY), radius = self.circle_radius, color = 'blue')
        self.domainFigAxes.add_patch(self.pendulumBob)
        pl.draw()
        sleep(self.dt)
        
    def showLearning(self,representation):
        numDiscrR = 20 # TODO get this some other way
        numDiscrC = 20 # TODO get this some other way
        angleRange = self.ANGLE_LIMITS[1] - self.ANGLE_LIMITS[0]
        angularRateRange = self.ANGULAR_RATE_LIMITS[1] - self.ANGULAR_RATE_LIMITS[0]
        valueFuncGrid = zeros((numDiscrR, numDiscrC),'uint8')
        policyGrid = zeros((numDiscrR, numDiscrC),'uint8')            
        V            = zeros((numDiscrR,numDiscrC))
        bestA  = zeros((numDiscrR,numDiscrC),dtype= 'uint8') # 0 = suboptimal action, 1 = optimal action
        thetaDot_list = arange(self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1], angularRateRange / numDiscrR)
        theta_list = arange(self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1], angleRange/numDiscrC)
        
        for row, thetaDot in zip(arange(numDiscrR), thetaDot_list):
            for col, theta in zip(arange(numDiscrC), theta_list):
                s        = [theta,thetaDot]
#                if self._earnedReward(s) > 0: V[r,c] = self 
                Qs,As    = representation.Qs(s)
                bestA[row,col]    = representation.bestAction(s)
                V[row,col]   = max(Qs)
#                    print r,c,Qs                  
        #Show Value Function
        
        xTickTuple = array([(col,int(theta_list[col] * 180/pi)) for col in arange(numDiscrC) if col % 4 == 0])
        yTickTuple = array([(row,int(thetaDot_list[row] * 180/pi)) for row in arange(numDiscrR) if row % 2 == 0])
        
        minV = min(min(v) for v in V)
        maxV = max(max(v) for v in V)
        
        if self.valueFunction_fig is None:
            self.valueFunction_fig = pl.figure(3)
#            self.valueFunction_canv = pl.figureCanvas(self, -1, self.valueFunction_fig)
            self.valueFunction_axes = self.valueFunction_fig.gca()
            self.valueFunction_im   = self.valueFunction_axes.imshow(valueFuncGrid, cmap='ValueFunction',interpolation='nearest',vmin=minV,vmax=maxV) 
            self.valueFunction_cbar = self.valueFunction_fig.colorbar(self.valueFunction_im) # Show the colorbar corresponding to the value function
            pl.xticks(xTickTuple[:,0], xTickTuple[:,1], fontsize=12)
            pl.yticks(yTickTuple[:,0], yTickTuple[:,1], fontsize=12)
            pl.xlabel('theta (deg)')
            pl.ylabel('theta (deg/s)')
            pl.title('Value Function')
    #            f.set_size_inches(10,20)
            pl.show()
            #pl.tight_layout()
        
        if self.policy_fig is None:
            self.policy_fig = pl.figure(4)
#            self.policy_canv = FigureCanvas(self, -1, self.policy_fig)
            self.policy_axes = self.policy_fig.gca()
            # Note that we re-use the value function color map below, since it contains
            self.policy_im = pl.imshow(policyGrid, cmap=self.policy_cmap, norm = self.policy_cmap_norm, interpolation='nearest',vmin=0,vmax=self.actions_num)
            pl.xticks(xTickTuple[:,0], xTickTuple[:,1], fontsize=12)
            pl.yticks(yTickTuple[:,0], yTickTuple[:,1], fontsize=12)
            pl.xlabel('theta (deg)')
            pl.ylabel('thetaDot (deg/s)')
            pl.title('Policy (Red = Right, bLack = Left, white = no action)')
#            f.set_size_inches(10,20)
            pl.show()
            #pl.tight_layout()
        
        self.valueFunction_cbar.set_clim(vmin = minV, vmax = maxV)
        self.valueFunction_cbar.draw_all()
        self.valueFunction_im.set_data(V)
        self.policy_im.set_data(bestA)
        self.valueFunction_fig.canvas.draw()
        self.policy_fig.canvas.draw()  

    def s0(self):    
        # Returns the initial state, [theta0, thetaDot0]
        return array([self.start_angle, self.start_rate])
    
    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)

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
        
        ns = rk4(self._dsdt, s, [0, self.dt])
        ns = ns[-1] # only care about final timestep of integration returned by integrator

         # wrap angle between -pi and pi (or whatever values assigned to ANGLE_LIMITS)
        ns[StateIndex.THETA] = wrap(ns[StateIndex.THETA],self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT] = bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        
        return self._earnedReward(ns), ns, self.NOT_TERMINATED
    
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
        return (thetaDot, thetaDotDot)
        # reverse signs from http://www.ece.ucsb.edu/courses/ECE594/594D_W10Byl/hw/cartpole_eom.pdf
        # because of different conventions in Wang

    def _earnedReward(self, s):
        if(self.GOAL_REGION[0] < s[StateIndex.THETA] < self.GOAL_REGION[1]):
            return self.GOAL_REWARD
        else: return 0
    def isTerminal(self,s):
        # Returns a boolean showing if s is terminal or not
        return self.NOT_TERMINATED # Pendulum has no absorbing state
if __name__ == '__main__':
    random.seed(0)
    p = InvertedPendulum(start_angle = 0, start_rate = 0, dt = 0.10, force_noise_max = 0);
    p.test(1000)
