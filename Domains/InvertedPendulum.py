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
    DEBUG               = 0 # Set to non-zero to enable print statements
    AVAIL_FORCE         = array([-50,0,50]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    FELL_REWARD         = -1
    ANGLE_LIMITS        = [-pi, pi] # rad - Limits on pendulum angle (NOTE we wrap the angle at 2*pi)
    ANGULAR_RATE_LIMITS =  [-8*pi, 8*pi] # Limits on pendulum rate [+-15pi in tutorial]
    start_angle         = 0 # rad - Starting angle of the pendulum
    start_rate          = 0 # rad/s - Starting rate of the pendulum
    MASS_PEND           = 2.0 # kilograms, kg - Mass of the bob at the end of the pendulum (assume zero arm mass) [2 per DPF]
    MASS_CART           = 8.0 # kilograms, kg - Mass of cart [8 per DPF]
    length              = 1.0 # meters, m - Length of the pendulum, meters [0.5 in DPF]
    ACCEL_G             = 9.81 # m/s^2 - gravitational constant
    ROT_INERTIA         = 0 # kg * m^2 - rotational inertia of the pendulum, computed in __init__
    dt                  = 0.1 # Time between steps [0.1 in DPF]
    force_noise_max     = 10 # Newtons, N - Maximum noise possible, uniformly distributed [10 in tutorial]
    episodeCap          = 3000 #Max Steps
    
    cur_action      = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
    cur_force_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    # Plotting variables
    pendulumArm         = None
    pendulumBob         = None
    actionArrow         = None
    domain_fig          = None
    circle_radius       = 0.1
    PENDULUM_PIVOT_X    = 0 # X position is also fixed in this visualization
    PENDULUM_PIVOT_Y    = 0 # Y position of pendulum pivot
    valueFunction_fig   = None
    policy_fig          = None
    MIN_RETURN          = 0 # Minimum return possible, used for graphical normalization, computed in init
    MAX_RETURN          = 0
    
    Theta_discritization    = 20 #Used for visualizing the policy and the value function
    ThetaDot_discritization = 20 #Used for visualizing the policy and the value function

    def __init__(self, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        self.actions_num        = len(self.AVAIL_FORCE)      # Number of Actions
        self.moment_arm         = self.length / 2.0
        self.rot_inertia        = self.MASS_PEND * self.moment_arm ** 2
        self.alpha              = 1.0 / (self.MASS_CART + self.MASS_PEND)
        self.continuous_dims    = [StateIndex.THETA, StateIndex.THETA_DOT]
        
        self.xTicks         = linspace(0,self.Theta_discritization-1,5)
        self.xTicksLabels   = ["$-\\pi$","$-\\frac{\\pi}{2}$","$0$","$\\frac{\\pi}{2}$","$\\pi$"]
        self.yTicks         = [0,self.Theta_discritization/4.0,self.ThetaDot_discritization/2.0,self.ThetaDot_discritization*3/4.0,self.ThetaDot_discritization-1]
        self.yTicksLabels   = ["$-8\\pi$","$-4\\pi$","$0$","$4\\pi$","$8\\pi$"]
        
        if self.logger: 
            self.logger.log("length:\t\t%0.2f(m)" % self.length)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)

#        if not ((2*pi / dt > self.ANGULAR_RATE_LIMITS[1]) and (2*pi / dt > -self.ANGULAR_RATE_LIMITS[0])):
#            print '''
#            WARNING:
#            # This has not been observed in practice, but conceivably
#            # if the bound on angular velocity is large compared with
#            # the time discretization, seemingly 'optimal' performance
#            # might result from a stroboscopic-like effect.
#            # For example, if dt = 1.0 sec, and the angular rate limits
#            # exceed -2pi or 2pi respectively, then it is possible that
#            # between consecutive timesteps, the pendulum will have
#            # the same position, even though it really completed a
#            # rotation, and thus we will find a solution that commands
#            # the pendulum to spin with angular rate in multiples of
#            # 2pi / dt.
#            '''
#            print 'Your selection, dt=',self.dt,'and limits',self.ANGULAR_RATE_LIMITS,'Are at risk.'
#            print 'Reduce your timestep dt (to increase # timesteps) or reduce angular rate limits so that 2pi / dt > max(AngularRateLimit)'
#            print 'Currently, 2pi / dt = ',2*pi/self.dt,', angular rate limits shown above.'
        self.MAX_RETURN = 0
        self.MIN_RETURN = self.FELL_REWARD
        
        super(InvertedPendulum,self).__init__(logger)       
    def showDomain(self,s,a = 0):
        # Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        
        if self.domain_fig == None: # Need to initialize the figure
            self.domain_fig = pl.subplot(1,3,1)
            self.pendulumArm = lines.Line2D([],[], linewidth = 3, color='black')
            self.pendulumBob = mpatches.Circle((0,0), radius = self.circle_radius)
            
            self.domain_fig.add_patch(self.pendulumBob)
            self.domain_fig.add_line(self.pendulumArm)
            # Allow room for pendulum to swing without getting cut off on graph
            viewableDistance = self.length + self.circle_radius + 0.5
            self.domain_fig.set_xlim(-viewableDistance, viewableDistance)
            self.domain_fig.set_ylim(-viewableDistance, viewableDistance)
            pl.axis('off')
            self.domain_fig.set_aspect('equal')
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
        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None
        
        if forceAction == 0: 
            pass # no torque
        else: # cw or ccw torque
            SHIFT = .5
            if forceAction > 0: # counterclockwise torque
                self.actionArrow = fromAtoB(SHIFT/2.0,.5*SHIFT,-SHIFT/2.0,-.5*SHIFT,'k',connectionstyle="arc3,rad=+1.2")
            else:# clockwise torque
                self.actionArrow = fromAtoB(-SHIFT/2.0,.5*SHIFT,+SHIFT/2.0,-.5*SHIFT,'r',connectionstyle="arc3,rad=-1.2")
            
        self.pendulumBob = mpatches.Circle((pendulumBobX,pendulumBobY), radius = self.circle_radius, color = 'blue')
        self.domain_fig.add_patch(self.pendulumBob)
        pl.draw()
    def showLearning(self,representation):
        
        pi      = zeros((self.Theta_discritization, self.ThetaDot_discritization),'uint8')            
        V       = zeros((self.Theta_discritization,self.ThetaDot_discritization))

        if self.valueFunction_fig is None:
            self.valueFunction_fig  = pl.subplot(1,3,2)
            self.valueFunction_fig   = pl.imshow(V, cmap='ValueFunction',interpolation='nearest',vmin=self.MIN_RETURN,vmax=self.MAX_RETURN) 
            #pl.colorbar() # Show the colorbar corresponding to the value function
            pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$")
            pl.ylabel(r"$\dot{\theta}$")
            pl.title('Value Function')
            
            self.policy_fig = pl.subplot(1,3,3)
            self.policy_fig = pl.imshow(pi, cmap='InvertedPendulumActions', interpolation='nearest',vmin=0,vmax=self.actions_num)
            pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$")
            pl.ylabel(r"$\dot{\theta}$")
            pl.title('Policy')
#            f.set_size_inches(10,20)
            pl.show()
            f = pl.gcf()
            f.subplots_adjust(left=0,wspace=.3)
            #pl.tight_layout()
        
        for row, thetaDot in enumerate(linspace(self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1], self.ThetaDot_discritization)):
            for col, theta in enumerate(linspace(self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1], self.ThetaDot_discritization)):
                s           = [theta,thetaDot]
                Qs,As       = representation.Qs(s)
                pi[row,col] = representation.bestAction(s)
                V[row,col]  = max(Qs)
        
        self.valueFunction_fig.set_data(V)
        self.policy_fig.set_data(pi)
        pl.draw()
    def s0(self):    
        # Returns the initial state, [theta0, thetaDot0]
        return array([0,0])
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
        ns[StateIndex.THETA]        = wrap(ns[StateIndex.THETA],self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT]    = bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        terminal                    = self.isTerminal(ns)
        reward                      = self.FELL_REWARD if terminal else 0 
        return reward, ns, terminal
    def _dsdt(self, s_continuous, t):
    # def _dsdt(self,t, s_continuous):
        # This function is needed for ode integration.  It calculates and returns the derivatives
        # at a given state
        force = self.cur_action + self.cur_force_noise
        g = self.ACCEL_G
        l = self.moment_arm
        m_pendAlphaTimesL = self.MASS_PEND * self.alpha * l
        theta       = s_continuous[StateIndex.THETA]
        thetaDot    = s_continuous[StateIndex.THETA_DOT]
        
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
    def isTerminal(self,s):
        return not (-pi/2.0 < s[StateIndex.THETA] < pi/2.0)

if __name__ == '__main__':
    random.seed(0)
    p = InvertedPendulum();
    p.test(1000)
