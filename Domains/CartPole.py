import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

#from scipy import integrate # for integration of state

#########################################################
# Robert H Klein, Alborz Geramifard at MIT, Dec. 6 2012 #
#########################################################
#
# Per Lagoudakis and Parr 2003, derived from Wang 1996.
# (See "1Link" implementation by L & P.)
# Dynamics in x, xdot derived from CartPole implementation
# of rl-community.org, from Sutton and Barto's Pole-balancing
# task in <Reinforcement Learning: An Introduction> (1998)
# (See CartPole implementation in the RL Community,
# http://library.rl-community.org/wiki/CartPole)
#
# State: [theta, thetaDot, x, xDot].
# Actions: [-50, 0, 50]
#
# theta    = Angular position of pendulum
# (relative to straight up at 0 rad),and positive clockwise.
# thetaDot = Angular rate of pendulum
# x        = Linear position of the cart on its track (positive right).
# xDot     = Linear velocity of the cart on its track.
#
# Actions take the form of force applied to cart;
# [-50, 0, 50] N force are the default available actions.
# Positive force acts to the right on the cart.
#
# Uniformly distributed noise is added with magnitude 10 N.
#
######################################################


## @todo: can eliminate an entire dimension of the state space, xdot.
# However, must internally keep track of it for use in dynamics.
# RL_Glue and RL Community use the full 4-state system.
# @author: Robert H. Klein
class CartPole(Domain):
    
    # Domain constants per RL Community / RL_Glue CartPole implementation.
    # (http://code.google.com/p/rl-library/wiki/CartpoleJava)
    AVAIL_FORCE         = array([-10,0,10]) # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    MASS_PEND           = 0.1   # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = 1.0   # kilograms, kg - Mass of cart
    LENGTH              = 1.0   # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = 9.8   # m/s^2 - gravitational constant
    dt                  = 0.02  # Time between steps
    force_noise_max     = 1     # Newtons, N - Maximum noise possible, uniformly distributed
    
    ANGULAR_RATE_LIMITS = [-6.0, 6.0]     # Limits on pendulum rate [per RL Community CartPole]
    GOAL_REWARD         = 1               # Reward received on each step the pendulum is in the goal region
    POSITON_LIMITS      = [-2.4, 2.4]     # m - Limits on cart position [Per RL Community CartPole]
    VELOCITY_LIMITS     = [-6.0, 6.0]     # m/s - Limits on cart velocity [per RL Community CartPole]   
    
    # Domain constants


    episodeCap          = 3000      # Max number of steps per trajectory
    DEBUG               = 0     # Set to non-zero to enable print statements

    # Domain constants computed in __init__
    MOMENT_ARM          = 0     # m - Length of the moment-arm to the center of mass, equal to half the pendulum length
                            # Note that some elsewhere refer to this simply as 'length' somewhat of a misnomer.
    _ALPHA_MASS         = 0     # 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)
    
#    cur_action = 0 # Current action, stored so that it can be accessed by methods whose headers are fixed in python
#    cur_force_noise = 0 # Randomly generated noise for this timestep, stored here for the same reasons as cur_action
    
    
    # Plotting variables
    pendulumArm = None
    cartBox = None
    actionArrow = None
    ACTION_ARROW_LENGTH = 0.4
    domainFig = None
    circle_radius = 0.05
    PENDULUM_PIVOT_Y = 0 # Y position of pendulum pivot
    RECT_WIDTH = 0.5 
    RECT_HEIGHT = .4
    BLOB_WIDTH = RECT_HEIGHT/2.0
    PEND_WIDTH = 2 
    GROUND_WIDTH = 2
    GROUND_HEIGHT = 1
    # vertices for the ground:

    # are constrained by the format expected by ode functions.
    def __init__(self, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
#        self.states_num = inf       # Number of states
        self.actions_num        = len(self.AVAIL_FORCE)      # Number of Actions
        self.continuous_dims    = [StateIndex.THETA, StateIndex.THETA_DOT, StateIndex.X, StateIndex.X_DOT]
        
        self.MOMENT_ARM         = self.LENGTH / 2.0
        self._ALPHA_MASS        = 1.0 / (self.MASS_CART + self.MASS_PEND)
        
#        if self.RECT_HEIGHT == 0: # No rectangle height specified
#            self.RECT_HEIGHT = self.MASS_CART / 20 # Arbitrary number, reasonable visualization
#        if self.PEND_WIDTH == 0: # No rectangle height specified
#            self.PEND_WIDTH = int(self.MASS_PEND / self.LENGTH)+1 # Arbitrary number, reasonable visualization
        
        if self.logger: 
            self.logger.log("length:\t\t%0.2f(m)" % self.LENGTH)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)
        self._assignGroundVerts()
        super(CartPole,self).__init__(logger)
        
    def showDomain(self,s,a = 0):
        ## Plot the pendulum and its angle, along with an arc-arrow indicating the 
        # direction of torque applied (not including noise!)
        # Pendulum rotation is centered at origin
        
        if self.domainFig == None: # Need to initialize the figure
            self.domainFig = pl.gcf()
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
            viewableDistance = self.LENGTH + self.circle_radius + 0.5
            ax.set_xlim(self.POSITON_LIMITS[0] - viewableDistance, self.POSITON_LIMITS[1] + viewableDistance)
            ax.set_ylim(-viewableDistance, viewableDistance)
            #ax.set_aspect('equal')
            
            pl.show()
            
        forceAction = self.AVAIL_FORCE[a]
        curX = s[StateIndex.X]
        curXDot = s[StateIndex.X_DOT]
        curTheta = s[StateIndex.THETA]
        
        pendulumBobX = curX + self.LENGTH  * sin(curTheta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.LENGTH * cos(curTheta)

        if self.DEBUG: print 'Pendulum Position: ',pendulumBobX,pendulumBobY
        
        # update pendulum arm on figure
        self.pendulumArm.set_data([curX, pendulumBobX],[self.PENDULUM_PIVOT_Y, pendulumBobY])
        self.cartBox.set_x(curX - self.RECT_WIDTH/2.0)
        self.cartBlob.set_x(curX - self.BLOB_WIDTH/2.0)
        
        
        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None
            
        if forceAction == 0: pass # no force
        else: # cw or ccw torque
            if forceAction > 0: # rightward force
                self.actionArrow = fromAtoB(
                                                  curX - self.ACTION_ARROW_LENGTH - self.RECT_WIDTH/2.0, 0, 
                                                  curX - self.RECT_WIDTH/2.0,  0, 
                                                  'k',"arc3,rad=0",
                                                  0,0, 'simple'
                                                  )
            else:# leftward force
                self.actionArrow = fromAtoB(
                                                  curX + self.ACTION_ARROW_LENGTH + self.RECT_WIDTH/2.0, 0,
                                                  curX + self.RECT_WIDTH/2.0, 0, 
                                                  'r',"arc3,rad=0",
                                                  0,0,'simple'
                                                  )
            
        pl.draw()
#        sleep(self.dt)
    
    def showLearning(self,representation):
        pass
    
    def s0(self):
        # Defined by children
        abstract
    
    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)

    def step(self,s,a):
        # Simulate one step of the CartPole after taking action a
        # Note that at present, this is almost identical to the step for the Pendulum.
             
        forceAction = self.AVAIL_FORCE[a]
        
        # Add noise to the force action
        if self.force_noise_max > 0:
            forceAction += random.uniform(-self.force_noise_max, self.force_noise_max)
        else: forceNoise = 0
        
        # Now, augment the state with our force action so it can be passed to _dsdt
        s_augmented = append(s, forceAction)
        
        # Decomment the line below to use inbuilt runge-kutta method.
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1] # only care about final timestep of integration returned by integrator
        ns = ns[0:4] # [theta, thetadot, x, xDot]
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        #self.s_continuous = ns_continuous[-1] # We only care about the state at the ''final timestep'', self.dt

        theta                   = wrap(ns[StateIndex.THETA],-pi,pi)
        ns[StateIndex.THETA]    = bound(theta,self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT]= bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        ns[StateIndex.X]        = bound(ns[StateIndex.X], self.POSITON_LIMITS[0], self.POSITON_LIMITS[1])
        ns[StateIndex.X_DOT]    = bound(ns[StateIndex.X_DOT], self.VELOCITY_LIMITS[0], self.VELOCITY_LIMITS[1])
        terminal                    = self.isTerminal(ns)
        reward                      = self._getReward(ns,a)
        return reward, ns, terminal
    
    ## From CartPole implementation described in class definition, from rlcommunity.org
    # (http://library.rl-community.org/wiki/CartPole)
    # Used by odeint to numerically integrate the differential equation
    def _dsdt(self, s_augmented, t):
        # This function is needed for ode integration.  It calculates and returns the
        # derivatives at a given state, s.  The last element of s_augmented is the
        # force action taken, required to compute these derivatives.
        #
        # ThetaDotDot = 
        #
        #     g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        #     -----------------------------------------------------------------------
        #                           4l/3  -  (alpha)ml*cos^2(theta)
        #     
        #         g sin(theta) - w cos(theta)
        #   =     ---------------------------
        #         4l/3 - (alpha)ml*cos^2(theta)
        #
        # where w = (alpha)u + (alpha)ml*(tdot)^2*sin(theta)
        # Note we use the trigonometric identity sin(2theta)/2 = cos(theta)*sin(theta)
        #
        # xDotDot = w - (alpha)ml * thetaDotDot * cos(theta)

#        force = self.cur_action + self.cur_force_noise
        g = self.ACCEL_G
        l = self.MOMENT_ARM
        m_pendAlphaTimesL = self.MASS_PEND * self._ALPHA_MASS * l
        theta       = s_augmented[StateIndex.THETA]
        thetaDot    = s_augmented[StateIndex.THETA_DOT]
        xDot        = s_augmented[StateIndex.X_DOT]
        force       = s_augmented[StateIndex.FORCE]
        
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        thetaDotSq = thetaDot ** 2
        
        term1 = force*self._ALPHA_MASS + m_pendAlphaTimesL * thetaDotSq * sinTheta
        numer = g * sinTheta - cosTheta * term1
        denom = 4.0 * l / 3.0  -  m_pendAlphaTimesL * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = numer / denom
        
        xDotDot = term1 - m_pendAlphaTimesL * thetaDotDot * cosTheta
        return (thetaDot, thetaDotDot, xDot, xDotDot, 0) # final cell corresponds to action passed in
    
    ## @param s: state
    #  @param a: action
    ## @return: Reward earned for this state-action pair.
    def _getReward(self, s, a):
        # Return the reward earned for this state-action pair
        abstract
    
    ## Assigns the GROUND_VERTS array, placed here to avoid cluttered code in init.
    def _assignGroundVerts(self):
        minPosition = self.POSITON_LIMITS[0]-self.RECT_WIDTH/2.0
        maxPosition = self.POSITON_LIMITS[1]+self.RECT_WIDTH/2.0
        self.GROUND_VERTS = array([
                          (minPosition,-self.RECT_HEIGHT/2.0),
                          (minPosition,self.RECT_HEIGHT/2.0),
                          (minPosition-self.GROUND_WIDTH, self.RECT_HEIGHT/2.0),
                          (minPosition-self.GROUND_WIDTH, self.RECT_HEIGHT/2.0-self.GROUND_HEIGHT),
                          (maxPosition+self.GROUND_WIDTH, self.RECT_HEIGHT/2.0-self.GROUND_HEIGHT),
                          (maxPosition+self.GROUND_WIDTH, self.RECT_HEIGHT/2.0),
                          (maxPosition, self.RECT_HEIGHT/2.0),
                          (maxPosition, -self.RECT_HEIGHT/2.0),
                          ])
    
## Flexible way to index states in this domain.
#
# This class enumerates the different indices used when indexing the state.
# e.g. s[StateIndex.THETA] is guaranteed to return the angle state.
class StateIndex:
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3
    FORCE = 4
                
if __name__ == '__main__':
    print 'Please run one of my children: Swingup, InvertedBalanced'