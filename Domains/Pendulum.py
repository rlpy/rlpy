import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domain import *

from scipy import integrate # For integrate.odeint (accurate, slow)
from matplotlib.mlab import rk4
from matplotlib import lines

##########################################################
# Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012#
##########################################################
# This is the parent class for pendulum-type problems with two
# (continuous) states only, and three discrete possible actions.
#
# NOTE: Despite the naming convention in common use
# here and in the community, the dynamics of the
# Pendulum actually correspond to the 4-state system
# with a pendulum on a cart, with x and xDot states omitted.
# Thus, actions still take the form of forces and not torques.
#
# All dynamics and conventions per Lagoudakis & Parr, 2003
#
# State: [theta, thetaDot].
# Actions: [-50, 0, 50]
#
# theta = angular position of pendulum
# (relative to straight up at 0 rad),and positive clockwise.
# thetaDot = Angular rate of pendulum
#
# Actions take the form of force applied to cart;
# [-50, 0, 50] N force are the default available actions.
# Positive force acts to the right on the cart.
#
# Uniformly distributed noise is added with magnitude 10 N.
#
######################################################


## 
# NOTE: This domain cannot be instantiated; it is a superclass for the specific domains
# Pendulum_SwingUp and Pendulum_InvertedBalance.
# @author: Robert H. Klein
class Pendulum(Domain):

    DEBUG               = 0     # Set to non-zero to enable print statements
    
    # Domain constants from children
    AVAIL_FORCE         = None # Newtons, N - Torque values available as actions [-50,0,50 per DPF]
    MASS_PEND           = None # kilograms, kg - Mass of the pendulum arm
    MASS_CART           = None # kilograms, kg - Mass of cart
    LENGTH              = None # meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    ACCEL_G             = None # m/s^2 - gravitational constant
    dt                  = None # Time between steps
    force_noise_max     = None # Newtons, N - Maximum noise possible, uniformly distributed
    
    ANGLE_LIMITS        = None
    ANGULAR_RATE_LIMITS = None
    episodeCap          = None  # Max number of steps per trajectory
    
    # Domain constants computed in __init__
    MOMENT_ARM          = 0     # m - Length of the moment-arm to the center of mass, equal to half the pendulum length
                            # Note that some elsewhere refer to this simply as 'length' somewhat of a misnomer.
    _ALPHA_MASS         = 0     # 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)
    
    # Internal Constants
    tol                 = 10 ** -5 # Tolerance in Runge-Kutta 
    
    # Plotting variables
    pendulumArm         = None
    pendulumBob         = None
    actionArrow         = None
    domain_fig          = None
    circle_radius       = 0.05
    ARM_LENGTH          = 1.0 
    PENDULUM_PIVOT_X    = 0 # X position is also fixed in this visualization
    PENDULUM_PIVOT_Y    = 0 # Y position of pendulum pivot
    valueFunction_fig   = None
    policy_fig          = None
    MIN_RETURN          = 0 # Minimum return possible, used for graphical normalization, computed in init
    MAX_RETURN          = 0
    
    Theta_discretization    = 20 #Used for visualizing the policy and the value function
    ThetaDot_discretization = 20 #Used for visualizing the policy and the value function

    # Variables from pendulum_ode45.m of the 1Link code, Lagoudakis & Parr 2003
    # The Fehlberg coefficients: 
    _alpha = array([1/4.0, 3/8.0, 12/13.0, 1, 1/2.0])
    _beta = array([
               [1/4.0,      0,      0,    0,      0,    0],
               [3/32.0,9/32.0,      0,     0,      0,    0],
               [ 1932/2197.0,  -7200/2197.0,   7296/2197.0,     0,      0,    0], 
               [ 8341/4104.0, -32832/4104.0,  29440/4104.0,  -845/4104.0,      0,    0],
               [-6080/20520.0,  41040/20520.0, -28352/20520.0,  9295/20520.0,  -5643/20520.0,    0],
               ])
    _gamma = array([ 
              [902880/7618050.0,  0,  3953664/7618050.0,  3855735/7618050.0,  -1371249/7618050.0,  277020/7618050.0], 
              [ -2090/752400.0,  0,    22528/752400.0,    21970/752400.0,    -15048/752400.0,  -27360/752400.0], 
              ])
    _pow    = 1/5.0
    
    def __init__(self, logger = None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        self.actions_num        = len(self.AVAIL_FORCE)
        self.continuous_dims    = [StateIndex.THETA, StateIndex.THETA_DOT]
        
        self.MOMENT_ARM         = self.LENGTH / 2.0
        self._ALPHA_MASS        = 1.0 / (self.MASS_CART + self.MASS_PEND)
        
        self.xTicks         = linspace(0,self.Theta_discretization-1,5)
        self.xTicksLabels   = ["$-\\pi$","$-\\frac{\\pi}{2}$","$0$","$\\frac{\\pi}{2}$","$\\pi$"]
        self.yTicks         = [0,self.Theta_discretization/4.0,self.ThetaDot_discretization/2.0,self.ThetaDot_discretization*3/4.0,self.ThetaDot_discretization-1]
        self.yTicksLabels   = ["$-8\\pi$","$-4\\pi$","$0$","$4\\pi$","$8\\pi$"]
        
        if self.logger: 
            self.logger.log("length:\t\t%0.2f(m)" % self.LENGTH)
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
        self.MIN_RETURN = -1
        
        # Transpose the matrices shown above, if they have not been already by another instance of the domain.
        if len(self._beta) == 5: self._beta = (self._beta).conj().transpose()
        if len(self._gamma) == 2: self._gamma = (self._gamma).conj().transpose()
        super(Pendulum,self).__init__(logger)     
          
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
            viewableDistance = self.LENGTH + self.circle_radius + 0.5
            self.domain_fig.set_xlim(-viewableDistance, viewableDistance)
            self.domain_fig.set_ylim(-viewableDistance, viewableDistance)
            pl.axis('off')
            self.domain_fig.set_aspect('equal')
            pl.show()
            
        forceAction = self.AVAIL_FORCE[a]
        theta = s[StateIndex.THETA] # Using continuous state
        
        # recall we define 0deg  up, 90 deg right
        pendulumBobX = self.PENDULUM_PIVOT_X + self.LENGTH * sin(theta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.LENGTH * cos(theta)
        
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
#        sleep(self.dt)

    
    def showLearning(self,representation):
        
        pi      = zeros((self.Theta_discretization, self.ThetaDot_discretization),'uint8')            
        V       = zeros((self.Theta_discretization,self.ThetaDot_discretization))

        if self.valueFunction_fig is None:
            self.valueFunction_fig   = pl.subplot(1,3,2)
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
        
        for row, thetaDot in enumerate(linspace(self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1], self.ThetaDot_discretization)):
            for col, theta in enumerate(linspace(self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1], self.ThetaDot_discretization)):
                s           = [theta,thetaDot]
                Qs,As       = representation.Qs(s)
                pi[row,col] = representation.bestAction(s)
                V[row,col]  = max(Qs)
        
        self.valueFunction_fig.set_data(V)
        self.policy_fig.set_data(pi)
        pl.draw()
#        sleep(self.dt)

    def s0(self):
        # Defined by children
        abstract

    def possibleActions(self,s): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)

    def step(self,s,a):
    # Simulate one step of the pendulum after taking force action a
    
        forceAction = self.AVAIL_FORCE[a]
        
        # Add noise to the force action
        if self.force_noise_max > 0:
            forceAction += random.uniform(-self.force_noise_max, self.force_noise_max)
        else: forceNoise = 0
        
        # Now, augment the state with our force action so it can be passed to _dsdt
        s_augmented = append(s, forceAction)

    ###########################################################################
    # There are several ways of integrating the nonlinear dynamics equations. #
    # For consistency with prior results, we include the custom integration   #
    # method devloped by Lagoudakis & Parr (2003) for their Pendulum Inverted #
    # Balance task, despite its slightly poorer performance in execution time #
    # and accuracy of result.                                                 #
    # For our own experiments we use rk4 from mlab.                           #
    #                                                                         #
    # Integration method         Sample runtime  Error with scipy.integrate   #
    #                                              (most accurate method)     #
    #  rk4 (mlab)                  3min 15sec            < 0.01%              #
    #  pendulum_ode45 (L&P '03)    7min 15sec            ~ 2.00%              #
    #  integrate.odeint (scipy)    9min 30sec            -------              #
    #                                                                         #
    # Use of any of these methods is supported by selectively commenting      #
    # sections below.                                                         #
    ###########################################################################
    
        # Decomment the 3 lines below to use mlab rk4 method.
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1] # only care about final timestep of integration returned by integrator
        ns = ns[0:2] # [theta, thetaDot]
        
        # Decomment the 2 lines below to use L & P's pendulum_ode45 method
#        ns = self.pendulum_ode45(0, self.dt, s_augmented, self.tol)
#        ns = ns[0:2] # omit the augmented state with action stored
        
        # Decomment the 3 lines below to use scipy.integrate.odeint
#        ns = integrate.odeint(self._dsdt, s_augmented, [0, self.dt], rtol = self.tol)
#        ns = ns[-1]
#        ns = ns[0:2]

         # wrap angle between -pi and pi (or whatever values assigned to ANGLE_LIMITS)
        ns[StateIndex.THETA]        = wrap(ns[StateIndex.THETA],-pi, pi)
        ns[StateIndex.THETA_DOT]    = bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        terminal                    = self.isTerminal(ns)
        reward                      = self._getReward(ns,a)
        return reward, ns, terminal
   
    ##
    # @param s_augmented: {The state at which to compute derivatives, augmented with the current action.
    # Specifically @code (theta,thetaDot,forceAction) @endcode .}
    #
    # @return: {The derivatives of the state s_augmented, here (thetaDot, thetaDotDot).}
    #
    # Numerically integrate the differential equations forFrom Lagoudakis & Parr, 2003, described in class definition.
    # NOTE: These dynamics are, in reality, for a CartPole system, despite
    # Lagoudakis and Parr's terminology.
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
        #
        # Note we use the trigonometric identity sin(2theta)/2 = cos(theta)*sin(theta)
        
        
        g = self.ACCEL_G
        l = self.MOMENT_ARM # NOTE that 'length' in these computations is actually the length of the center of mass
        m_pendAlphaTimesL = self.MASS_PEND * self._ALPHA_MASS * l
        theta       = s_augmented[StateIndex.THETA]
        thetaDot    = s_augmented[StateIndex.THETA_DOT]
        force       = s_augmented[StateIndex.FORCE]
        
        sinTheta = sin(theta)
        cosTheta = cos(theta)
        thetaDotSq = thetaDot ** 2
        
        term1 = force*self._ALPHA_MASS + m_pendAlphaTimesL * thetaDotSq * sinTheta
        numer = g * sinTheta - cosTheta * term1
        denom = 4.0 * l / 3.0  -  m_pendAlphaTimesL * (cosTheta ** 2)
        
        thetaDotDot = numer / denom
        return (thetaDot, thetaDotDot, 0) # final cell corresponds to action passed in
    
#    def _dsdt(self, s_augmented, t):
#        # NOTE These dynamics correspond to a true pendulum, pinned at its base.
#        # Accepts TORQUE as input, not force.
#        # This function is needed for ode integration.  It calculates and returns the
#        # derivatives at a given state, s.  The last element of s_augmented is the
#        # force action taken, required to compute these derivatives.
#        #
#        # ThetaDotDot = 
#        #
#        #         mlg sin(theta) + T
#        #         -------------------
#        #                4l^2/3
#        #
#        # where T is the applied torque
#        
#        
#        g = self.ACCEL_G
#        l = self.MOMENT_ARM # NOTE that 'length' in these computations is actually the length of the center of mass
#        theta       = s_augmented[StateIndex.THETA]
#        torque       = s_augmented[StateIndex.TORQUE]
#        
#        thetaDotDot = (self.MASS_PEND *l * g * sin(theta) + torque) / (4.0/3.0 * l**2)
#        
#        return (s_augmented[StateIndex.THETA_DOT], thetaDotDot, 0) # final cell corresponds to action passed in
    
    ## 
    def pendulum_ode45(self, t0, tfinal, y0, tol):
    ### CURRENTLY NOT IN USE - scipy.integrate functions preferred. ###
    # ODE function from "1Link" inverted pendulum implementation,
    # Lagoudakis & Parr 2003.
    #
    # Identical to L & P, with the change that only the
    # final state is outputted.  This improves performance, since
    # the output array does not need to be extended on each timestep.
    # An alternative would be to preallocate and limit the output to a finite size.
    ###########################################################################
    #
    # ode45_us customized for the pendulum
    #
    #ODE45  Integrate a system of ordinary differential equations using 
    #       4th and 5th order Runge-Kutta formulas.  See also ODE23 and 
    #       ODEDEMO.M. 
    #       [T,Y] = ODE45('yprime', T0, Tfinal, Y0, ... 
    #                A, B1, B2, C, OBsq ) integrates the system 
    #       of ordinary differential equations described by the M-file 
    #       YPRIME.M over the interval T0 to Tfinal and using initial 
    #       conditions Y0. 
    #       [T, Y] = ODE45(F, T0, Tfinal, Y0, TOL, 1) uses tolerance TOL 
    #       and displays status while the integration proceeds. 
    # 
    # INPUT: 
    # t0    - Initial value of t. 
    # tfinal- Final value of t. 
    # y0    - Initial value column-vector. 
    # tol   - The desired accuracy. (Default: tol = 1.e-6). 
    # 
    # OUTPUT: 
    # T  - Returned integration time points (row-vector). 
    # Y  - Returned solution, one solution column-vector per tout-value. 
    # 
    # The result can be displayed by: plot(tout, yout). 
     
    #   C.B. Moler, 3-25-87. 
    #   Copyright (c) 1987 by the MathWorks, Inc. 
    #   All rights reserved. 
    
    # TODO - an option would be to pre-allocate 
        t = t0; 
            
        hmax = (tfinal - t)
        hmin = (tfinal - t)/1000.0
        h = (tfinal - t)
        y = y0[:] # y[2] contains force action
        #numStates = len(y)-1 # subtract 1 since last state element is actually an action
        numStates = len(y)
    #    y.shape = (numStates, 1)
        yStates = y[0:numStates]
        f = zeros([numStates, 6])
        tout = array([t])
        yout = y[:]
        tau = tol * max(linalg.norm(yStates, ord=inf), 1); #numpy.inf nump.linalg.norm
         
        
        ##### The main loop 
        
        while (t < tfinal) and (h >= hmin):
            if t + h > tfinal: h = tfinal - t 
            
            # Compute the slopes 
            f[:,0] = self._dsdt(y,t)
            dotHF = dot(h,f)
            for j in arange(0,5):
    # note that 'dot' is used to multiply numpy arrays (even shaped as matrices)
                f[:,j+1] = self._dsdt(y+dot(dotHF,self._beta[:,j]), t+self._alpha[j]*h)
    
            # Estimate the error and the acceptable error
            delta = linalg.norm(dot(dotHF,self._gamma[:,1]),ord=inf)
            tau = tol*max(linalg.norm(y,ord=inf),1.0); 
     
            # Update the solution only if the error is acceptable 
            if delta <= tau:
                t = t + h
                y = y + dot(dotHF,self._gamma[:,0])
                # Replace the two lines below to output all states and a vector of associated times.
#                tout = append(tout, t)
#                yout = vstack((yout,y))
                tout = t
                yout = y
     
          # Update the step size 
            if delta != 0.0: h = min(hmax, 0.8*h*(tau/delta) ** self._pow)          
        if (t < tfinal): print 'SINGULARITY LIKELY at: ',t 
        return yout # Optionally also return tout here.


    ## @param s: state
    #  @param a: action
    ## @return: Reward earned for this state-action pair.
    def _getReward(self, s, a):
        # Return the reward earned for this state-action pair
        abstract

## Flexible way to index states in this domain.
#
# This class enumerates the different indices used when indexing the state.
# e.g. s[StateIndex.THETA] is guaranteed to return the angle state.
class StateIndex:
    THETA, THETA_DOT = 0,1
    FORCE = 2 # Used by the state augmented with input in dynamics calculations

