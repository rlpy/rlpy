"""Pendulum base domain."""

from Tools import *
from Domain import *

# REQURES matplotlib for rk4 integration
from scipy import integrate # For integrate.odeint (accurate, slow)


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Alborz Geramifard", "Robert H. Klein"]


##########################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
##########################################################
# This is the parent class for pendulum-type problems with two
# (continuous) states only, and three discrete possible actions.
#
# \note Despite the naming convention in common use
# here and in the community, the dynamics of the
# Pendulum actually correspond to the 4-state system
# with a pendulum on a cart, with x and xDot states omitted.
# Thus, actions still take the form of forces and not torques.
#
# State: [theta, thetaDot]. \n
# Actions: [-50, 0, 50]
#
# theta = angular position of pendulum
# (relative to straight up at 0 rad), and positive clockwise. \n
# thetaDot = Angular rate of pendulum
#
# Actions take the form of force applied to cart; \n
# [-50, 0, 50] N force are the default available actions.
# Positive force acts to the right on the cart.
#
# Uniformly distributed noise is added with magnitude 10 N.
#
# All dynamics and conventions per Lagoudakis & Parr, 2003
#
# \note This domain cannot be instantiated; it is a superclass for the specific domains
# Pendulum_SwingUp and Pendulum_InvertedBalance.
##########################################################

class Pendulum(Domain):

    DEBUG               = 0     # Set to non-zero to enable print statements

    # Domain constants from children - Pendulum parameters are not standardized
    # (SwingUp and InvertedBalance parameters are different in the literature)
	## Newtons, N - Torque values available as actions
    AVAIL_FORCE         = None
	## kilograms, kg - Mass of the pendulum arm
    MASS_PEND           = None
	## kilograms, kg - Mass of cart
    MASS_CART           = None
	## meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH              = None
	## m/s^2 - gravitational constant
    ACCEL_G             = None
	## Time between steps
    dt                  = None
	## Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max     = None

    ANGLE_LIMITS        = None
    ANGULAR_RATE_LIMITS = None
	## Max number of steps per trajectory
    episodeCap          = None

    # Domain constants computed in __init__
	## m - Length of the moment-arm to the center of mass, equal to half the pendulum length
    MOMENT_ARM          = 0 # Note that some elsewhere refer to this simply as 'length' somewhat of a misnomer.
	## 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)
    _ALPHA_MASS         = 0

    # Internal Constants
    tol                     = 10 ** -5 # Tolerance used for pendulum_ode45 integration

    #Visual Stuff
    valueFunction_fig       = None
    policy_fig              = None
    MIN_RETURN              = None # Minimum return possible, used for graphical normalization, computed in init
    MAX_RETURN              = None # Minimum return possible, used for graphical normalization, computed in init
    circle_radius           = 0.05
    ARM_LENGTH              = 1.0
    PENDULUM_PIVOT_X        = 0 # X position is also fixed in this visualization
    PENDULUM_PIVOT_Y        = 0 # Y position of pendulum pivot
    pendulumArm             = None
    pendulumBob             = None
    actionArrow             = None
    domain_fig              = None
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
        self.xTicksLabels   = around(linspace(self.statespace_limits[0,0]*180/pi,self.statespace_limits[0,1]*180/pi,5)).astype(int)
        #self.xTicksLabels   = ["$-\\pi$","$-\\frac{\\pi}{2}$","$0$","$\\frac{\\pi}{2}$","$\\pi$"]
        self.yTicks         = [0,self.Theta_discretization/4.0,self.ThetaDot_discretization/2.0,self.ThetaDot_discretization*3/4.0,self.ThetaDot_discretization-1]
        self.yTicksLabels   = around(linspace(self.statespace_limits[1,0]*180/pi,self.statespace_limits[1,1]*180/pi,5)).astype(int)
        #self.yTicksLabels   = ["$-8\\pi$","$-4\\pi$","$0$","$4\\pi$","$8\\pi$"]
        self.DimNames       = ['Theta','Thetadot']
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

        # Transpose the matrices shown above, if they have not been already by another instance of the domain.
        if len(self._beta) == 5: self._beta = (self._beta).conj().transpose()
        if len(self._gamma) == 2: self._gamma = (self._gamma).conj().transpose()
        super(Pendulum,self).__init__(logger)

    def showDomain(self, a=0):
        s = self.state
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

        #Fix the arrows because they are forces not torques
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
            if (
                (forceAction > 0 and (-pi/2. < theta < pi/2.)) or
                (forceAction < 0 and not (-pi/2. < theta < pi/2.))
                ): # counterclockwise torque
                self.actionArrow = fromAtoB(SHIFT/2.0,.5*SHIFT,-SHIFT/2.0,-.5*SHIFT,'k',connectionstyle="arc3,rad=+1.2", ax =self.domain_fig)
            else:# clockwise torque
                self.actionArrow = fromAtoB(-SHIFT/2.0,.5*SHIFT,+SHIFT/2.0,-.5*SHIFT,'r',connectionstyle="arc3,rad=-1.2", ax =self.domain_fig)

        self.pendulumBob = mpatches.Circle((pendulumBobX,pendulumBobY), radius = self.circle_radius, color = 'blue')
        self.domain_fig.add_patch(self.pendulumBob)
        pl.draw()

    def showLearning(self,representation):
        granularity = 10.
        self.xTicks         = linspace(0,granularity * self.Theta_discretization - 1, 5)
        self.yTicks         = linspace(0,granularity * self.Theta_discretization - 1, 5)

        pi      = zeros((self.Theta_discretization*granularity, self.ThetaDot_discretization*granularity),'uint8')
        V       = zeros((self.Theta_discretization*granularity,self.ThetaDot_discretization*granularity))

        if self.valueFunction_fig is None:
            self.valueFunction_fig   = pl.subplot(1,3,2)
            self.valueFunction_fig   = pl.imshow(V, cmap='ValueFunction',interpolation='nearest',origin='lower',vmin=self.MIN_RETURN,vmax=self.MAX_RETURN)
            pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$ (degree)")
            pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
            pl.title('Value Function')

            self.policy_fig = pl.subplot(1,3,3)
            self.policy_fig = pl.imshow(pi, cmap='InvertedPendulumActions', interpolation='nearest',origin='lower',vmin=0,vmax=self.actions_num)
            pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$ (degree)")
            pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
            pl.title('Policy')
            pl.show()
            f = pl.gcf()
            f.subplots_adjust(left=0,wspace=.5)
            #pl.tight_layout()

        # Create the center of the grid cells both in theta and thetadot_dimension
        theta_binWidth      = (self.ANGLE_LIMITS[1]-self.ANGLE_LIMITS[0])/(self.Theta_discretization*granularity)
        thetas              = linspace(self.ANGLE_LIMITS[0]+theta_binWidth/2, self.ANGLE_LIMITS[1]-theta_binWidth/2, self.Theta_discretization*granularity)
        theta_dot_binWidth  = (self.ANGULAR_RATE_LIMITS[1]-self.ANGULAR_RATE_LIMITS[0])/(self.ThetaDot_discretization*granularity)
        theta_dots          = linspace(self.ANGULAR_RATE_LIMITS[0]+theta_dot_binWidth/2, self.ANGULAR_RATE_LIMITS[1]-theta_dot_binWidth/2, self.ThetaDot_discretization*granularity)
        #print thetas
        #print theta_dots
        for row, thetaDot in enumerate(theta_dots):
            for col, theta in enumerate(thetas):
                s = np.array([theta,thetaDot])
                As = self.possibleActions(s)
                terminal = self.isTerminal(s)
                Qs = representation.Qs(s, terminal)
                pi[row,col] = As[argmax(Qs)]
                V[row,col]  = max(Qs)
        #Update the value function
        # Wireframe, needs some work
        #X = linspace(self.ANGLE_LIMITS[0],self.ANGLE_LIMITS[1],self.Theta_discretization)
        #Y = linspace(self.ANGULAR_RATE_LIMITS[0],self.ANGULAR_RATE_LIMITS[1],self.ThetaDot_discretization)
        #X, Y = meshgrid(X, Y)
        #ax = pl.gcf().add_subplot(132,projection='3d',aspect='1')
        #ax.set_size_inches(6,6)
        #self.valueFunction_fig = ax.plot_surface(X, Y, V)#cmap='ValueFunction')
        #self.valueFunction_fig = ax.plot_wireframe(X, Y, V)#cmap='ValueFunction')

        norm = colors.Normalize(vmin=V.min(), vmax=V.max())
        self.valueFunction_fig.set_data(V)
        self.valueFunction_fig.set_norm(norm)
        self.policy_fig.set_data(pi)
        pl.draw()

    def s0(self):
        # Defined by children
        abstract

    def possibleActions(self, s=None): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)

    def step(self,a):
    # Simulate one step of the pendulum after taking force action a

        forceAction = self.AVAIL_FORCE[a]

        # Add noise to the force action
        if self.force_noise_max > 0:
            forceAction += self.random_state.uniform(-self.force_noise_max, self.force_noise_max)
        else: forceNoise = 0

        # Now, augment the state with our force action so it can be passed to _dsdt
        s_augmented = append(self.state, forceAction)

        #-------------------------------------------------------------------------#
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
        #-------------------------------------------------------------------------#

        # Decomment the 3 lines below to use mlab rk4 method.
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1] # only care about final timestep of integration returned by integrator
        ns = ns[0:2] # [theta, thetaDot]

         # wrap angle between -pi and pi (or whatever values assigned to ANGLE_LIMITS)
        ns[StateIndex.THETA]        = bound(wrap(ns[StateIndex.THETA],-pi, pi), self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT]    = bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        self.state = ns.copy()
        terminal                    = self.isTerminal()
        reward                      = self._getReward(a)
        return reward, ns, terminal, self.possibleActions()

    #
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

    ## \note \b CURRENTLY \b NOT \b IN \b USE - scipy.integrate functions preferred.
    def pendulum_ode45(self, t0, tfinal, y0, tol):
        # ODE function from "1Link" inverted pendulum implementation,
        # Lagoudakis & Parr 2003.
        #
        # Identical to L & P, with the change that only the
        # final state is outputted.  This improves performance, since
        # the output array does not need to be extended on each timestep.
        # An alternative would be to preallocate and limit the output to a finite size.
        #-------------------------------------------------------------------------------
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


    #  @param a: action
    #  @return: Reward earned for this state-action pair.
    def _getReward(self, a):
        # Return the reward earned for this state-action pair
        abstract

## \cond DEV
# Flexible way to index states in the Pendulum Domain
#
# This class enumerates the different indices used when indexing the state. \n
# e.g. s[StateIndex.THETA] is guaranteed to return the angle state.

class StateIndex:
    THETA, THETA_DOT = 0,1
    FORCE = 2 # Used by the state augmented with input in dynamics calculations


# \endcond
