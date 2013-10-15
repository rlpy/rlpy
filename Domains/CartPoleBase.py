"""Base class for all cartpole domains"""

from Domain import Domain
import numpy as np
import scipy.integrate
from Tools import pl, mpatches, mpath, fromAtoB, lines, rk4, wrap, bound, colors

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Robert H. Klein"]
pi = np.pi

class CartPoleBase(Domain):
    """
    Base class for all CartPole domains.  Dynamics shared across all, but
    domain task (and associated reward function and termination condition)
    as well as number of states varies among subclasses.
    See ``InfTrackCartPole`` and ``FiniteTrackCartPole``.
    
    Domain constants (such as ``AVAIL_FORCE``, ``MASS_PEND``, and ``LENGTH``)
    vary as well.
    
    .. note::

        Changing any of the state limits (e.g. ``ANGLE_LIMITS``,
        ``POSITION_LIMITS``, etc.) will affect the resolution of the
        discretized value function representation (and thus the policy as well)
    
    .. note::
        This is an abstract class and cannot be instantiated directly.
    
    """
    #: Newtons, N - Force values available as actions
    AVAIL_FORCE         = np.array([-10, 10])
    #: kilograms, kg - Mass of the pendulum arm
    MASS_PEND           = 0.1
    #: kilograms, kg - Mass of cart
    MASS_CART           = 1.0
    #: meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH              = 1.0
    #: m/s^2 - gravitational constant
    ACCEL_G             = 9.8
    #: seconds, s - Time between steps
    dt                  = 0.02
    #: Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max     = 0.

    #: integration type, can be 'rk4', 'odeint' or 'euler'. Defaults to rk4.
    int_type = 'euler'

    #: number of steps for Euler integration
    num_euler_steps = 1

    #: Limits on pendulum rate [per RL Community CartPole]
    ANGULAR_RATE_LIMITS = [-6.0, 6.0]
    #: Limits on pendulum angle (wrapped around a full cycle)
    ANGLE_LIMITS        = [-pi, pi]
    #: Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1

    # Domain constants

    #: Max number of steps per trajectory
    episodeCap          = 3000
    #: Set to non-zero to enable print statements
    DEBUG               = 0

    # m - Length of moment-arm to center of mass (= half the pendulum length)
    MOMENT_ARM          = LENGTH / 2.
    # 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)
    _ALPHA_MASS         = 1.0 / (MASS_CART + MASS_PEND)
    
    
    
    ### Domain visual objects
    pendulumArm         = None
    cartBox             = None
    actionArrow         = None
    domainFig           = None
    
    ### Domain visual constants (these DO NOT affect dynamics, purely visual).
    ACTION_ARROW_LENGTH = 0.4   # length of arrow showing force action on cart
    PENDULUM_PIVOT_Y    = 0     # Y position of pendulum pivot
    RECT_WIDTH          = 0.5   # width of cart rectangle
    RECT_HEIGHT         = 0.4   # height of cart rectangle
    BLOB_WIDTH          = RECT_HEIGHT / 2.0 # visual square at cart center of mass
    PEND_WIDTH          = 2     # width of pendulum arm rectangle
    GROUND_WIDTH        = 2     # width of ground rectangle
    GROUND_HEIGHT       = 1     # height of ground rectangle

    ### Value and policy figure objects + constants
    #
    valueFunction_fig       = None
    #
    policy_fig              = None
    # Minimum return possible, used for graphic normalization, computed in init
    MIN_RETURN              = None
    # Maximum return possible, used for graphic normalization, computed in init
    MAX_RETURN              = None
    
    def __init__(self, logger=None):
        """
        Setup required domain constants.
        """
        self.actions_num        = len(self.AVAIL_FORCE)      # Number of Actions
        
        if not self._isParamsValid():
            # For now, continue execution with warnings.
            # Optionally halt execution here.
            pass
            

        if self.logger:
            self.logger.log("Pendulum length:\t\t%0.2f(m)" % self.LENGTH)
            self.logger.log("Timestep length dt:\t\t\t%0.2f(s)" % self.dt)
        self._assignGroundVerts()
        super(CartPoleBase, self).__init__(logger)
    
    def _isParamsValid(self):
        if not ((2*pi / dt > self.ANGULAR_RATE_LIMITS[1]) and (2*pi / dt > -self.ANGULAR_RATE_LIMITS[0])):
            errStr = """
            WARNING:
            If the bound on angular velocity is large compared with
            the time discretization, seemingly 'optimal' performance
            might result from a stroboscopic-like effect.
            For example, if dt = 1.0 sec, and the angular rate limits
            exceed -2pi or 2pi respectively, then it is possible that
            between consecutive timesteps, the pendulum will have
            the same position, even though it really completed a
            rotation, and thus we will find a solution that commands
            the pendulum to spin with angular rate in multiples of
            2pi / dt instead of truly remaining stationary (0 rad/s) at goal.
            """
            print errStr
            print 'Your selection, dt=',self.dt,'and limits',self.ANGULAR_RATE_LIMITS,'Are at risk.'
            print 'Reduce your timestep dt (to increase # timesteps) or reduce angular rate limits so that 2pi / dt > max(AngularRateLimit)'
            print 'Currently, 2pi / dt = ',2*pi/self.dt,', angular rate limits shown above.'
            return False
        
        return True
    
    ## Assigns the GROUND_VERTS array, placed here to avoid cluttered code in init.
    def _assignGroundVerts(self):
        """
        Assign the ``GROUND_VERTS`` array, which defines the coordinates used
        for domain visualization.
        
        .. note::
            This function *does not* affect system dynamics.
            
        """
        minPosition = self.POSITON_LIMITS[0]-self.RECT_WIDTH/2.0
        maxPosition = self.POSITON_LIMITS[1]+self.RECT_WIDTH/2.0
        self.GROUND_VERTS = np.array([
            (minPosition, -self.RECT_HEIGHT / 2.0),
            (minPosition,self.RECT_HEIGHT / 2.0),
            (minPosition-self.GROUND_WIDTH, self.RECT_HEIGHT/2.0),
            (minPosition-self.GROUND_WIDTH, self.RECT_HEIGHT/2.0-self.GROUND_HEIGHT),
            (maxPosition+self.GROUND_WIDTH, self.RECT_HEIGHT/2.0-self.GROUND_HEIGHT),
            (maxPosition+self.GROUND_WIDTH, self.RECT_HEIGHT/2.0),
            (maxPosition, self.RECT_HEIGHT/2.0),
            (maxPosition, -self.RECT_HEIGHT/2.0),
        ])
        
    def _getReward(self, a, s=None):
        """
        Obtain the reward corresponding to this domain implementation
        (e.g. for swinging the pendulum up vs. balancing).
        
        """
        raise NotImplementedError
    
    def s0(self):
        raise NotImplementedError
    
    def possibleActions(self, s=None):
        """
        Returns an integer for each available action.  Some child domains allow
        different numbers of actions.
        """
        return np.arange(self.actions_num)

    def step(self, a):

        forceAction = self.AVAIL_FORCE[a]

        # Add noise to the force action
        if self.force_noise_max > 0:
            forceAction += self.random_state.uniform(-self.force_noise_max, self.force_noise_max)

        # Now, augment the state with our force action so it can be passed to _dsdt
        s_augmented = np.append(self.state, forceAction)
        if self.int_type == "euler":
            int_fun = self.euler_int
        elif self.int_type == "odeint":
            int_fun = scipy.integrate.odeint
        else:
            int_fun = rk4
        ns = int_fun(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1]  # only care about final timestep of integration returned by integrator

        ns = ns[0:4]  # [theta, thetadot, x, xDot]
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        #self.s_continuous = ns_continuous[-1] # We only care about the state at the ''final timestep'', self.dt

        theta                   = wrap(ns[StateIndex.THETA], -np.pi, np.pi)
        ns[StateIndex.THETA]    = bound(theta, self.ANGLE_LIMITS[0], self.ANGLE_LIMITS[1])
        ns[StateIndex.THETA_DOT]= bound(ns[StateIndex.THETA_DOT], self.ANGULAR_RATE_LIMITS[0], self.ANGULAR_RATE_LIMITS[1])
        ns[StateIndex.X]        = bound(ns[StateIndex.X], self.POSITON_LIMITS[0], self.POSITON_LIMITS[1])
        ns[StateIndex.X_DOT]    = bound(ns[StateIndex.X_DOT], self.VELOCITY_LIMITS[0], self.VELOCITY_LIMITS[1])
        self.state = ns.copy()
        terminal                    = self.isTerminal()
        reward                      = self._getReward(a)
        return reward, ns, terminal, self.possibleActions()
    
    def _dsdt(self, s_augmented, t):
        """
        This function is needed for ode integration.  It calculates and returns the
        derivatives at a given state, s.  The last element of s_augmented is the
        force action taken, required to compute these derivatives.
        
        The equation used::
        
            ThetaDotDot =
            
              g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
              -----------------------------------------------------------------------
                                    4l/3  -  (alpha)ml*cos^2(theta)
            
                  g sin(theta) - w cos(theta)
            =     ---------------------------
                  4l/3 - (alpha)ml*cos^2(theta)
            
            where w = (alpha)u + (alpha)ml*(tdot)^2*sin(theta)
            Note we use the trigonometric identity sin(2theta)/2 = cos(theta)*sin(theta)
            
            xDotDot = w - (alpha)ml * thetaDotDot * cos(theta)
        
        """
        g = self.ACCEL_G
        l = self.MOMENT_ARM
        m_pendAlphaTimesL = self.MASS_PEND * self._ALPHA_MASS * l
        theta       = s_augmented[StateIndex.THETA]
        thetaDot    = s_augmented[StateIndex.THETA_DOT]
        xDot        = s_augmented[StateIndex.X_DOT]
        force       = s_augmented[StateIndex.FORCE]

        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        term1 = force*self._ALPHA_MASS + m_pendAlphaTimesL * (thetaDot ** 2) * sinTheta
        numer = g * sinTheta - cosTheta * term1
        denom = 4.0 * l / 3.0  -  m_pendAlphaTimesL * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = numer / denom

        xDotDot = term1 - m_pendAlphaTimesL * thetaDotDot * cosTheta
        return np.array((thetaDot, thetaDotDot, xDot, xDotDot, 0))  # final cell corresponds to action passed in
    
    def showDomain(self, a=0):
        """
        Display the 4-d state of the cartpole and arrow indicating current
        force action (not including noise!).
        Note that for 2-D systems the cartpole is still displayed, but appears
        static; see ``InfTrackCartPole``.
        
        """
        s = self.state
        if self.domainFig is None:  # Need to initialize the figure
            self.domainFig = pl.figure("Domain")
            ax = self.domainFig.add_axes([0, 0, 1, 1], frameon=True, aspect=1.)
            self.pendulumArm = lines.Line2D([], [], linewidth = self.PEND_WIDTH, color='black')
            self.cartBox    = mpatches.Rectangle([0, self.PENDULUM_PIVOT_Y - self.RECT_HEIGHT / 2.0], self.RECT_WIDTH, self.RECT_HEIGHT, alpha=.4)
            self.cartBlob   = mpatches.Rectangle([0, self.PENDULUM_PIVOT_Y - self.BLOB_WIDTH / 2.0], self.BLOB_WIDTH, self.BLOB_WIDTH, alpha=.4)
            ax.add_patch(self.cartBox)
            ax.add_line(self.pendulumArm)
            ax.add_patch(self.cartBlob)
            #Draw Ground
            groundPath    = mpath.Path(self.GROUND_VERTS)
            groundPatch   = mpatches.PathPatch(groundPath,hatch="//")
            ax.add_patch(groundPatch)
            self.timeText = ax.text(self.POSITON_LIMITS[1], self.LENGTH,"")
            self.rewardText = ax.text(self.POSITON_LIMITS[0], self.LENGTH,"")
            # Allow room for pendulum to swing without getting cut off on graph
            viewableDistance = self.LENGTH + self.circle_radius + 0.5
            ax.set_xlim(self.POSITON_LIMITS[0] - viewableDistance, self.POSITON_LIMITS[1] + viewableDistance)
            ax.set_ylim(-viewableDistance, viewableDistance)
            #ax.set_aspect('equal')

            pl.show()

        self.domainFig = pl.figure("Domain")
        forceAction = self.AVAIL_FORCE[a]
        curX = s[StateIndex.X]
        curTheta = s[StateIndex.THETA]

        pendulumBobX = curX + self.LENGTH  * np.sin(curTheta)
        pendulumBobY = self.PENDULUM_PIVOT_Y + self.LENGTH * np.cos(curTheta)

        r = self._getReward(a, s=s)
        self.rewardText.set_text("Reward {0:g}".format(r, pendulumBobX, pendulumBobY))
        if self.DEBUG: print 'Pendulum Position: ',pendulumBobX,pendulumBobY

        # update pendulum arm on figure
        self.pendulumArm.set_data([curX, pendulumBobX],[self.PENDULUM_PIVOT_Y, pendulumBobY])
        self.cartBox.set_x(curX - self.RECT_WIDTH/2.0)
        self.cartBlob.set_x(curX - self.BLOB_WIDTH/2.0)


        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None

        if forceAction == 0: pass # no force
        else:  # cw or ccw torque
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
    
    
    def showLearning(self, representation):
        """
        Shows the policy and value function of the cart (with resolution
        dependent on the representation.
        
        :param representation: learned value function representation to display
            (optimal policy computed w.r.t. this representation)
        
        .. note::
            this function only shows the ``theta`` and ``thetaDot`` dimensions;
            for the 4-state cartpole systems, ``x`` and ``xdot`` are assumed 0.
        """
        
        if # state has length 4:
            errStr = "WARNING: showLearning() called with 4-state "
                "cartpole; no plot shown."
            if self.logger:
                self.logger.log(errStr)
            else: print errStr
            
            return
        # else we continue and display the plot
        
        # multiplies each dimension, used to make displayed grid appear finer
        granularity = 5.
        
        
        # TODO: currently we only allow for equal discretization in all
        # dimensions.  In future, must modify below code to handle non-uniform
        # discretization.
        thetaDiscr = representation.discretization
        thetaDotDiscr = representation.discretization
        
        
        pi = np.zeros((thetaDiscr*granularity, self.thetaDotDiscr*granularity),'uint8')
        V = np.zeros((thetaDiscr*granularity,self.thetaDotDiscr*granularity))

        if self.valueFunction_fig is None:
            self.valueFunction_fig   = pl.figure("Value Function")
            self.valueFunction_fig   = pl.imshow(V, cmap='ValueFunction',interpolation='nearest',origin='lower',vmin=self.MIN_RETURN,vmax=self.MAX_RETURN)
            #pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            #pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$ (degree)")
            pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
            pl.title('Value Function')

            self.policy_fig = pl.figure("Policy")
            self.policy_fig = pl.imshow(pi, cmap='InvertedPendulumActions', interpolation='nearest',origin='lower',vmin=0,vmax=self.actions_num)
            #pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            #pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            pl.xlabel(r"$\theta$ (degree)")
            pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
            pl.title('Policy')
            pl.show()
            f = pl.gcf()
            f.subplots_adjust(left=0,wspace=.5)

        # Create the center of the grid cells both in theta and thetadot_dimension
        theta_binWidth      = (self.ANGLE_LIMITS[1]-self.ANGLE_LIMITS[0])/(thetaDiscr*granularity)
        thetas              = np.linspace(self.ANGLE_LIMITS[0]+theta_binWidth/2, self.ANGLE_LIMITS[1]-theta_binWidth/2, thetaDiscr*granularity)
        theta_dot_binWidth  = (self.ANGULAR_RATE_LIMITS[1]-self.ANGULAR_RATE_LIMITS[0])/(self.thetaDotDiscr*granularity)
        theta_dots          = np.linspace(self.ANGULAR_RATE_LIMITS[0]+theta_dot_binWidth/2, self.ANGULAR_RATE_LIMITS[1]-theta_dot_binWidth/2, self.thetaDotDiscr*granularity)
        for row, thetaDot in enumerate(theta_dots):
            for col, theta in enumerate(thetas):
                s           = np.array([theta,thetaDot, 0, 0])
                # Array of Q-function evaluated at all possible actions at state s
                Qs       = representation.Qs(s, False)
                # Array of all possible actions at state s
                As       = self.possibleActions(s=s)
                # Assign pi to be optimal action (which maximizes Q-function)
                pi[row,col] = As[np.argmax(Qs)]
                # Assign V to be the value of the Q-function under optimal action
                V[row,col]  = max(Qs)

        norm = colors.Normalize(vmin=V.min(), vmax=V.max())
        self.valueFunction_fig.set_data(V)
        self.valueFunction_fig.set_norm(norm)
        self.policy_fig.set_data(pi)
        pl.figure("Policy")
        pl.draw()
        pl.figure("Value Function")
        pl.draw()
        
#     def euler_int(self, df, x0, times):
#         """
#         performs Euler integration with interface similar to other methods.
#         
#         :param df: TODO
#         :param x0: initial state
#         :param times: times at which to estimate integration values
#         
#         .. warning::
#             All but the final cell of ``times`` argument are ignored.
#             
#         """
#         steps = self.num_euler_steps
#         dt = float(times[-1])
#         ns = x0.copy()
#         for i in range(steps):
#             ns += dt / steps * df(ns, i * dt / steps)
#         return [ns]
        
class StateIndex:
    """
    Flexible way to index states in the CartPole Domain.

    This class enumerates the different indices used when indexing the state. \n
    e.g. ``s[StateIndex.THETA]`` is guaranteed to return the angle state.

    """
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3
    FORCE = 4 # i.e., action
