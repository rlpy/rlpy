"""Cart with a pole domains"""

from Domain import Domain
import numpy as np
import scipy.integrate
from Tools import pl, mpatches, mpath, fromAtoB, lines, rk4, wrap, bound, colors

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Robert H. Klein", "Alborz Geramifard"]
pi = np.pi

class CartPole(Domain):
    """
    | **State**
    theta    = Angular position of pendulum
    (relative to straight up at 0 rad), positive clockwise. \n
    thetaDot = Angular rate of pendulum. positive clockwise. \n
    x        = Linear position of the cart on its track (positive right). \n
    xDot     = Linear velocity of the cart on its track.

    | **Actions**
    Actions take the form of force applied to cart; \n
    [-50, 0, 50] N force are the default available actions. \n
    Positive force acts to the right on the cart.

    Uniformly distributed noise is added with magnitude 10 N.

    | **Reference**
    For details, see:
        Michail G. Lagoudakis, Ronald Parr, and L. Bartlett
        Least-squares policy iteration.  Journal of Machine Learning Research
        (2003) Issue 4.
    
        .. note::
            Note: for full domain description, see:
            Wang, H., Tanaka, K., and Griffin, M. An approach to fuzzy control
            of nonlinear systems; Stability and design issues.
            IEEE Trans. on Fuzzy Systems, 4(1):14-23, 1996.
        
        .. note::
            Note: See also `RL-Library CartPole <http://library.rl-community.org/wiki/CartPole>`_
            Domain constants per RL Community / RL_Glue
            `CartPole implementation <http://code.google.com/p/rl-library/wiki/CartpoleJava>`_
    
    """
    #: Newtons, N - Torque values available as actions [-50,0,50 per DPF]
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

    #: integration type, can be 'rk4', 'odeint' or 'euler'
    int_type = 'euler'

    #: number of steps for Euler integration
    num_euler_steps = 1

    #: Limits on pendulum rate [per RL Community CartPole]
    ANGULAR_RATE_LIMITS = [-6.0, 6.0]
    #: Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1
    #: m - Limits on cart position [Per RL Community CartPole]
    POSITON_LIMITS      = [-2.4, 2.4]
    #: m/s - Limits on cart velocity [per RL Community CartPole]
    VELOCITY_LIMITS     = [-6.0, 6.0]

    # Domain constants

    #: Max number of steps per trajectory
    episodeCap          = 3000
    #: Set to non-zero to enable print statements
    DEBUG               = 0

    #: m - Length of the moment-arm to the center of mass, equal to half the pendulum length
    MOMENT_ARM          = LENGTH / 2.
    #: 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND + MASS_CART)
    _ALPHA_MASS         = 1.0 / (MASS_CART + MASS_PEND)

    def __init__(self, logger=None):
        # Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        #        self.states_num = inf       # Number of states
        self.actions_num        = len(self.AVAIL_FORCE)      # Number of Actions
        self.continuous_dims    = [StateIndex.THETA, StateIndex.THETA_DOT, StateIndex.X, StateIndex.X_DOT]

        self.DimNames           = ['Theta', 'Thetadot', 'X', 'Xdot']

        if self.logger:
            self.logger.log("length:\t\t%0.2f(m)" % self.LENGTH)
            self.logger.log("dt:\t\t\t%0.2f(s)" % self.dt)
        self._assignGroundVerts()
        super(CartPole, self).__init__(logger)

    def showDomain(self, a=0):
        """Plot the pendulum and its angle, along with an arc-arrow indicating the
        direction of torque applied (not including noise!)
        Pendulum rotation is centered at origin"""
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
            path    = mpath.Path(self.GROUND_VERTS)
            patch   = mpatches.PathPatch(path,hatch="//")
            ax.add_patch(patch)
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

    def possibleActions(self, s=None): # Return list of all indices corresponding to actions available
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

    def euler_int(self, df, x0, times):
        """
        performs Euler integration with interface similar to other methods.
        BEWARE: times argument ignored
        """
        steps = self.num_euler_steps
        dt = float(times[-1])
        ns = x0.copy()
        for i in range(steps):
            ns += dt / steps * df(ns, i * dt / steps)
        return [ns]


    def _dsdt(self, s_augmented, t):
        """
        This function is needed for ode integration.  It calculates and returns the
        derivatives at a given state, s.  The last element of s_augmented is the
        force action taken, required to compute these derivatives.

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
        thetaDotSq = thetaDot ** 2

        term1 = force*self._ALPHA_MASS + m_pendAlphaTimesL * thetaDotSq * sinTheta
        numer = g * sinTheta - cosTheta * term1
        denom = 4.0 * l / 3.0  -  m_pendAlphaTimesL * (cosTheta ** 2)
        # g sin(theta) - (alpha)ml(tdot)^2 * sin(2theta)/2  -  (alpha)cos(theta)u
        # -----------------------------------------------------------------------
        #                     4l/3  -  (alpha)ml*cos^2(theta)
        thetaDotDot = numer / denom

        xDotDot = term1 - m_pendAlphaTimesL * thetaDotDot * cosTheta
        return np.array((thetaDot, thetaDotDot, xDot, xDotDot, 0))  # final cell corresponds to action passed in

    def _getReward(self, a, s=None):
        return NotImplementedError

    ## Assigns the GROUND_VERTS array, placed here to avoid cluttered code in init.
    def _assignGroundVerts(self):
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

    def showLearning(self, representation):
        """
        Shows the policy and value function of the cart (with resolution
        dependent on the representation.
        
        :param representation: learned value function representation to display
            (optimal policy computed w.r.t. this representation)
        """
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
                Qs       = representation.Qs(s, False)
                As       = self.possibleActions(s=s)
                pi[row,col] = As[np.argmax(Qs)]
                V[row,col]  = max(Qs)

        norm = colors.Normalize(vmin=V.min(), vmax=V.max())
        self.valueFunction_fig.set_data(V)
        self.valueFunction_fig.set_norm(norm)
        self.policy_fig.set_data(pi)
        pl.figure("Policy")
        pl.draw()
        pl.figure("Value Function")
        pl.draw()

class StateIndex:
    """
    Flexible way to index states in the CartPole Domain.

    This class enumerates the different indices used when indexing the state. \n
    e.g. s[StateIndex.THETA] is guaranteed to return the angle state.

    """
    THETA, THETA_DOT = 0,1
    X, X_DOT = 2,3
    FORCE = 4


class CartPole_InvertedBalance(CartPole):
    """
    *OBJECTIVE*
    Reward 1 is received on each timestep spent within the goal region,
    zero elsewhere.
    This is also the terminal condition.

    RL Community has the following bounds for failure:
    theta: [-12, 12] degrees  -->  [-pi/15, pi/15]
    x: [-2.4, 2.4] meters

    Pendulum starts straight up, theta = 0, with the
    cart at x = 0.
    (see CartPole parent class for coordinate definitions).

    Note that the reduction in angle limits affects the resolution
    of the discretization of the continuous space.

    Note that if unbounded x is desired, set x (and/or xdot) limits to
    be `[float("-inf"), float("inf")]`

    .. note::
        The reward scheme is different from that in `Pendulum_InvertedBalance`

    (See CartPole implementation in the RL Community,
    http://library.rl-community.org/wiki/CartPole)
    """
    #: Limit on theta (Note that this may affect your representation's discretization)
    ANGLE_LIMITS        = [-pi/15.0, pi/15.0]
    #: Based on Parr 2003 and ICML 11 (RL-Matlab)
    ANGULAR_RATE_LIMITS = [-2.0, 2.0]
    gamma               = .999

    def __init__(self, logger = None):
        self.statespace_limits  = np.array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        self.state_space_dims = len(self.statespace_limits)
        super(CartPole_InvertedBalance,self).__init__(logger)

    def s0(self):
        # Returns the initial state, pendulum vertical
        self.state = np.zeros(4)
        return self.state.copy(), self.isTerminal(), self.possibleActions()


    def _getReward(self, a, s=None):
        # On this domain, reward of 1 is given for each step spent within goal region.
        # There is no specific penalty for failure.
        if s is None:
            s = self.state
        return self.GOAL_REWARD if -pi/15 < s[StateIndex.THETA] < pi/15 else 0

    def isTerminal(self):
        s = self.state
        return (not (-pi/15 < s[StateIndex.THETA] < pi/15) or \
                not (-2.4    < s[StateIndex.X]     < 2.4))


class CartPoleBalanceOriginal(CartPole):
    """
    Domain and constants per
    `original definition and code <http://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c>`_,
    see reference below.
    
    **Reference**\n
    Sutton, Richard S., and Andrew G. Barto:
    Reinforcement learning: An introduction.
    Cambridge: MIT press, 1998.
    """
    __author__ = "Christoph Dann"

    ANGLE_LIMITS        = [-np.pi/15.0, np.pi/15.0]
    ANGULAR_RATE_LIMITS = [-2.0, 2.0]
    gamma               = .95
    AVAIL_FORCE         = np.array([-10, 10])
    int_type = "euler"
    num_euler_steps = 1

    def __init__(self, logger=None, good_reward=0.):
        self.good_reward = good_reward
        self.statespace_limits  = np.array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        self.state_space_dims = len(self.statespace_limits)
        super(CartPoleBalanceOriginal, self).__init__(logger)

    def s0(self):
        self.state = np.zeros(4)
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        return self.good_reward if not self.isTerminal(s=s) else -1.

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return (not (-np.pi/15 < s[StateIndex.THETA] < np.pi/15) or
                not (-2.4    < s[StateIndex.X]     < 2.4))


class CartPoleBalanceModern(CartPole):
    """
    A more realistic version of balancing with 3 actions (left, right, none)
    and some uniform noise
    """
    
    __author__ = "Christoph Dann"
    
    int_type = 'rk4'
    AVAIL_FORCE = np.array([-10., 0., 10.])
    force_noise_max = 1.
    gamma = .95
    int_type = "euler"
    num_euler_steps = 1
    ANGLE_LIMITS        = [-np.pi/15.0, np.pi/15.0]
    ANGULAR_RATE_LIMITS = [-2.0, 2.0]

    def __init__(self, logger=None):
        self.statespace_limits  = np.array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        self.state_space_dims = len(self.statespace_limits)
        super(CartPoleBalanceModern, self).__init__(logger)

    def s0(self):
        self.state = np.array([self.random_state.randn()*0.01, 0., 0., 0.])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        return 0. if not self.isTerminal(s=s) else -1.

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return (not (-np.pi/15 < s[StateIndex.THETA] < np.pi/15) or
                not (-2.4    < s[StateIndex.X]     < 2.4))


class CartPole_SwingUp(CartPole):
    """
    *OBJECTIVE*
    Reward is 1 within the goal region, 0 elsewhere.
    The episode terminates if x leaves its bounds, [-2.4, 2.4]

    Pendulum starts straight down, theta = pi, with the
    cart at x = 0.
    (see CartPole parent class for coordinate definitions).

    The objective is to get and then keep the pendulum in the goal
    region for as long as possible, with +1 reward for
    each step in which this condition is met; the expected
    optimum then is to swing the pendulum vertically and
    hold it there, collapsing the problem to Pendulum_InvertedBalance
    but with much tighter bounds on the goal region.

    (See `CartPole <http://library.rl-community.org/wiki/CartPole>`_
    implementation in the RL Community)

    """
    ANGLE_LIMITS        = [-pi, pi]     #: Limit on theta (used for discretization)
    def __init__(self, logger = None):
        self.statespace_limits  = np.array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS, self.POSITON_LIMITS, self.VELOCITY_LIMITS])
        super(CartPole_SwingUp,self).__init__(logger)

    def s0(self):
        # Returns the initial state, pendulum vertical
        self.state = np.array([pi,0,0,0])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        return self.GOAL_REWARD if -pi/6 < s[StateIndex.THETA] < pi/6 else 0

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return not (-2.4 < s[StateIndex.X] < 2.4)


class CartPole_SwingUpFriction(CartPole_SwingUp):
    """
    Modifies ``CartPole`` dynamics to include friction.
    
    """
    #: Limits on pendulum rate [per RL Community CartPole]
    ANGULAR_RATE_LIMITS = [-3.0, 3.0]
    #: Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1
    #: m - Limits on cart position [Per RL Community CartPole]
    POSITON_LIMITS      = [-2.4, 2.4]
    #: m/s - Limits on cart velocity [per RL Community CartPole]
    VELOCITY_LIMITS     = [-3.0, 3.0]

    MASS_CART = 0.5 #: kilograms, kg - Mass of cart
    MASS_PEND = 0.5 #: kilograms, kg - Mass of the pendulum arm
    LENGTH = 0.6 #: meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    A = .5
    DT = 0.1 #: seconds, s - Time between steps
    episodeCap          = 400
    gamma = 0.95
    B = 0.1 # Friction between cart and ground

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        if not (-2.4 < s[StateIndex.X] < 2.4):
            return -30
        pen_pos = np.array([s[StateIndex.X] + self.LENGTH * np.sin(s[StateIndex.THETA]),
                         self.LENGTH * np.cos(s[StateIndex.THETA])])
        diff = pen_pos - np.array([0, self.LENGTH])
        #diff[1] *= 1.5
        return np.exp(-.5* sum(diff**2) * self.A) - .5

    def _dsdt(self, s_aug, t):
        s = np.zeros((4))
        s[0] = s_aug[StateIndex.X]
        s[1] = s_aug[StateIndex.X_DOT]
        s[3] = pi - s_aug[StateIndex.THETA]
        s[2] = - s_aug[StateIndex.THETA_DOT]
        a = s_aug[4]
        ds = self._ode(s, t, a, self.MASS_PEND, self.LENGTH, self.MASS_CART, self.B)
        ds_aug = s_aug.copy()
        ds_aug[StateIndex.X] = ds[0]
        ds_aug[StateIndex.X_DOT] = ds[1]
        ds_aug[StateIndex.THETA_DOT] = ds[2]
        ds_aug[StateIndex.THETA] = ds[3]
        return ds_aug

    def _ode(self, s, t, a, m, l, M, b):
        """
        [x, dx, dtheta, theta]
        """

        #cdef double g, c3, s3
        s3 = np.sin(s[3])
        c3 = np.cos(s[3])
        g = self.ACCEL_G
        ds = np.zeros(4)
        ds[0] = s[1]
        ds[1] = (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a - 4 * b * s[1])\
            / (4 * (M + m) - 3 * m * c3 ** 2)
        ds[2] = (-3 * m * l * s[2] ** 2 * s3*c3 - 6 * (M + m) * g * s3 - 6 * (a - b * s[1]) * c3)\
            / (4 * l * (m + M) - 3 * m * l * c3 ** 2)
        ds[3] = s[2]
        return ds
