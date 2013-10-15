"""Cart with a pole domains"""

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

class FiniteTrackCartPole(CartPoleBase):
    """
    Finite Track Cart Pole.\n
    Inherits dynamics from ``CartPoleBase`` and utilizes four states - angular
    quantities of pendulum (position and velocity) and lateral quantities of cart.
    Not only does this increase the state space relative to
    ``InfTrackCartPole``, but the cart must remain in a finite interval
    corresponding to a physical rail, which affects valid solutions.
    
    
    | **State**
    theta    = Angular position of pendulum
    (relative to straight up at 0 rad), positive clockwise. \n
    thetaDot = Angular rate of pendulum. positive clockwise. \n
    x        = Linear position of the cart on its track (positive right). \n
    xDot     = Linear velocity of the cart on its track.

    | **Actions**
    Actions take the form of force applied to cart; \n
    Positive force acts to the right on the cart. \n

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
    be ``[float("-inf"), float("inf")]``

    .. note::
        The reward scheme is different from that in ``Pendulum_InvertedBalance``

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
