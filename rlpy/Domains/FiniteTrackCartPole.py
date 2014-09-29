"""Cart with a pole domains"""

from .Domain import Domain
from .CartPoleBase import CartPoleBase, StateIndex
import numpy as np
import scipy.integrate
from rlpy.Tools import pl, mpatches, mpath, fromAtoB, lines, rk4, wrap, bound, colors, plt

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
pi = np.pi


class FiniteTrackCartPole(CartPoleBase):

    """
    Finite Track Cart Pole.\n
    Inherits dynamics from ``CartPoleBase`` and utilizes four states - angular
    quantities of pendulum (position and velocity) and lateral quantities of
    the cart.
    Not only does this increase the state space relative to
    \c %InfTrackCartPole, but the cart must remain in a finite interval
    corresponding to a physical rail, which affects valid solutions/policies.


    **State** \n
    theta    = Angular position of pendulum
    (relative to straight up at 0 rad), positive clockwise. \n
    thetaDot = Angular rate of pendulum. \n
    x        = Linear position of the cart on its track (positive right). \n
    xDot     = Linear velocity of the cart on its track.

    **Actions** \n
    Actions take the form of force applied to cart; \n
    Positive force acts to the right on the cart. \n

    Note the domain defaults in \c %CartPoleBase.

    .. warning::

        For \"Swing-Up\" tasks where the goal is to swing the pendulum from
        rest to vertical, Lagoudakis, Parr, and Bartlett's default [-2, 2] rad/s
        is unphysically slow; the Pendulum often saturates it.\n
        RLPy will issue truncation warnings if this is occurring.

    **Reference** \n
    For details, see:
        Michail G. Lagoudakis, Ronald Parr, and L. Bartlett
        Least-squares policy iteration.  Journal of Machine Learning Research
        (2003) Issue 4.

    .. note::

        For full domain description, see: \n
        Wang, H., Tanaka, K., and Griffin, M. An approach to fuzzy control
        of nonlinear systems; Stability and design issues.
        IEEE Trans. on Fuzzy Systems, 4(1):14-23, 1996.

    """

    __author__ = ["Robert H. Klein"]

    #: Default limits on theta
    ANGLE_LIMITS = [-pi / 15.0, pi / 15.0]
    #: Default limits on pendulum rate
    ANGULAR_RATE_LIMITS = [-2.0, 2.0]
    #: m - Default limits on cart position [Per RL Community CartPole]
    POSITION_LIMITS = [-2.4, 2.4]
    #: m/s - Default limits on cart velocity [per RL Community CartPole]
    VELOCITY_LIMITS = [-6.0, 6.0]

     #: Newtons, N - Force values available as actions
    AVAIL_FORCE = np.array([-10, 10])

    #: kilograms, kg - Mass of the pendulum arm
    MASS_PEND = 0.1
    #: kilograms, kg - Mass of cart
    MASS_CART = 1.0
    #: meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH = 1.0
    # m - Length of moment-arm to center of mass (= half the pendulum length)
    MOMENT_ARM = LENGTH / 2.
    # 1/kg - Used in dynamics computations, equal to 1 / (MASS_PEND +
    # MASS_CART)
    _ALPHA_MASS = 1.0 / (MASS_CART + MASS_PEND)
    #: seconds, s - Time between steps
    dt = 0.02
    #: Newtons, N - Maximum noise possible, uniformly distributed.  Default 0.
    force_noise_max = 0.

    def __init__(self):
        # Limits of each dimension of the state space.
        # Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits = np.array(
            [self.ANGLE_LIMITS,
             self.ANGULAR_RATE_LIMITS,
             self.POSITION_LIMITS,
             self.VELOCITY_LIMITS])
        self.continuous_dims = [
            StateIndex.THETA,
            StateIndex.THETA_DOT,
            StateIndex.X,
            StateIndex.X_DOT]
        self.DimNames = ['Theta', 'Thetadot', 'X', 'Xdot']
        super(FiniteTrackCartPole, self).__init__()

    def step(self, a):
        s = self.state
        ns = self._stepFourState(s, a)
        self.state = ns.copy()
        terminal = self.isTerminal()  # automatically uses self.state
        reward = self._getReward(a)  # Automatically uses self.state
        possibleActions = self.possibleActions()
        return reward, ns, terminal, possibleActions

    def s0(self):
        # defined by children
        raise NotImplementedError

    def showLearning(self, representation):
        """

        ``xSlice`` and ``xDotSlice`` - the value of ``x`` and ``xDot``
        respectively, associated with the plotted value function and policy
        (which are each 2-D grids across ``theta`` and ``thetaDot``).

        """
        xSlice = 0.  # value of x assumed when plotting V and pi
        xDotSlice = 0.  # value of xDot assumed when plotting V and pi

        warnStr = "WARNING: showLearning() called with 4-state "\
            "cartpole; only showing slice at (x, xDot) = (%.2f, %.2f)" % (
                xSlice,
                xDotSlice)

        (thetas, theta_dots) = self._setup_learning(representation)

        pi = np.zeros((len(theta_dots), len(thetas)), 'uint8')
        V = np.zeros((len(theta_dots), len(thetas)))

        for row, thetaDot in enumerate(theta_dots):
            for col, theta in enumerate(thetas):
                s = np.array([theta, thetaDot, xSlice, xDotSlice])
                terminal = self.isTerminal(s)
                # Array of Q-function evaluated at all possible actions at
                # state s
                Qs = representation.Qs(s, terminal)
                # Array of all possible actions at state s
                As = self.possibleActions(s=s)
                # If multiple optimal actions, pick one randomly
                a = np.random.choice(As[Qs.max() == Qs])
                # Assign pi to be an optimal action (which maximizes
                # Q-function)
                pi[row, col] = a
                # Assign V to be the value of the Q-function under optimal
                # action
                V[row, col] = max(Qs)

        self._plot_policy(pi)
        plt.title("Policy (Slice at x=0, xDot=0)")
        self._plot_valfun(V)
        plt.title("Value Function (Slice at x=0, xDot=0)")

        pl.draw()

    def showDomain(self, a=0):
        """
        Display the 4-d state of the cartpole and arrow indicating current
        force action (not including noise!).

        """
        fourState = self.state
        self._plot_state(fourState, a)


class FiniteCartPoleBalance(FiniteTrackCartPole):

    """
    **Goal** \n
    Reward 1 is received on each timestep spent within the goal region,
    zero elsewhere.
    This is also the terminal condition.

    The bounds for failure match those used in the RL-Community (see Reference) \n
    ``theta``: [-12, 12] degrees  -->  [-pi/15, pi/15] \n
    ``x``: [-2.4, 2.4] meters

    Pendulum starts straight up, ``theta = 0``, with the
    cart at ``x = 0``.

    **Reference** \n
        See `RL-Library CartPole <http://library.rl-community.org/wiki/CartPole>`_ \n
        Domain constants per RL Community / RL-Library
        `CartPole implementation <http://code.google.com/p/rl-library/wiki/CartpoleJava>`_

    """
    #: Discount factor
    discount_factor = .999

    def __init__(self):
        super(FiniteCartPoleBalance, self).__init__()

    def s0(self):
        # Returns the initial state, pendulum vertical
        self.state = np.zeros(4)
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        # On this domain, reward of 1 is given for each step spent within goal region.
        # There is no specific penalty for failure.
        if s is None:
            s = self.state
        return (
            self.GOAL_REWARD if -pi / 15 < s[StateIndex.THETA] < pi / 15 else 0
        )

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return (not (-pi / 15 < s[StateIndex.THETA] < pi / 15) or
                not (-2.4 < s[StateIndex.X] < 2.4))


class FiniteCartPoleBalanceOriginal(FiniteTrackCartPole):

    """
    **Reference** \n
    Sutton, Richard S., and Andrew G. Barto:
    Reinforcement learning: An introduction.
    Cambridge: MIT press, 1998.

    See :class:`Domains.FiniteTrackCartPole.FiniteCartPoleBalance` \n

    .. note::

        `original definition and code <http://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c>`_

    """
    __author__ = "Christoph Dann"

    def __init__(self, good_reward=0.):
        self.good_reward = good_reward
        super(FiniteCartPoleBalanceOriginal, self).__init__()

    def s0(self):
        self.state = np.zeros(4)
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        return self.good_reward if not self.isTerminal(s=s) else -1.

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return (not (-np.pi / 15 < s[StateIndex.THETA] < np.pi / 15) or
                not (-2.4 < s[StateIndex.X] < 2.4))


class FiniteCartPoleBalanceModern(FiniteTrackCartPole):

    """
    A more realistic version of balancing with 3 actions (left, right, none)
    instead of the default (left, right), and nonzero, uniform noise in actions.\n
    See :class:`Domains.FiniteTrackCartPole.FiniteCartPoleBalance`.\n

    Note that the start state has some noise.

    """

    __author__ = "Christoph Dann"

    #: Newtons, N - Force values available as actions (Note we add a 0-force action)
    AVAIL_FORCE = np.array([-10., 0., 10.])
    #: Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max = 1.

    def __init__(self):
        super(FiniteCartPoleBalanceModern, self).__init__()

    def s0(self):
        self.state = np.array([self.random_state.randn() * 0.01, 0., 0., 0.])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        return 0. if not self.isTerminal(s=s) else -1.

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return (not (-np.pi / 15 < s[StateIndex.THETA] < np.pi / 15) or
                not (-2.4 < s[StateIndex.X] < 2.4))


class FiniteCartPoleSwingUp(FiniteTrackCartPole):

    """
    **Goal** \n
    Reward is 1 within the goal region, 0 elsewhere. \n

    Pendulum starts straight down, theta = pi, with the
    cart at x = 0.

    The objective is to get and then keep the pendulum in the goal
    region for as long as possible, with +1 reward for
    each step in which this condition is met; the expected
    optimum then is to swing the pendulum vertically and
    hold it there, collapsing the problem to InfCartPoleBalance
    but with much tighter bounds on the goal region.

    See parent class :class:`Domains.FiniteTrackCartPole.FiniteTrackCartPole` for more information.

    """
    #: Limit on pendulum angle (no termination, pendulum can make full cycle)
    ANGLE_LIMITS = [-pi, pi]

    # NOTE that L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
    # saturates them frequently when falling; more realistic to use 2*pi.

    def __init__(self):
        super(FiniteCartPoleSwingUp, self).__init__()

    def s0(self):
        # Returns the initial state, pendulum vertical
        self.state = np.array([pi, 0, 0, 0])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        return (
            self.GOAL_REWARD if -pi / 6 < s[StateIndex.THETA] < pi / 6 else 0
        )

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return not (-2.4 < s[StateIndex.X] < 2.4)


class FiniteCartPoleSwingUpFriction(FiniteCartPoleSwingUp):

    """
    Modifies ``CartPole`` dynamics to include friction. \n
    This domain is a child of :class:`Domains.FiniteTrackCartPole.FiniteCartPoleSwingUp`.

    """

    __author__ = "Christoph Dann"

    # TODO - needs reference

    #: Limit on pendulum angle (no termination, pendulum can make full cycle)
    ANGLE_LIMITS = [-pi, pi]
    #: Limits on pendulum rate
    ANGULAR_RATE_LIMITS = [-3.0, 3.0]
    #: m - Limits on cart position
    POSITION_LIMITS = [-2.4, 2.4]
    #: m/s - Limits on cart velocity
    VELOCITY_LIMITS = [-3.0, 3.0]

    MASS_CART = 0.5  # : kilograms, kg - Mass of cart
    MASS_PEND = 0.5  # : kilograms, kg - Mass of the pendulum arm
    #: meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH = 0.6
    # a friction coefficient
    A = .5
    #: seconds, s - Time between steps
    dt = 0.10
    #: Max number of steps per trajectory (reduced from default of 3000)
    episodeCap = 400
    # Friction coefficient between cart and ground
    B = 0.1

    def __init__(self):
        super(FiniteCartPoleSwingUpFriction, self).__init__()

    def _getReward(self, a, s=None):
        if s is None:
            s = self.state
        if not (self.POSITION_LIMITS[0] < s[StateIndex.X] < self.POSITION_LIMITS[1]):
            return -30
        pen_pos = np.array(
            [s[StateIndex.X] + self.LENGTH * np.sin(s[StateIndex.THETA]),
             self.LENGTH * np.cos(s[StateIndex.THETA])])
        diff = pen_pos - np.array([0, self.LENGTH])
        #diff[1] *= 1.5
        return np.exp(-.5 * sum(diff ** 2) * self.A) - .5

    def _dsdt(self, s_aug, t):
        s = np.zeros((4))
        s[0] = s_aug[StateIndex.X]
        s[1] = s_aug[StateIndex.X_DOT]
        s[3] = pi - s_aug[StateIndex.THETA]
        s[2] = - s_aug[StateIndex.THETA_DOT]
        a = s_aug[4]
        ds = self._ode(
            s,
            t,
            a,
            self.MASS_PEND,
            self.LENGTH,
            self.MASS_CART,
            self.B)
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

        # cdef double g, c3, s3
        s3 = np.sin(s[3])
        c3 = np.cos(s[3])
        g = self.ACCEL_G
        ds = np.zeros(4)
        ds[0] = s[1]
        ds[1] = (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a - 4 * b * s[1])\
            / (4 * (M + m) - 3 * m * c3 ** 2)
        ds[2] = (-3 * m * l * s[2] ** 2 * s3 * c3 - 6 * (M + m) * g * s3 - 6 * (a - b * s[1]) * c3)\
            / (4 * l * (m + M) - 3 * m * l * c3 ** 2)
        ds[3] = s[2]
        return ds
