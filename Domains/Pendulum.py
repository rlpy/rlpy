"""Pendulum base domain."""

from Tools import *
from Domain import *

from scipy import integrate # For integrate.odeint (accurate, slow)

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Robert H. Klein"]


class InfTrackCartPole(CartPoleBase):
    """
    Infinite Track Cart Pole.\n
    Inherits dynamics from ``CartPoleBase`` and utilizes two states - the
    angular position and velocity of the pendulum on the cart.
    This domain is effectively a CartPole moving on an infinite rail.\n
    An identical physical implementation consists of a pendulum mounted instead
    on a rotating rod. (if the length of the rod to pendulum base is assumed
    to be 1.0m, then force actions on the cart (Newtons) correspond directly
    to torques on the rod (N*m).)\n
    See `Tor Aamodt's B.A.Sc. Thesis <http://www.eecg.utoronto.ca/~aamodt/BAScThesis/>`_
    for a visualization.
    
    .. warning::
        In the literature, this domain is often referred as a ``Pendulum`` and
        is often accompanied by misleading or even incorrect diagrams.
        However, the cited underlying dynamics correspond to a pendulum on a
        cart (\c %CartPoleBase), with cart position and velocity ignored and
        unbounded (but **still impacting the pendulum dynamics**).\n
        Actions thus still take the form of forces and not torques.
    
    | **State**
    theta    = Angular position of pendulum
    (relative to straight up at 0 rad), positive clockwise. \n
    thetaDot = Angular rate of pendulum. positive clockwise. \n

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
        # Limits of each dimension of the state space.
        # Each row corresponds to one dimension and has two elements [min, max]
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        self.continuous_dims    = [StateIndex.THETA, StateIndex.THETA_DOT]
        self.DimNames       = ['Theta','Thetadot']
        
        # Transpose the matrices shown above, if they have not been already by another instance of the domain.
        if len(self._beta) == 5: self._beta = (self._beta).conj().transpose()
        if len(self._gamma) == 2: self._gamma = (self._gamma).conj().transpose()
        super(InfTrackCartPole,self).__init__(logger)

    def s0(self):
        # Defined by children
        raise NotImplementedError

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
        
        
        
        
    
    
    
        
#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
#
# ---OBJECTIVE--- \n
# Reward is 0 everywhere, -1 if theta falls below horizontal. \n
# This is also the terminal condition for an episode. \n
#
# theta:    [-pi/2, pi/2] \n
# thetaDot: [-2,2]    Limits imposed per L & P 2003 \n
#
# Pendulum starts unmoving, straight up, theta = 0.
# (see Pendulum parent class for coordinate definitions). \n
#
# (See 1Link implementation by Lagoudakis & Parr, 2003)
#####################################################################

class InfCartInvertedBalance(Pendulum):
    """
    Domain constants per 1Link implementation by Lagoudakis & Parr, 2003.
    
    .. warning::
        
        L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
        saturates them frequently when falling; more realistic to use 2*pi.
        
    
    """
    ## Newtons, N - Torque values available as actions
    AVAIL_FORCE         = array([-50,0,50])
    ## kilograms, kg - Mass of the pendulum arm
    MASS_PEND           = 2.0
    ## kilograms, kg - Mass of cart
    MASS_CART           = 8.0
    ## meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH              = 1.0
    ## m/s^2 - gravitational constant
    ACCEL_G             = 9.8
    ## Time between steps
    dt                  = 0.1
    ## Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max     = 10

    ## Reward received when the pendulum falls below the horizontal
    FELL_REWARD         = -1
    ## Limit on theta (Note that this may affect your representation's discretization)
    ANGLE_LIMITS        = [-pi/2.0, pi/2.0]
    ## Limits on pendulum rate, per 1Link of Lagoudakis & Parr
    ANGULAR_RATE_LIMITS = [-2., 2.] # NOTE that L+P's rate limits [-2,2] are actually unphysically slow, and the pendulum
                                # saturates them frequently when falling; more realistic to use 2*pi.
    ## Max number of steps per trajectory (default 3000)
    episodeCap          = None
    ## Based on Parr 2003 and ICML 11 (RL-Matlab)
    gamma               = .95

    # For Visual Stuff
    MAX_RETURN = 0
    MIN_RETURN = -1

    def __init__(self, episodeCap = 3000, logger = None):
        self.episodeCap = episodeCap
        super(Pendulum_InvertedBalance,self).__init__(logger)

    def s0(self):
        # Returns the initial state, pendulum vertical
        # Initial state is uniformly random between [-.2,.2] for both dimensions
        self.state = (self.random_state.rand(2)*2-1)*0.2
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a):
        # Return the reward earned for this state-action pair
        # On this domain, reward of -1 is given for failure, |angle| exceeding pi/2
        return self.FELL_REWARD if self.isTerminal() else 0

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        return not (-pi/2.0 < s[StateIndex.THETA] < pi/2.0) # per L & P 2003



#####################################################################
# \author Robert H. Klein, Alborz Geramifard at MIT, Nov. 30 2012
#####################################################################
# ---OBJECTIVE--- \n
# Reward is 1 within the goal region, 0 elsewhere. \n
# There is no terminal condition aside from episodeCap. \n
#
# Pendulum starts straight down, theta = pi
# (see Pendulum parent class for coordinate definitions). \n
#
# The objective is to get and then keep the pendulum in the goal
# region for as long as possible, with +1 reward for
# each step in which this condition is met; the expected
# optimum then is to swing the pendulum vertically and
# hold it there, collapsing the problem to Pendulum_InvertedBalance
# but with much tighter bounds on the goal region. \n
#
# (See 1Link implementation by Lagoudakis & Parr, 2003)
#
#####################################################################

class Pendulum_SwingUp(Pendulum):

    # Domain constants [temporary, per Lagoudakis & Parr 2003, for InvertedBalance Task]
    ## Newtons, N - Torque values available as actions
    AVAIL_FORCE         = array([-50,0,50])
    ## kilograms, kg - Mass of the pendulum arm
    MASS_PEND           = 2.0
    ## kilograms, kg - Mass of cart
    MASS_CART           = 8.0
    ## meters, m - Physical length of the pendulum, meters (note the moment-arm lies at half this distance)
    LENGTH              = 1.0
    ## m/s^2 - gravitational constant
    ACCEL_G             = 9.8
    ## Time between steps
    dt                  = 0.1
    ## Newtons, N - Maximum noise possible, uniformly distributed
    force_noise_max     = 10

    ## Reward received on each step the pendulum is in the goal region
    GOAL_REWARD         = 1
    ## Goal region for reward [temporary values]
    GOAL_LIMITS         = [-pi/6, pi/6]
    ## Limit on theta
    ANGLE_LIMITS        = [-pi, pi]
    ## Limits on pendulum rate [temporary values, copied from InvertedBalance task of RL Community]
    ANGULAR_RATE_LIMITS = [-3*pi, 3*pi]
    ## Max number of steps per trajectory
    episodeCap          = 300
    # For Visual Stuff
    MAX_RETURN = episodeCap*GOAL_REWARD
    MIN_RETURN = 0
    gamma               = .9   #

    def __init__(self, logger = None):
        self.statespace_limits  = array([self.ANGLE_LIMITS, self.ANGULAR_RATE_LIMITS])
        super(Pendulum_SwingUp,self).__init__(logger)

    def s0(self):
        #Returns the initial state: pendulum straight up and unmoving.
        self.state = array([pi,0])
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def _getReward(self, a):
        s  = self.state
        """Return the reward earned for this state-action pair.
        On this domain, reward of 1 is given for success, which occurs when |theta| < pi/6"""
        return self.GOAL_REWARD if self.GOAL_LIMITS[0] < s[StateIndex.THETA] < self.GOAL_LIMITS[1] else 0

    def isTerminal(self, s=None):
        return False
