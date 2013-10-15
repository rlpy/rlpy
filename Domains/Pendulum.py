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
        cart (``CartPoleBase``), with cart position and velocity ignored and
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

class Pendulum(Domain):
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

    def s0(self):
        # Defined by children
        raise NotImplementedError

    def possibleActions(self, s=None): # Return list of all indices corresponding to actions available
        return arange(self.actions_num)



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
