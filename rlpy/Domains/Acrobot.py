"""classic Acrobot task"""
from rlpy.Tools import wrap, bound, lines, fromAtoB, rk4
from .Domain import Domain
import numpy as np
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"


class Acrobot(Domain):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated
    Intitially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.

    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.

    **STATE:**
    The state consists of the two rotational joint angles and their velocities
    [theta1 theta2 thetaDot1 thetaDot2]. An angle of 0 corresponds to corresponds
    to the respective link pointing downwards (angles are in world coordinates).

    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.

    .. note::

        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondance
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.

        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'

    **REFERENCE:**

    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)

    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.

    .. warning::

        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    episodeCap = 1000
    dt = .2
    continuous_dims = np.arange(4)
    discount_factor = 1.

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.
    statespace_limits = np.array([[-np.pi, np.pi]] * 2
                                 + [[-MAX_VEL_1, MAX_VEL_1]]
                                 + [[-MAX_VEL_2, MAX_VEL_2]])

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def s0(self):
        self.state = np.zeros((4))
        return np.zeros((4)), self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        s = self.state
        return -np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.random_state.uniform(-
                                                self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns.copy()
        terminal = self.isTerminal()
        reward = -1. if not terminal else 0.
        return reward, ns, terminal, self.possibleActions()

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def showDomain(self, a=0):
        """
        Plot the 2 links + action arrows
        """
        s = self.state
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.gcf()
            self.domain_ax = self.domain_fig.add_axes(
                [0, 0, 1, 1], frameon=True, aspect=1.)
            ax = self.domain_ax
            self.link1 = lines.Line2D([], [], linewidth=2, color='black')
            self.link2 = lines.Line2D([], [], linewidth=2, color='blue')
            ax.add_line(self.link1)
            ax.add_line(self.link2)

            # Allow room for pendulum to swing without getting cut off on graph
            viewable_distance = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.5
            ax.set_xlim(-viewable_distance, +viewable_distance)
            ax.set_ylim(-viewable_distance, viewable_distance)
            # add bar
            bar = lines.Line2D([-viewable_distance, viewable_distance],
                               [self.LINK_LENGTH_1, self.LINK_LENGTH_1],
                               linewidth=1, color='red')
            ax.add_line(bar)
            # ax.set_aspect('equal')

            plt.show()

        if self.action_arrow is not None:
            self.action_arrow.remove()
            self.action_arrow = None

        torque = self.AVAIL_TORQUE[a]
        SHIFT = .5
        if torque > 0:  # counterclockwise torque
            self.action_arrow = fromAtoB(SHIFT / 2.0, .5 * SHIFT, -SHIFT / 2.0,
                                         -.5 * SHIFT, 'k', connectionstyle="arc3,rad=+1.2",
                                         ax=self.domain_ax)
        elif torque < 0:  # clockwise torque
            self.action_arrow = fromAtoB(
                -SHIFT / 2.0, .5 * SHIFT, +SHIFT / 2.0,
                -.5 * SHIFT, 'r', connectionstyle="arc3,rad=-1.2",
                ax=self.domain_ax)

        # update pendulum arm on figure
        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        self.link1.set_data([0., p1[1]], [0., p1[0]])
        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]
        self.link2.set_data([p1[1], p2[1]], [p1[0], p2[0]])
        plt.draw()


class AcrobotLegacy(Acrobot):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector to a height at least the length of one link above the base.

    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.

    **STATE:**
    The state consists of the two rotational joint angles and their velocities
    [theta1 theta2 thetaDot1 thetaDot2]. An angle of 0 corresponds to
    the respective link pointing downwards (angles are in world coordinates).

    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.

    .. note::

        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondance
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.

        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'

    **REFERENCE:**

    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)

    .. seealso::
        Sutton, Richard S., and Andrew G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.

    """

    book_or_nips = "book"

    def step(self, a):

        torque = self.AVAIL_TORQUE[a]
        s = self.state

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.random_state.uniform(-
                                                self.torque_noise_max, self.torque_noise_max)

        s_augmented = np.append(s, torque)
        for i in range(4):
            s_dot = np.array(self._dsdt(s_augmented, 0))
            s_augmented += s_dot * self.dt / 4.

            # make sure that we don't have 2 free pendulums but a "gymnast"
            # for k in range(2):
            #    if np.abs(s_augmented[k]) > np.pi:
            #        s_augmented[k] = np.sign(s_augmented[k]) * np.pi
            #        s_augmented[k + 2] = 0.
            s_augmented[0] = wrap(s_augmented[0], -np.pi, np.pi)
            s_augmented[1] = wrap(s_augmented[1], -np.pi, np.pi)
            s_augmented[2] = bound(
                s_augmented[2],
                -self.MAX_VEL_1,
                self.MAX_VEL_1)
            s_augmented[3] = bound(
                s_augmented[3],
                -self.MAX_VEL_2,
                self.MAX_VEL_2)

        ns = s_augmented[:4]  # omit action
        self.state = ns.copy()
        terminal = self.isTerminal()
        reward = -1. if not terminal else 0.
        return reward, ns, terminal, self.possibleActions()
