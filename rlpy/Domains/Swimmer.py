"""multi-link swimmer moving in a fluid."""

from .Domain import Domain
import numpy as np
from rlpy.Tools import plt, rk4, cartesian, colors
from rlpy.Tools import matplotlib as mpl

from rlpy.Policies.SwimmerPolicy import SwimmerPolicy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class Swimmer(Domain):

    """
    A swimmer consisting of a chain of d links connected by rotational joints.
    Each joint is actuated. The goal is to move the swimmer to a specified goal
    position.

    *States*:
        | 2 dimensions: position of nose relative to goal
        | d -1 dimensions: angles
        | 2 dimensions: velocity of the nose
        | d dimensions: angular velocities

    *Actions*:
        each joint torque is discretized in 3 values: -2, 0, 2

    .. note::
        adapted from Yuval Tassas swimmer implementation in Matlab available at
        http://www.cs.washington.edu/people/postdocs/tassa/code/

    .. seealso::
        Tassa, Y., Erez, T., & Smart, B. (2007).
        *Receding Horizon Differential Dynamic Programming.*
        In Advances in Neural Information Processing Systems.
    """
    dt = 0.03
    episodeCap = 1000
    discount_factor = 0.98

    def __init__(self, d=3, k1=7.5, k2=0.3):
        """
        d:
            number of joints
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.nose = 0
        self.masses = np.ones(d)
        self.lengths = np.ones(d)
        self.inertia = self.masses * self.lengths * self.lengths / 12.
        self.goal = np.zeros(2)

        # reward function parameters
        self.cu = 0.04
        self.cx = 2.

        Q = np.eye(self.d, k=1) - np.eye(self.d)
        Q[-1, :] = self.masses
        A = np.eye(self.d, k=1) + np.eye(self.d)
        A[-1, -1] = 0.
        self.P = np.dot(np.linalg.inv(Q), A * self.lengths[None, :]) / 2.

        self.U = np.eye(self.d) - np.eye(self.d, k=-1)
        self.U = self.U[:, :-1]
        self.G = np.dot(self.P.T * self.masses[None, :], self.P)

        # incidator variables for angles in a state representation
        self.angles = np.zeros(2 + self.d * 2 + 1, dtype=np.bool)
        self.angles[2:2 + self.d - 1] = True
        self.angles[-self.d - 2:] = True

        self.actions = cartesian((d - 1) * [[-2., 0., 2]])
        self.actions_num = len(self.actions)

        self.statespace_limits = [[-15, 15]] * 2 + [[-np.pi, np.pi]] * (d - 1) \
            + [[-2, 2]] * 2 + [[-np.pi * 2, np.pi * 2]] * d
        self.statespace_limits = np.array(self.statespace_limits)
        self.continuous_dims = range(self.statespace_limits.shape[0])
        super(Swimmer, self).__init__()

    def s0(self):
        self.theta = np.zeros(self.d)
        self.pos_cm = np.array([10, 0])
        self.v_cm = np.zeros(2)
        self.dtheta = np.zeros(self.d)
        return self.state, self.isTerminal(), self.possibleActions()

    @property
    def state(self):
        return np.hstack(self._body_coord())

    def isTerminal(self):
        return False

    def possibleActions(self, s=None):
        return np.arange(self.actions_num)

    def showDomain(self, a=None):
        if a is not None:
            a = self.actions[a]
        T = np.empty((self.d, 2))
        T[:, 0] = np.cos(self.theta)
        T[:, 1] = np.sin(self.theta)
        R = np.dot(self.P, T)
        R1 = R - .5 * self.lengths[:, None] * T
        R2 = R + .5 * self.lengths[:, None] * T
        Rx = np.hstack([R1[:, 0], R2[:, 0]]) + self.pos_cm[0]
        Ry = np.hstack([R1[:, 1], R2[:, 1]]) + self.pos_cm[1]
        print Rx
        print Ry
        f = plt.figure("Swimmer Domain")
        if not hasattr(self, "swimmer_lines"):
            plt.plot(0., 0., "ro")
            self.swimmer_lines = plt.plot(Rx, Ry)[0]
            self.action_text = plt.text(-2, -8, str(a))
            plt.xlim(-5, 15)
            plt.ylim(-10, 10)
        else:
            self.swimmer_lines.set_data(Rx, Ry)
            self.action_text.set_text(str(a))
        plt.draw()

    def showLearning(self, representation):
        good_pol = SwimmerPolicy(
            representation=representation,
            epsilon=0)
        id1 = 2
        id2 = 3
        res = 200
        s = np.zeros(self.state_space_dims)
        l1 = np.linspace(
            self.statespace_limits[id1, 0], self.statespace_limits[id1, 1], res)
        l2 = np.linspace(
            self.statespace_limits[id2, 0], self.statespace_limits[id2, 1], res)

        pi = np.zeros((res, res), 'uint8')
        good_pi = np.zeros((res, res), 'uint8')
        V = np.zeros((res, res))

        for row, x1 in enumerate(l1):
            for col, x2 in enumerate(l2):
                s[id1] = x1
                s[id2] = x2
                # Array of Q-function evaluated at all possible actions at
                # state s
                Qs = representation.Qs(s, False)
                # Assign pi to be optimal action (which maximizes Q-function)
                maxQ = np.max(Qs)
                pi[row, col] = np.random.choice(np.arange(len(Qs))[Qs == maxQ])
                good_pi[row, col] = good_pol.pi(
                    s, False, np.arange(self.actions_num))
                # Assign V to be the value of the Q-function under optimal
                # action
                V[row, col] = maxQ

        self._plot_policy(
            pi,
            title="Learned Policy",
            ylim=self.statespace_limits[id1],
            xlim=self.statespace_limits[id2])
        self._plot_policy(
            good_pi,
            title="Good Policy",
            var="good_policy_fig",
            ylim=self.statespace_limits[id1],
            xlim=self.statespace_limits[id2])
        self._plot_valfun(
            V,
            ylim=self.statespace_limits[id1],
            xlim=self.statespace_limits[id2])

        if self.policy_fig is None or self.valueFunction_fig is None:
            plt.show()

    def _plot_policy(self, piMat, title="Policy",
                     var="policy_fig", xlim=None, ylim=None):
        """
        :returns: handle to the figure
        """

        if getattr(self, var, None) is None:
            plt.figure(title)
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            cmaplist[0] = (.5, .5, .5, 1.0)
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0, self.actions_num, self.actions_num + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            if xlim is not None and ylim is not None:
                extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
            else:
                extent = [0, 1, 0, 1]
            self.__dict__[var] = plt.imshow(
                piMat,
                interpolation='nearest',
                origin='lower',
                cmap=cmap,
                norm=norm,
                extent=extent)
            #pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
            #pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
            #pl.xlabel(r"$\theta$ (degree)")
            #pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
            plt.title(title)

            plt.colorbar()
        plt.figure(title)
        self.__dict__[var].set_data(piMat)
        plt.draw()

    def _plot_valfun(self, VMat, xlim=None, ylim=None):
        """
        :returns: handle to the figure
        """
        plt.figure("Value Function")
        #pl.xticks(self.xTicks,self.xTicksLabels, fontsize=12)
        #pl.yticks(self.yTicks,self.yTicksLabels, fontsize=12)
        #pl.xlabel(r"$\theta$ (degree)")
        #pl.ylabel(r"$\dot{\theta}$ (degree/sec)")
        plt.title('Value Function')
        if xlim is not None and ylim is not None:
            extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        else:
            extent = [0, 1, 0, 1]
        self.valueFunction_fig = plt.imshow(
            VMat,
            cmap='ValueFunction',
            interpolation='nearest',
            origin='lower',
            extent=extent)

        norm = colors.Normalize(vmin=VMat.min(), vmax=VMat.max())
        self.valueFunction_fig.set_data(VMat)
        self.valueFunction_fig.set_norm(norm)
        plt.draw()

    def _body_coord(self):
        """
        transforms the current state into coordinates that are more
        reasonable for learning
        returns a 4-tupel consisting of:
        nose position, joint angles (d-1), nose velocity, angular velocities

        The nose position and nose velocities are referenced to the nose rotation.
        """
        cth = np.cos(self.theta)
        sth = np.sin(self.theta)
        M = self.P - 0.5 * np.diag(self.lengths)
        #  stores the vector from the center of mass to the nose
        c2n = np.array([np.dot(M[self.nose], cth), np.dot(M[self.nose], sth)])
        #  absolute position of nose
        T = -self.pos_cm - c2n - self.goal
        #  rotating coordinate such that nose is axis-aligned (nose frame)
        #  (no effect when  \theta_{nose} = 0)
        c2n_x = np.array([cth[self.nose], sth[self.nose]])
        c2n_y = np.array([-sth[self.nose], cth[self.nose]])
        Tcn = np.array([np.sum(T * c2n_x), np.sum(T * c2n_y)])

        #  velocity at each joint relative to center of mass velocity
        vx = -np.dot(M, sth * self.dtheta)
        vy = np.dot(M, cth * self.dtheta)
        #  velocity at nose (world frame) relative to center of mass velocity
        v2n = np.array([vx[self.nose], vy[self.nose]])
        #  rotating nose velocity to be in nose frame
        Vcn = np.array([np.sum((self.v_cm + v2n) * c2n_x),
                        np.sum((self.v_cm + v2n) * c2n_y)])
        #  angles should be in [-pi, pi]
        ang = np.mod(
            self.theta[1:] - self.theta[:-1] + np.pi,
            2 * np.pi) - np.pi
        return Tcn, ang, Vcn, self.dtheta

    def step(self, a):
        d = self.d
        a = self.actions[a]
        s = np.hstack((self.pos_cm, self.theta, self.v_cm, self.dtheta))
        ns = rk4(
            dsdt, s, [0,
                      self.dt], a, self.P, self.inertia, self.G, self.U, self.lengths,
            self.masses, self.k1, self.k2)[-1]

        self.theta = ns[2:2 + d]
        self.v_cm = ns[2 + d:4 + d]
        self.dtheta = ns[4 + d:]
        self.pos_cm = ns[:2]
        return (
            self._reward(
                a), self.state, self.isTerminal(), self.possibleActions()
        )

    def _dsdt(self, s, a):
        """ just a convenience function for testing and debugging, not really used"""
        return dsdt(
            s, 0., a, self.P, self.inertia, self.G, self.U, self.lengths,
            self.masses, self.k1, self.k2)

    def _reward(self, a):
        """
        penalizes the l2 distance to the goal (almost linearly) and
        a small penalty for torques coming from actions
        """

        xrel = self._body_coord()[0] - self.goal
        dist = np.sum(xrel ** 2)
        return (
            - self.cx * dist / (np.sqrt(dist) + 1) - self.cu * np.sum(a ** 2)
        )


def dsdt(s, t, a, P, I, G, U, lengths, masses, k1, k2):
    """
    time derivative of system dynamics
    """
    d = len(a) + 1
    theta = s[2:2 + d]
    vcm = s[2 + d:4 + d]
    dtheta = s[4 + d:]

    cth = np.cos(theta)
    sth = np.sin(theta)
    rVx = np.dot(P, -sth * dtheta)
    rVy = np.dot(P, cth * dtheta)
    Vx = rVx + vcm[0]
    Vy = rVy + vcm[1]

    Vn = -sth * Vx + cth * Vy
    Vt = cth * Vx + sth * Vy

    EL1 = np.dot((v1Mv2(-sth, G, cth) + v1Mv2(cth, G, sth)) * dtheta[None, :]
                 + (v1Mv2(cth, G, -sth) + v1Mv2(sth, G, cth)) * dtheta[:, None], dtheta)
    EL3 = np.diag(I) + v1Mv2(sth, G, sth) + v1Mv2(cth, G, cth)
    EL2 = - k1 * np.dot((v1Mv2(-sth, P.T, -sth) + v1Mv2(cth, P.T, cth)) * lengths[None, :], Vn) \
          - k1 * np.power(lengths, 3) * dtheta / 12. \
          - k2 * \
        np.dot((v1Mv2(-sth, P.T, cth) + v1Mv2(cth, P.T, sth))
               * lengths[None, :], Vt)
    ds = np.zeros_like(s)
    ds[:2] = vcm
    ds[2:2 + d] = dtheta
    ds[2 + d] = - \
        (k1 * np.sum(-sth * Vn) + k2 * np.sum(cth * Vt)) / np.sum(masses)
    ds[3 + d] = - \
        (k1 * np.sum(cth * Vn) + k2 * np.sum(sth * Vt)) / np.sum(masses)
    ds[4 + d:] = np.linalg.solve(EL3, EL1 + EL2 + np.dot(U, a))
    return ds


def v1Mv2(v1, M, v2):
    """
    computes diag(v1) dot M dot diag(v2).
    returns np.ndarray with same dimensions as M
    """
    return v1[:, None] * M * v2[None, :]
