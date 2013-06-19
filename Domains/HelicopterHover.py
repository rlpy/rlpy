"""
WORK IN PROGRESS
"""
#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


__author__ = "Christoph Dann"

import sys
import os
#Add all paths
RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT + '/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RLPy'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)

from Domain import Domain
import numpy as np
import Tools.transformations as trans
from Tools.GeneralTools import cartesian
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """
    Helper class for plotting arrows in 3d
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class HelicopterHoverExtended(Domain):
    """
    Implementation of a simulator that models one of the Stanford
    autonomous helicopters (an XCell Tempest helicopter) in the flight
    regime close to hover.

    See
    Abbeel, P., Ganapathi, V. & Ng, A. Learning vehicular dynamics,
    with application to modeling helicopters.
    Advances in Neural Information Systems (2006).

    Adapted from the Java Implementation documented at
    http://library.rl-community.org/wiki/Helicopter_(Java)

    The state of the helicopter is described by a 20-dimensional vector
    with the following entries:

    0: xerr [helicopter x-coord position - desired x-coord position] -- helicopter's x-axis points forward
    1: yerr [helicopter y-coord position - desired y-coord position] -- helicopter's y-axis points to the right
    2: zerr [helicopter z-coord position - desired z-coord position] -- helicopter's z-axis points down
    3: u [forward velocity]
    4: v [sideways velocity (to the right)]
    5: w [downward velocity]
    6: p [angular rate around helicopter's x axis]
    7: q [angular rate around helicopter's y axis]
    8: r [angular rate around helicopter's z axis]
    9-12: orientation of heli in world as quaterion
    13-18: current noise due to gusts (usually not observable!)
    19: t number of timesteps in current episode
    """

    MAX_POS = 20.  # m
    MAX_VEL = 10.  # m/s
    MAX_ANG_RATE = 4 * np.pi
    MAX_ANG = 1.
    WIND_MAX = 5.
    MIN_QW_BEFORE_HITTING_TERMINAL_STATE = np.cos(30. / 2. * np.pi / 180.)

    wind = np.array([.0, .0, 0.])  # wind in neutral orientation
    gamma = 0.9  # discout factor
    gust_memory = 0.8
    domain_fig = None

    episodeCap = 6000
    noise_std = np.array([0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734])
    drag_vel_body = np.array([.18, .43, .49])
    drag_ang_rate = np.array([12.78, 10.12, 8.16])
    u_coeffs = np.array([-33.04, -33.32, 70.54, -42.15])
    tail_rotor_side_thrust = -0.54

    dt = 0.01  # length of one timestep
    continuous_dims = np.arange(20)
    statespace_limits_full = np.array([[-MAX_POS, MAX_POS]] * 3
                                 + [[-MAX_VEL, MAX_VEL]] * 3
                                 + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3
                                 + [[-MAX_ANG, MAX_ANG]] * 4
                                 + [[-2., 2.]] * 6
                                 + [[0, episodeCap]])
    statespace_limits = statespace_limits_full
    # create all combinations of possible actions

    #def _make_slice(l, u, n):
    #    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))
    _action_bounds = np.array([[-2., 2.]] * 4)
    _actions_dim = np.array([[-.2, -0.05, 0.05, 0.2]] * 3 + [[0., 0.15, 0.3, 0.5]])  # maximum action: 2
    #_action_slices = [3]*3+[4]  # 3 actions per dimension
    #_action_slices = [_make_slice(
    #    b[0], b[1], n) for b, n in zip(_action_bounds, _action_slices)]
    #actions = np.mgrid[_action_slices[0], _action_slices[1],
    #                   _action_slices[2], _action_slices[3]]
    actions = cartesian(list(_actions_dim))
    actions_num = np.prod(actions.shape[0])

    def __init__(self, logger=None, noise_level=1., gamma=0.95):
        self.noise_level = noise_level
        self.gamma = gamma
        super(HelicopterHoverExtended, self).__init__(logger)
        if self.logger:
            self.logger.log("Actions\n{0}".format(self._actions_dim))
            self.logger.log("Noise level:\t{0}".format(self.noise_level))
            self.logger.log("Wind:\t\t\t{0},{0}".format(self.wind[0], self.wind[1]))

    def s0(self):
        s = np.zeros((20))
        s[9] = 1.
        return s

    def isTerminal(self, s):
        if np.any(self.statespace_limits_full[:9, 0] > s[:9]) or np.any(self.statespace_limits_full[:9, 1] < s[:9]):
            return True

        if len(s) <= 12:
            w = np.sqrt(1. - np.sum(s[9:12] ** 2))
        else:
            w = s[9.]

        return np.abs(w) < self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE

    def _get_reward(self, s):
        if self.isTerminal(s):
            r = -np.sum(self.statespace_limits[:9, 1] ** 2)
            #r -= np.sum(self.statespace_limits[10:12, 1] ** 2)
            r -= (1. - self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE ** 2)
            return r * (self.episodeCap - s[-1])
        else:
            return -np.sum(s[:9] ** 2) - np.sum(s[10:12] ** 2)

    def possibleActions(self, s):
        return np.arange(self.actions_num)

    """    def test(self):
        a = np.array([0., 0.2, 0., 0.2])
        s = np.zeros(19)
        s[9:13] = [np.sqrt(1-(.1**2 + .2**2 + .3**2)), -0.1, 0.2, 0.3]
        for i in range(3):
            r,s,t = self.step(s, a)
    """
    def step(self, s, a):
        a = self.actions[a]
        # make sure the actions are not beyond their limits
        a = np.maximum(self._action_bounds[:, 0], np.minimum(a,
                       self._action_bounds[:, 1]))

        pos, vel, ang_rate, ori_bases, q = self._state_in_world(s)
        t = s[-1]
        gust_noise = s[13:19]
        gust_noise = (self.gust_memory * gust_noise
                      + (1. - self.gust_memory) * np.random.randn(6) * self.noise_level * self.noise_std)
        # update noise which simulates gusts
        for i in range(10):
            # Euler integration
            # position
            pos += self.dt * vel
            # compute acceleration on the helicopter
            vel_body = self._in_world_coord(vel, q)
            wind_body = self._in_world_coord(self.wind, q)
            wind_body[-1] = 0.  # the java implementation
                                # has it this way
            acc_body = -self.drag_vel_body * (vel_body + wind_body)
            acc_body[-1] += self.u_coeffs[-1] * a[-1]
            acc_body[1] += self.tail_rotor_side_thrust
            acc_body += gust_noise[:3]
            acc = self._in_body_coord(acc_body, q)
            acc[-1] += 9.81  # gravity

            # velocity
            vel += self.dt * acc
            # orientation
            tmp = self.dt * ang_rate
            qdt = trans.quaternion_about_axis(np.linalg.norm(tmp), tmp)
            q = trans.quaternion_multiply(q, qdt)
            #assert np.allclose(1., np.sum(q**2))
            # angular accelerations
            ang_acc = -ang_rate * self.drag_ang_rate + self.u_coeffs[:3] * a[:3]
            ang_acc += gust_noise[3:]

            ang_rate += self.dt * ang_acc

        st = np.zeros_like(s)
        st[:3] = -self._in_body_coord(pos, q)
        st[3:6] = self._in_body_coord(vel, q)
        st[6:9] = ang_rate
        st[9:13] = q
        st[13:19] = gust_noise
        st[-1] = t + 1
        return (self._get_reward(s=st), st, self.isTerminal(st))

    def _state_in_world(self, s):
        """
        angular rate still in body frame!
        """
        pos_body = s[:3]
        vel_body = s[3:6]
        ang_rate = s[6:9].copy()
        q = s[9:13].copy()

        pos = self._in_world_coord(-pos_body, q)
        vel = self._in_world_coord(vel_body, q)

        rot = trans.quaternion_matrix(trans.quaternion_conjugate(q))[:3, :3]
        return pos, vel, ang_rate, rot, q

    def _in_body_coord(self, p, q):
        """
        q is the inverse quaternion of the rotation of the helicopter in world coordinates
        """
        q_pos = np.zeros((4))
        q_pos[1:] = p
        q_p = trans.quaternion_multiply(trans.quaternion_multiply(q, q_pos),
                                        trans.quaternion_conjugate(q))
        return q_p[1:]

    def _in_world_coord(self, p, q):
        """
        q is the inverse quaternion of the rotation of the helicopter in world coordinates
        """
        return self._in_body_coord(p, trans.quaternion_conjugate(q))

    def test(self):
        s = self.s0()
        a = np.zeros((4))
        a[0] = .0
        a[1] = 0.
        a[2] = 0.
        a[3] = .3
        for i in range(20):
            r, s, t = self.step(s, a)
            self.showDomain(s, a)
            import time
            time.sleep(.5)
    def showDomain(self, s, a=None):
        if a is not None:
            a = self.actions[a].copy() * 3 # amplify for visualization
        pos, vel, ang_rate, ori_bases, _ = self._state_in_world(s)
        coords = np.zeros((3, 3, 2)) + pos[None, :, None]
        coords[:, :, 1] += ori_bases * 4
        u, v = np.mgrid[0:2*np.pi:10j, 0:2:1.]

        # rotor coordinates
        coord = np.zeros([3] + list(u.shape))
        coord[0] = .1 * np.sin(u)*v
        coord[1] = 0.
        coord[2] = .1 * np.cos(u)*v
        coord[0] -= 0.8
        coord_side = np.einsum("ij,jkl->ikl", np.linalg.pinv(ori_bases), coord)
        coord_side += pos[:, None, None]

        coord = np.zeros([3] + list(u.shape))
        coord[0] = .6 * np.cos(u)*v
        coord[1] = .6 * np.sin(u)*v
        coord[2] = -.4
        coord_main = np.einsum("ij,jkl->ikl", np.linalg.pinv(ori_bases), coord)
        coord_main += pos[:, None, None]

        style = dict(fc="r", ec="r", lw=2., head_width=0.05, head_length=0.1)
        if self.domain_fig is None:
            self.domain_fig = plt.figure(figsize=(12,8))
            # action axes
            ax1 = plt.subplot2grid((1, 3), (0, 0), frameon=False)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            lim = 2  # self.MAX_POS
            ax1.set_xlim(-lim, lim)
            ax1.set_ylim(-lim, lim)
            if a is None:
                a = np.zeros((4))
            # main rotor
            ax1.add_artist(Circle(np.zeros((2)), radius=0.6))
            ax1.add_artist(Ellipse(np.array([0, 1.5]), height=0.3, width=0.02))
            # TODO make sure the actions are plotted right
            # main rotor direction?
            arr1 = ax1.arrow(0, 0, a[0], 0, **style)
            arr2 = ax1.arrow(0, 0, 0, a[1], **style)
            # side rotor throttle?
            arr3 = ax1.arrow(0, 1.5, a[2],0, **style)
            # main rotor throttle
            arr4 = ax1.arrow( 1.5, 0, 0, a[3], **style)
            ax1.set_aspect("equal")

            self.action_arrows = (arr1, arr2, arr3, arr4)
            self.action_ax = ax1
            #ax = self.domain_fig.gca(projection='3d')
            ax = plt.subplot2grid((1, 3), (0, 1), colspan=2, projection='3d')
            ax.view_init(elev=np.pi)
            # print origin
            x = Arrow3D([0, 2], [0, 0], [0, 0], mutation_scale=30, lw=1,
                        arrowstyle="-|>", color="r")
            y = Arrow3D([0, 0], [0, 2], [0, 0], mutation_scale=30, lw=1,
                        arrowstyle="-|>", color="b")
            z = Arrow3D([0, 0], [0, 0], [0, 2], mutation_scale=30, lw=1,
                        arrowstyle="-|>", color="g")
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)

            # print helicopter coordinate axes
            x = Arrow3D(*coords[0], mutation_scale=30, lw=2, arrowstyle="-|>",
                        color="r")
            y = Arrow3D(*coords[1], mutation_scale=30, lw=2, arrowstyle="-|>",
                        color="b")
            z = Arrow3D(*coords[2], mutation_scale=30, lw=2, arrowstyle="-|>",
                        color="g")
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)
            self.heli_arrows = (x, y, z)

            self._wframe_main = ax.plot_wireframe(coord_main[0], coord_main[1],
                                                  coord_main[2], color="k")
            self._wframe_side = ax.plot_wireframe(coord_side[0], coord_side[1],
                                                  coord_side[2], color="k")
            self._ax = ax
            ax.set_aspect("equal")
            lim = 5  # self.MAX_POS
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.view_init(elev=-135)
            plt.show()
        else:
            self.heli_arrows[0]._verts3d = tuple(coords[0])
            self.heli_arrows[1]._verts3d = tuple(coords[1])
            self.heli_arrows[2]._verts3d = tuple(coords[2])
            ax = self._ax
            ax.collections.remove(self._wframe_main)
            ax.collections.remove(self._wframe_side)
            for arr in self.action_arrows:
                self.action_ax.artists.remove(arr)
            ax1 = self.action_ax
            # TODO make sure the actions are plotted right
            # main rotor direction?
            arr1 = ax1.arrow(0, 0, a[0], 0, **style)
            arr2 = ax1.arrow(0, 0, 0, a[1], **style)
            # side rotor throttle?
            arr3 = ax1.arrow(0, 1.5, a[2],0, **style)
            # main rotor throttle
            arr4 = ax1.arrow( 1.5, 0, 0, a[3], **style)
            self.action_arrows = (arr1, arr2, arr3, arr4)

            self._wframe_main = ax.plot_wireframe(coord_main[0], coord_main[1],
                                                  coord_main[2], color="k")
            self._wframe_side = ax.plot_wireframe(coord_side[0], coord_side[1],
                                                  coord_side[2], color="k")
            ax.set_aspect("equal")
            lim = 5  # self.MAX_POS
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.view_init(elev=-135)
            self.domain_fig.canvas.draw()


class HelicopterHover(HelicopterHoverExtended):
    """
    WARNING: This domain has an internal hidden state, as it actually is
    a POMDP. Besides the 12-dimensional observable state, there is an internal
    state saved as self.hidden_state_ (time and long-term noise which
    simulated gusts of wind).
    be aware of this state if you use this class to produce samples which are
    not in order

    Implementation of a simulator that models one of the Stanford
    autonomous helicopters (an XCell Tempest helicopter) in the flight
    regime close to hover.

    See
    Abbeel, P., Ganapathi, V. & Ng, A. Learning vehicular dynamics,
    with application to modeling helicopters.
    Advances in Neural Information Systems (2006).

    Adapted from the Java Implementation documented at
    http://library.rl-community.org/wiki/Helicopter_(Java)
    However: adapted to be a true MDP; on termination a constant negative
    reward is given, not dependent on the remaining simulation time and
    the gusts are simulated by gaussian noise for each time-step

    The state of the helicopter is described by a 12-dimensional vector
    with the following entries:

    0: xerr [helicopter x-coord position - desired x-coord position] -- helicopter's x-axis points forward
    1: yerr [helicopter y-coord position - desired y-coord position] -- helicopter's y-axis points to the right
    2: zerr [helicopter z-coord position - desired z-coord position] -- helicopter's z-axis points down
    3: u [forward velocity]
    4: v [sideways velocity (to the right)]
    5: w [downward velocity]
    6: p [angular rate around helicopter's x axis]
    7: q [angular rate around helicopter's y axis]
    8: r [angular rate around helicopter's z axis]
    9-11: orientation of the world in the heli system as quaterion
    """

    episodeCap = 6000
    MAX_POS = 20.  # m
    MAX_VEL = 10.  # m/s
    MAX_ANG_RATE = 4 * np.pi
    MAX_ANG = 1.
    WIND_MAX = 5.

    continuous_dims = np.arange(12)
    statespace_limits = np.array([[-MAX_POS, MAX_POS]] * 3
                                + [[-MAX_VEL, MAX_VEL]] * 3
                                + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3
                                + [[-MAX_ANG, MAX_ANG]] * 3)

    full_state_ = np.zeros((20))

    def s0(self):
        self.hidden_state_ = np.zeros((8))
        self.hidden_state_[0] = 1.
        s = np.zeros((12))
        return s

    def _augment_state(self, s):
        s_extended = self.full_state_
        s_extended[:9] = s[:9]
        s_extended[9] = self.hidden_state_[0]
        s_extended[10:13] = s[9:12]
        s_extended[13:] = self.hidden_state_[1:]
        return s_extended

    def _split_state(self, s):
        s_observable = np.zeros((12))
        s_observable[:9] = s[:9]
        s_observable[9:12] = s[10:13]
        s_hidden = np.zeros((8))
        s_hidden[0] = s[9]
        s_hidden[1:] = s[13:]
        return s_observable, s_hidden

    def step(self, s, a):
        s_extended = self._augment_state(s)
        r, st_extended, term = super(HelicopterHover, self).step(s_extended, a)
        st, self.hidden_state_[:] = self._split_state(st_extended)
        return (r, st, term)

    def showDomain(self, s, a=None):
        s_extended = self._augment_state(s)
        super(HelicopterHover, self).showDomain(s_extended, a)

if __name__ == "__main__":
    h = HelicopterHoverExtended(logger=None, noise_level=0.)
    h.test()
