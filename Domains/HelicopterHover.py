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

from Tools import *
from Domain import *
import numpy as np
import Tools.transformations as trans
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
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

    The state of the helicopter is described by a 18-dimensional vector
    with the following entries:

    0: xerr [helicopter x-coord position - desired x-coord position] -- helicopter's x-axis points forward
    1: yerr [helicopter y-coord position - desired y-coord position] -- helicopter's y-axis points to the right
    2: zerr [helicopter z-coord position - desired z-coord position] -- helicopter's z-axis points down
    3: u [forward velocity]
    4: v [sideways velocity (to the right)]
    5: w [downward velocity]
    6-8: orientation of heli in world as imaginary part of the quaterion representation
    9: p [angular rate around helicopter's x axis]
    10: q [angular rate around helicopter's y axis]
    11: r [angular rate around helicopter's z axis]
    12-17: current noise due to gusts (usually not observable!)
    18: t number of timesteps in current episode
    """

    MAX_POS = 20.  # m
    MAX_VEL = 10.  # m/s
    MAX_ANG_RATE = 4 * np.pi
    MAX_ANG = 1.
    WIND_MAX = 5.
    MIN_QW_BEFORE_HITTING_TERMINAL_STATE = np.cos(30. / 2. * np.pi / 180.) # ??

    wind = np.array([5., 2., 0.]) # wind in neutral orientation

    gust_memory = 0.8
    domain_fig = None
    states_num = 12
    episodeCap = 6000
    continuous_dims = np.arange(18)
    statespace_limits = np.array([[-MAX_POS, MAX_POS]] * 3 + [[-MAX_VEL, MAX_VEL]] * 3
                                 + [[-MAX_ANG, MAX_ANG]] * 3
                                 + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3
                                 + [[0,episodeCap]]
                                 + [[-2., 2.]] * 6)

    # create all combinations of possible actions
    def _make_slice(l, u, n):
        return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

    _action_bounds = np.array([[-.2, .2]] * 4) # maximum action: 2
    _action_slices = [3] * 4 # 3 actions per dimension
    _action_slices = [_make_slice(
        b[0], b[1], n) for b, n in zip(_action_bounds, _action_slices)]
    actions = np.mgrid[_action_slices[0], _action_slices[1],
                       _action_slices[2], _action_slices[3]]
    actions_num = np.prod(actions[0].shape)

    def s0(self):
        return np.zeros((19))

    def isTerminal(self, s):
        return np.any(self.statespace_limits[:12, 0] > s[:12]) or np.any(self.statespace_limits[:12, 1] < s[:12])

    def _get_reward(self, s):
        """ defines the reward for a given state
        differs from the JAVA implementation!"""
        if self.isTerminal(s):
            r = -np.sum(self.statespace_limits[:12, 1] ** 2)
            r -= self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE ** 2
            return r * (self.episodeCap - s[-1])
        else:
            return -np.sum(s[:12] ** 2)

    def possibleActions(self, s):
        return np.arange(self.actions_num)

    def step(self, s, a):
        a = self.actions.reshape(self.actions.shape[0], -1)[:, a]
        # make sure the actions are not beyond their limits
        a = np.maximum(self._action_bounds[:, 0], np.minimum(a,
                       self._action_bounds[:, 1]))

        noise_std = np.array([0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734])
        drag_vel_body = np.array([.18, .43, .49])
        drag_ang_rate = np.array([12.78, 10.12, 8.16])
        u_coeffs = np.array([-33.04, -33.32, 70.54, -42.15])
        tail_rotor_side_thrust = -0.54
        dt = 0.01

        pos, vel, ang_rate, ori_bases, q = self._state_in_world(s[:12])
        t = s[18]
        gust_noise = s[12:18]
        gust_noise = (self.gust_memory * gust_noise
                      + (1. - self.gust_memory) * np.random.randn(6) * noise_std)
        # update noise which simulates gusts

        vel_body = s[3:6].copy()
        wind_body = self._in_body_coord(self.wind, q)
        acc_body = np.zeros((3))
        for i in range(10):

            # Euler integration
            # position
            pos += dt * vel

            # compute acceleration on the helicopter
            acc_body = -drag_vel_body * (vel_body + wind_body)
            acc_body[-1] += u_coeffs[-1] * a[-1]
            acc_body[1] += tail_rotor_side_thrust
            acc_body += gust_noise[:3]
            acc = self._in_world_coord(acc_body, q)
            acc[-1] += 9.81  # gravity

            # velocity
            vel += dt * acc

            # orientation
            tmp = dt*ang_rate
            qdt = trans.quaternion_about_axis(np.linalg.norm(tmp), tmp)
            q = trans.quaternion_multiply(q, qdt)
            assert np.allclose(1., np.sum(q**2))

            # angular accelerations
            ang_acc = -ang_rate * drag_ang_rate + u_coeffs[:3] * a[:3]
            ang_acc += gust_noise[3:]

            ang_rate += dt * ang_acc
        st = np.zeros_like(s)
        st[:3] = self._in_body_coord(pos, q)
        st[3:6] = self._in_body_coord(vel, q)
        st[6:9] = q[1:]
        #st[6:9] = trans.euler_from_quaternion(trans.quaternion_conjugate(q))
        st[9:12] = ang_rate
        st[-1] = t + 1
        st[12:18] = gust_noise
        return (self._get_reward(s=st), st, self.isTerminal(st))



    def _state_in_world(self, s):
        """
        angular rate still in body frame!
        """
        pos_body = s[:3].copy()
        vel_body = s[3:6].copy()
        ori = s[6:9].copy()
        ang_rate = s[9:12].copy()

        q_ned_body = np.zeros((4))
        q_ned_body[1:] = ori
        q_ned_body[0] = np.sqrt(1. - np.sum(q_ned_body[1:]**2))
        q_body_ned = trans.quaternion_conjugate(q_ned_body)
        q_pos = np.zeros((4))
        q_pos[1:] = -pos_body
        q_pos = trans.quaternion_multiply(trans.quaternion_multiply(q_body_ned, q_pos), q_ned_body)
        pos_ned = q_pos[1:]
        q_vel = np.zeros((4))
        q_vel[1:] = vel_body
        q_vel = trans.quaternion_multiply(trans.quaternion_multiply(q_body_ned, q_vel), q_ned_body)
        vel_ned = q_vel[1:]

        rot = trans.quaternion_matrix(q_body_ned)[:3, :3]
        return pos_ned, vel_ned, ang_rate, rot, q_ned_body

    def _in_body_coord(self, p, q):
        """
        q is the quaternion of the rotation of the helicopter in world coordinates
        """
        q_pos = np.zeros((4))
        q_pos[1:] = p
        q_p = trans.quaternion_multiply(trans.quaternion_multiply(q, q_pos), trans.quaternion_conjugate(q))
        return q_p[1:]

    def _in_world_coord(self, p, q):
        """
        q is the quaternion of the rotation of the helicopter in world coordinates
        """
        return self._in_body_coord(p, trans.quaternion_conjugate(q))

    def showDomain(self, s,a=None):
        pos, vel, ang_rate, ori_bases, _ = self._state_in_world(s[:12])
        coords = np.zeros((3, 3, 2)) + pos[None, :, None]
        coords[:, :, 1] += ori_bases * 2
        if self.domain_fig is None:
            self.domain_fig = plt.figure()
            ax = self.domain_fig.gca(projection='3d')
            ax.set_aspect("equal")
            lim = 2  # self.MAX_POS
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            # print origin
            x = Arrow3D([0,.5],[0,0],[0,0], mutation_scale=30, lw=1, arrowstyle="-|>", color="r")
            y = Arrow3D([0,0],[0,.5],[0,0], mutation_scale=30, lw=1, arrowstyle="-|>", color="b")
            z = Arrow3D([0,0],[0,0],[0,.5], mutation_scale=30, lw=1, arrowstyle="-|>", color="g")
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)

            # print helicopter
            x = Arrow3D(*coords[0], mutation_scale=30, lw=2, arrowstyle="-|>", color="r")
            y = Arrow3D(*coords[1], mutation_scale=30, lw=2, arrowstyle="-|>", color="b")
            z = Arrow3D(*coords[2], mutation_scale=30, lw=2, arrowstyle="-|>", color="g")
            ax.add_artist(x)
            ax.add_artist(y)
            ax.add_artist(z)
            self.heli_arrows = (x, y, z)
            plt.show()
        else:
            self.heli_arrows[0]._verts3d = tuple(coords[0])
            self.heli_arrows[1]._verts3d = tuple(coords[1])
            self.heli_arrows[2]._verts3d = tuple(coords[2])
            self.domain_fig.canvas.draw()



class HelicopterHover(HelicopterExtended):
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
    6-8: orientation of heli in world as imaginary part of the quaterion representation
    9: p [angular rate around helicopter's x axis]
    10: q [angular rate around helicopter's y axis]
    11: r [angular rate around helicopter's z axis]
    """

    domain_fig = None
    states_num = 12
    episodeCap = 6000
    continuous_dims = np.arange(12)
    statespace_limits = np.array([[-MAX_POS, MAX_POS]] * 3 + [[-MAX_VEL, MAX_VEL]] * 3
                                 + [[-MAX_ANG, MAX_ANG]] * 3 + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3)

    def s0(self):
        return np.zeros((12))

    def _get_reward(self, s):
        """ defines the reward for a given state
        differs from the JAVA implementation!"""
        if self.isTerminal(s):
            r = -np.sum(self.statespace_limits[:, 1] ** 2)
            r -= self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE ** 2
            return r * self.episodeCap
        else:
            return -np.sum(s ** 2)

    def step(self, s, a):
        s_extended = np.zeros((18))
        s_extended[:12] = s

        r, st_extended, term = super(HelicopterHover,self).step(s_extended, a)
        return (r, st_extended[:12], term)


    def showDomain(self, s,a=None):
        s_extended = np.zeros((18))
        s_extended[:12] = s

        super(HelicopterHover,self).showDomain(s_extended, a)

if __name__ == "__main__":
    h = HelicopterHoverExtended(None)
    h.test(100)
