"""Bicycle balancing task."""

from Domain import Domain
import numpy as np
from itertools import product

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class BicycleBalancing(Domain):
    """
    Simulation of balancing a bicycle.
    
    **STATE:**
    The state contains of 5 variables. Four are related to the bicycle and
    the remaining three describe its position on the plane.

    * *omega:*     angle from the vertical to the bicycle [rad]
    * *omega dot:* angular velocity for omega [rad / s]
    * *theta:*     angle the handlebars are displaced from normal [rad]
    * *theta dot:* angular velocity for theta [rad / s]
    * *psi:*       angle formed by bicycle frame and x-axis [rad]
    [x_b:       x-coordinate where the back tire touches the ground [m]] \n
    [y_b:       y-coordinate where the back tire touches the ground [m]] \N
    
    *The state variables *x_b* and *y_b* are not observable.*

    **ACTIONS:**
    
    * T in {-2, 0, 2}:      the torque applied to the handlebar
    * d in {-.02, 0, .02}:  displacement of the rider
    i.e., 9 actions in total.


    For details see

        Ernst, D., Geurts, P. & Wehenkel, L. Tree-Based Batch Mode Reinforcement Learning.
        Journal of Machine Learning Research (2005) Issue 6

    .. warning::
        This domain is tested only marginally, use with a care.
    """

    gamma = 0.98
    continuous_dims = np.arange(5)
    actions = np.array(list(product([-2, 0, 2], [-.02, 0., .02])))
    actions_num = len(actions)
    episodeCap = 50000  #: Total episode duration is ``episodeCap * dt`` sec.
    dt = 0.01  #: Frequency is ``1 / dt``.
    statespace_limits = np.array([[-np.pi*12/180, np.pi*12/180],
                                  [-np.pi, np.pi],
                                  [-np.pi*80/180, np.pi*80/180],
                                  [-np.pi, np.pi],
                                  [-np.pi, np.pi]])

    def step(self, a):
        s = self.state
        T, d = self.actions[a]
        omega, domega, theta, dtheta, psi = s

        v = 10 / 3.6
        g = 9.82
        d_CM = 0.3
        c = 0.66
        h = 0.94
        M_c = 15.
        M_d = 1.7
        M_p = 60.
        M = M_c + M_p
        r = 0.34
        dsigma = v / r
        I = 13/3. * M_c * h**2 + M_p * (h + d_CM)**2
        I_dc = M_d * r**2
        I_dv = 3 / 2. * M_d * r**2
        I_dl = M_d / 2 * r**2
        l = 1.11

        w = self.random_state.uniform(-0.02, 0.02)

        phi = omega + np.arctan(d + w) / h
        invr_f = np.abs(np.sin(theta)) / l
        invr_b = np.abs(np.tan(theta)) / l
        invr_CM = ((l - c) ** 2 + invr_b ** (-2)) ** (-.5) if theta != 0. else 0.

        nomega = omega + self.dt * domega
        ndomega = domega + self.dt * (M*h*g*np.sin(phi) - np.cos(phi) *
                                      (I_dc * dsigma * dtheta +
                                       np.sign(theta) * v**2*(M_d * r * (invr_f + invr_b) + M*h*invr_CM))) / I
        out = theta + self.dt * dtheta
        ntheta = out if abs(out) > 80. / 180 * np.pi else np.sign(out)*80./180*np.pi
        ndtheta = dtheta + self.dt * (T - I_dv * dsigma * domega) / I_dl if abs(out) > 80./180*np.pi else 0.
        npsi = psi * self.dt * np.sign(theta) * v * invr_b
        npsi = npsi % (2 * np.pi)
        if npsi > np.pi:
            npsi -= 2 * np.pi
        ns = np.array([nomega, ndomega, ntheta, ndtheta, npsi])
        self.state = ns
        return self._reward(s), ns, self.isTerminal(), self.possibleActions()

    def isTerminal(self):
        s = self.state
        omega = s[0]
        return omega < -np.pi * 12. / 180 or omega > np.pi * 12. / 180.

    def _reward(self, s):
        return -1. if self.isTerminal() else 0.

    def possibleActions(self):
        return np.arange(9)

    def s0(self):
        # non-healthy stable state of the system
        s = np.zeros(5)
        self.state = s
        return s, self.isTerminal(), self.possibleActions()


class BicycleRiding(BicycleBalancing):
    def _reward(self, s):
        ns = self.state
        psi = s[-1]
        npsi = ns[-1]
        if self.isTerminal():
            return -1.
        return 0.1*(psi - npsi)
