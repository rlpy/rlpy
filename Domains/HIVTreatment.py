from Domain import Domain
import numpy as np
from scipy.integrate import odeint

class HIVTreatment(Domain):
    """
    Simulation of HIV Treatment. The aims is to find an optimal drug schedule.

    The state contains concentrations of 6 different cells:

        T1: non-infected CD4+ T-lymphocytes [cells / ml]
        T1*:    infected CD4+ T-lymphocytes [cells / ml]
        T2: non-infected macrophages [cells / ml]
        T2*:    infected macrophages [cells / ml]
        V: number of free HI viruses [copies / ml]
        E: number of cytotoxic T-lymphocytes [cells / ml]

    The therapy consists of 2 drugs (reverse transcriptase inhibitor [RTI] and
    protease inhibitor [PI]) which are activated or not. The action space
    contains therefore of 4 actions:

        0: non active
        1: RTI active
        2: PI active
        3: RTI and PI active

    For details see

        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV :
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """

    gamma = 0.98
    continuous_dims = np.arange(6)
    actions = np.array([[0., 0.], [.7, 0.], [.3, 0.], [.7, .3]])
    actions_num = 4
    episodeCap = 200  #: total of 1000 days with a measurement every 5 days
    dt = 5  #: measurement every 5 days
    logspace = True  #: whether observed states are in log10 space or not
    if logspace:
        statespace_limits = np.array([[-5, 8]] * 6)
    else:
        statespace_limits = np.array([[0., 1e8]] * 6)

    def step(self, a):
        #if self.logspace:
        #    s = np.power(10, s)

        eps1, eps2 = self.actions[a]
        ns = odeint(dsdt, self.state, [0, self.dt], args=(eps1, eps2), mxstep=1000)[-1]
        T1, T2, T1s, T2s, V, E = ns
        # the reward function penalizes treatment because of side-effects
        reward = - 0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2  ** 2 + 1e3 * E
        self.state = ns.copy()
        if self.logspace:
            ns = np.log10(ns)
        return reward, ns, False, self.possibleActions()

    def possibleActions(self):
        return np.arange(4)

    def s0(self):
        # non-healthy stable state of the system
        s = np.array([163573., 5., 11945., 46., 63919., 24.])
        self.state = s.copy()
        if self.logspace:
            return np.log10(s), self.isTerminal(), self.possibleActions()
        return s, self.isTerminal(), self.possibleActions()


def dsdt(s, t, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = .34
    k1 = 8e-7
    k2 = 1e-4
    delta = .7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.
    c = 13.
    rho1 = 1.
    rho2 = 1.
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
            - ((1. - eps1) * rho1 * k1 * T1 + (1. - f * eps1) * rho2 * k2 * T2) * V
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
            - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])

try:
    from HIVTreatment_dynamics import dsdt
    print "Use cython extension for HIVTreatment dynamics"
except Exception, e:
    print e
    print "Cython extension for HIVTreatment dynamics not available, expect slow runtime"
