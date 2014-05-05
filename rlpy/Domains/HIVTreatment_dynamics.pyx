# -*- coding: utf-8 -*-
"""
Dynamics for the HIVTreatment domain
"""
import numpy as np
#cython: boundscheck=False
#cython: cdivision=True


__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"
cimport numpy as np
cimport cython

def dsdt(np.ndarray[np.double_t, ndim=1] s, double t, double eps1, double eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    cdef double lambda1 = 1e4
    cdef double lambda2 = 31.98
    cdef double d1 = 0.01
    cdef double d2 = 0.01
    cdef double f = .34
    cdef double k1 = 8e-7
    cdef double k2 = 1e-4
    cdef double delta = .7
    cdef double m1 = 1e-5
    cdef double m2 = 1e-5
    cdef double NT = 100.
    cdef double c = 13.
    cdef double rho1 = 1.
    cdef double rho2 = 1.
    cdef double lambdaE = 1
    cdef double bE = 0.3
    cdef double Kb = 100
    cdef double d_E = 0.25
    cdef double Kd = 500
    cdef double deltaE = 0.1

    # decompose state
    cdef double T1, T2, T1s, T2s, V, E
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    cdef double tmp1 = (1. - eps1) * k1 * V * T1
    cdef double tmp2 = (1. - f * eps1) * k2 * V * T2
    cdef double dT1 = lambda1 - d1 * T1 - tmp1
    cdef double dT2 = lambda2 - d2 * T2 - tmp2
    cdef double dT1s = tmp1 - delta * T1s - m1 * E * T1s
    cdef double dT2s = tmp2 - delta * T2s - m2 * E * T2s
    cdef double dV = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
               - ((1. - eps1) * rho1 * k1 * T1 + (1. - f * eps1) * rho2 * k2 * T2) * V
    cdef double dE
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
            - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])
