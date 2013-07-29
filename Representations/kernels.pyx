# -*- coding: utf-8 -*-
"""
Kernel functions used in KiFDD
"""
__author__ = "Christoph Dann <cdann@cdann.de>"
import numpy as np
#cython: boundscheck=False
#cython: cdivision=True

cimport numpy as np
cimport cython
cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double exp(double)
    double floor(double)
    double fmin(double, double)
    double fmax(double, double)

def truncated_gaussian_kernel(np.ndarray[np.double_t, ndim=1] x,
                             np.ndarray[np.double_t, ndim=1] y,
                             list dim,
                             np.ndarray[np.double_t, ndim=1] sigma,
                             double threshold):
    cdef double res = 0.
    cdef int i=0
    for d in dim:
        i = int(d) # from cython's html output, I assume this casts makes
                   # the execution much faster
        res += ((x[i] - y[i]) / sigma[i]) ** 2
    res = exp(-res)
    res -= threshold
    res *= 1. / (1. - threshold)
    res = fmax(res, 0.)
    return res

def gaussian_kernel(np.ndarray[np.double_t, ndim=1] x,
                    np.ndarray[np.double_t, ndim=1] y,
                    list dim,
                    np.ndarray[np.double_t, ndim=1] sigma):
    cdef double res = 0.
    cdef int i=0
    for d in dim:
        i = int(d) # from cython's html output, I assume this casts makes
                   # the execution much faster
        res += ((x[i] - y[i]) / sigma[i]) ** 2
    res = exp(-res)
    return res


def discretization_kernel(np.ndarray[np.double_t, ndim=1] x,
                          np.ndarray[np.double_t, ndim=1] y,
                          list dim,
                          np.ndarray[np.double_t, ndim=1] sigma):
    cdef int i=0
    for d in dim:
        i = int(d)
        if floor(x[i] / sigma[i]) != floor(y[i] / sigma[i]):
            return 0.
    return 1.


def linf_kernel(np.ndarray[np.double_t, ndim=1] x,
                          np.ndarray[np.double_t, ndim=1] y,
                          list dim,
                          np.ndarray[np.double_t, ndim=1] sigma):
    cdef int i = 0
    for d in dim:
        i = int(d)
        if abs(x[i] - y[i]) > sigma[i]:
            return 0.
    return 1.


def linf_triangle_kernel(np.ndarray[np.double_t, ndim=1] x,
                          np.ndarray[np.double_t, ndim=1] y,
                          list dim,
                          np.ndarray[np.double_t, ndim=1] sigma):
    cdef double mi = 1.
    cdef int i = 0
    cdef double r
    for d in dim:
        i = int(d)
        r = 1 - abs(x[i] - y[i]) / sigma[i]
        if r <= 0:
            return 0.
        else:
            mi = fmin(r, mi)
    return mi
