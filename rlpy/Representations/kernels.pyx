# -*- coding: utf-8 -*-
"""
Kernel functions used in KiFDD
"""
__author__ = "Christoph Dann <cdann@cdann.de>"
#cython: boundscheck=False
#cython: cdivision=True
from libcpp.vector cimport vector
from libcpp.string cimport string as stlstring
from libcpp cimport bool
from libcpp.set cimport set as sett
cimport numpy as np
import numpy as np
cimport cython
ctypedef unsigned int uint
cimport c_kernels
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
    if dim is None:
        for i in range(len(x)):
            res += ((x[i] - y[i]) / sigma[i]) ** 2
    else:
        for d in dim:
            i = int(d) # from cython's html output, I assume this casts makes
                   # the execution much faster
            res += ((x[i] - y[i]) / sigma[i]) ** 2
    res = exp(-res)
    res -= threshold
    res *= 1. / (1. - threshold)
    res = fmax(res, 0.)
    return res


def all_gaussian_kernel(np.ndarray[np.double_t, ndim=1] x,
                        np.ndarray[np.double_t, ndim=2, mode="c"] centers,
                        np.ndarray[np.double_t, ndim=2, mode="c"] widths):
    cdef vector[unsigned int] dimv = xrange(len(x))
    cdef np.ndarray[double, ndim=1, mode="c"] x1
    cdef np.ndarray[double, ndim=1] res = np.zeros(len(centers))
    cdef np.ndarray[double, ndim=2, mode="c"] y1 = centers
    cdef np.ndarray[double, ndim=2, mode="c"] s1 = widths
    x1 = np.ascontiguousarray(x, dtype=np.float64)
    cdef int i

    for i in xrange(len(centers)):
        res[i] = c_kernels.gaussian_kernel(&x1[0], &y1[i,0], dimv, &s1[i, 0])
    return res


def all_linf_triangle_kernel(np.ndarray[np.double_t, ndim=1] x,
                             np.ndarray[np.double_t, ndim=2, mode="c"] centers,
                             np.ndarray[np.double_t, ndim=2, mode="c"] widths):
    cdef vector[unsigned int] dimv = xrange(len(x))
    cdef np.ndarray[double, ndim=1, mode="c"] x1
    cdef np.ndarray[double, ndim=1] res = np.zeros(len(centers))
    cdef np.ndarray[double, ndim=2, mode="c"] y1 = centers
    cdef np.ndarray[double, ndim=2, mode="c"] s1 = widths
    x1 = np.ascontiguousarray(x, dtype=np.float64)
    cdef int i

    for i in xrange(len(centers)):
        res[i] = c_kernels.linf_triangle_kernel(&x1[0], &y1[i,0], dimv, &s1[i, 0])
    return res


def gaussian_kernel(np.ndarray[np.double_t, ndim=1] x,
                    np.ndarray[np.double_t, ndim=1] y,
                    list dim,
                    np.ndarray[np.double_t, ndim=1] sigma):
    cdef vector[unsigned int] dimv = dim
    cdef np.ndarray[double, ndim=1, mode="c"] x1
    cdef np.ndarray[double, ndim=1, mode="c"] y1
    cdef np.ndarray[double, ndim=1, mode="c"] s1

    # unbox NumPy array into local variable x
    # make sure we have a contiguous array in C order
    # this might produce a temporary copy

    x1 = np.ascontiguousarray(x, dtype=np.float64)
    y1 = np.ascontiguousarray(y, dtype=np.float64)
    s1 = np.ascontiguousarray(sigma, dtype=np.float64)

    return c_kernels.gaussian_kernel(&x1[0], &y1[0], dimv, &s1[0])
batch = {}
batch["gaussian_kernel"] = all_gaussian_kernel
batch["linf_triangle_kernel"] = all_linf_triangle_kernel


def discretization_kernel(np.ndarray[np.double_t, ndim=1] x,
                          np.ndarray[np.double_t, ndim=1] y,
                          list dim,
                          np.ndarray[np.double_t, ndim=1] sigma):

    cdef vector[unsigned int] dimv = dim
    cdef np.ndarray[double, ndim=1, mode="c"] x1
    cdef np.ndarray[double, ndim=1, mode="c"] y1
    cdef np.ndarray[double, ndim=1, mode="c"] s1

    # unbox NumPy array into local variable x
    # make sure we have a contiguous array in C order
    # this might produce a temporary copy

    x1 = np.ascontiguousarray(x, dtype=np.float64)
    y1 = np.ascontiguousarray(y, dtype=np.float64)
    s1 = np.ascontiguousarray(sigma, dtype=np.float64)

    return c_kernels.linf_triangle_kernel(&x1[0], &y1[0], dimv, &s1[0])


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
