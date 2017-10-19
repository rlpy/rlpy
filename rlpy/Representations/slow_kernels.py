"""Pure python implementation of the kernels module"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from past.utils import old_div
import numpy as np


def gaussian_kernel(x, y, dim, sigma):
    return np.exp(- float(((old_div((x[dim] - y[dim]), sigma[dim])) ** 2).sum()))


def truncated_gaussian_kernel(x, y, dim, sigma, threshold):
    res = np.exp(- float(((old_div((x[dim] - y[dim]), sigma[dim])) ** 2).sum()))
    res -= threshold
    if res < 0:
        return 0.
    res *= old_div(1., (1. - threshold))
    return res


def discretization_kernel(x, y, dim, sigma):
    return (
        np.all(np.floor(old_div(x[dim], sigma[dim])) == np.floor(old_div(y[dim], sigma[dim])))
    )


def linf_kernel(x, y, dim, sigma):
    return np.all(np.abs(x[dim] - y[dim]) < sigma[dim])


def linf_triangle_kernel(x, y, dim, sigma):
    d = 1. - (old_div(abs(x[dim] - y[dim]), sigma[dim]))
    d[d <= 0] = 0
    return d.min()


def all_gaussian_kernel(x, centers, widths):
    dimv = list(range(len(x)))
    res = np.zeros(len(centers))

    for i in range(len(centers)):
        res[i] = gaussian_kernel(x, centers[i], dimv, widths[i])
    return res


def all_linf_triangle_kernel(x, centers, widths):

    dimv = list(range(len(x)))
    res = np.zeros(len(centers))

    for i in range(len(centers)):
        res[i] = linf_triangle_kernel(x, centers[i], dimv, widths[i])
    return res


batch = {}
batch["gaussian_kernel"] = all_gaussian_kernel
batch["linf_triangle_kernel"] = all_linf_triangle_kernel
