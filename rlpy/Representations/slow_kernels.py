"""Pure python implementation of the kernels module"""
import numpy as np


def gaussian_kernel(x, y, dim, sigma):
    return np.exp(- float((((x[dim] - y[dim]) / sigma[dim]) ** 2).sum()))


def truncated_gaussian_kernel(x, y, dim, sigma, threshold):
    res = np.exp(- float((((x[dim] - y[dim]) / sigma[dim]) ** 2).sum()))
    res -= threshold
    if res < 0:
        return 0.
    res *= 1. / (1. - threshold)
    return res


def discretization_kernel(x, y, dim, sigma):
    return (
        np.all(np.floor(x[dim] / sigma[dim]) == np.floor(y[dim] / sigma[dim]))
    )


def linf_kernel(x, y, dim, sigma):
    return np.all(np.abs(x[dim] - y[dim]) < sigma[dim])


def linf_triangle_kernel(x, y, dim, sigma):
    d = 1. - (abs(x[dim] - y[dim]) / sigma[dim])
    d[d <= 0] = 0
    return d.min()


def all_gaussian_kernel(x, centers, widths):
    dimv = range(len(x))
    res = np.zeros(len(centers))

    for i in xrange(len(centers)):
        res[i] = gaussian_kernel(x, centers[i], dimv, widths[i])
    return res


def all_linf_triangle_kernel(x, centers, widths):

    dimv = range(len(x))
    res = np.zeros(len(centers))

    for i in xrange(len(centers)):
        res[i] = linf_triangle_kernel(x, centers[i], dimv, widths[i])
    return res


batch = {}
batch["gaussian_kernel"] = all_gaussian_kernel
batch["linf_triangle_kernel"] = all_linf_triangle_kernel
