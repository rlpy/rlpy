# -*- coding: utf-8 -*-
"""
Representations which use local bases function (e.g. kernels) distributed
in the statespace according to some scheme (e.g. grid, random, on previous
samples)
"""
from Representation import Representation
import numpy as np
from kernels import batch


class LocalBases(Representation):

    centers = []
    widths = []

    def __init__(self, domain, logger, kernel, normalization=False, **kwargs):
        self.kernel = batch[kernel.__name__]
        self.normalization = normalization
        super(LocalBases, self).__init__(domain, logger)

    def phi_nonTerminal(self, s):
        v = self.kernel(s, self.centers, self.widths)
        if self.normalization and not v.sum() == 0.:
            # normalize such that each vector has a l1 norm of 1
            v /= v.sum()
        return v


class RandomLocalBases(LocalBases):

    def __init__(self, domain, logger, kernel, num=100, resolution_min=5,
                 resolution_max=None, seed=1, **kwargs):
        self.features_num = num
        dim_widths = (domain.statespace_limits[:, 1]
                      - domain.statespace_limits[:, 0])
        super(RandomLocalBases, self).__init__(domain, logger, kernel, **kwargs)
        rand_stream = np.random.RandomState(seed=seed)
        self.centers = np.zeros((num, len(dim_widths)))
        self.widths = np.zeros((num, len(dim_widths)))
        for i in xrange(num):
            for d in xrange(len(dim_widths)):
                self.centers[i, d] = rand_stream.uniform(domain.statespace_limits[d, 0],
                                             domain.statespace_limits[d, 1])
                self.widths[i, d] = rand_stream.uniform(dim_widths[d] / resolution_max,
                                             dim_widths[d] / resolution_min)
