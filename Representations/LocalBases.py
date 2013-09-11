# -*- coding: utf-8 -*-
"""
Representations which use local bases function (e.g. kernels) distributed
in the statespace according to some scheme (e.g. grid, random, on previous
samples)
"""
from Representation import Representation
import numpy as np
from kernels import batch
from Tools.GeneralTools import addNewElementForAllActions
import matplotlib.pyplot as plt


class LocalBases(Representation):
    """
    abstract base class for representations that use local basis functions
    """
    #: centers of bases
    centers = None
    #: widths of bases
    widths = None

    def __init__(self, domain, logger, kernel, normalization=False, **kwargs):
        self.kernel = batch[kernel.__name__]
        self.normalization = normalization
        self.centers = np.zeros((0, domain.statespace_limits.shape[0]))
        self.widths = np.zeros((0, domain.statespace_limits.shape[0]))
        super(LocalBases, self).__init__(domain, logger)

    def phi_nonTerminal(self, s):
        v = self.kernel(s, self.centers, self.widths)
        if self.normalization and not v.sum() == 0.:
            # normalize such that each vector has a l1 norm of 1
            v /= v.sum()
        return v

    def plot_2d_feature_centers(self, d1=None, d2=None):
        """
        plot the centers of all features in dimension d1 and d2.
        If no dimensions are specified, the first two continuous dimensions
        are shown.

        d1, d2: indices of dimensions to show
        """
        if d1 is None and d2 is None:
            # just take the first two dimensions
            d1, d2 = self.domain.continuous_dims[:2]
        plt.figure("Feature Dimensions {} and {}".format(d1, d2))
        for i in xrange(self.centers.shape[0]):
            plt.plot([self.centers[i, d1]], [self.centers[i, d2]], "r", marker="x")
        plt.draw()


class NonparametricLocalBases(LocalBases):
    def __init__(self, domain, logger, kernel, max_similarity=0.9, resolution=5, **kwargs):
        self.max_similarity = max_similarity
        self.common_width = (domain.statespace_limits[:, 1]
                             - domain.statespace_limits[:, 0]) / resolution
        self.features_num = 0
        super(NonparametricLocalBases, self).__init__(domain, logger, kernel, **kwargs)

    def pre_discover(self, s, terminal, a, sn, terminaln):
        norm = self.normalization
        expanded = 0
        self.normalization = False
        if not terminal:
            phi_s = self.phi_nonTerminal(s)
            if np.all(phi_s < self.max_similarity):
                self._add_feature(s)
                expanded += 1
        if not terminaln:
            phi_s = self.phi_nonTerminal(sn)
            if np.all(phi_s < self.max_similarity):
                self._add_feature(sn)
                expanded += 1
        self.normalization = norm
        return expanded

    def _add_feature(self, center):
        self.features_num += 1
        self.centers = np.vstack((self.centers, center))
        self.widths = np.vstack((self.widths, self.common_width))
        #TODO if normalized, use Q estimate for center to fill theta
        new = np.zeros((self.domain.actions_num, 1))
        self.theta = addNewElementForAllActions(self.theta, self.domain.actions_num, new)


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
