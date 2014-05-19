from rlpy.Representations.LocalBases import NonparametricLocalBases, RandomLocalBases
from rlpy.Domains import GridWorld, InfiniteTrackCartPole
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

try:
    from rlpy.Representations.kernels import *
except ImportError:
    from rlpy.Representations.slow_kernels import *
    print "C-Extensions for kernels not available, expect slow runtime"

def test_parametric_rep():
    """
    For fixed representations: test successful kernel function use, using
    varying number of features.
    Ensure get expected result.  Test normalization, ensure expected result.
    """
    for normalization in [False, True]: # verify everything with/out norm

        kernel = gaussian_kernel
        domain = InfiniteTrackCartPole.InfTrackCartPole() #2 continuous dims
        discretization = 20 # not used
        num = 1 # number of basis functions to use IN EACH DIMENSION
        resolution_min=1
        resolution_max=5
        rep = RandomLocalBases(domain, kernel, num, resolution_min,
                               resolution_max,
                               seed=1, normalization=normalization,
                               discretization=discretization)
        assert rep.features_num == num # in reality, theres one in each dim.

        # Center lies within statespace limits
        assert np.all(domain.statespace_limits[:,0] <= rep.centers[0])
        assert np.all(rep.centers[0] <= domain.statespace_limits[:,1])

        # width lies within resolution bounds
        statespace_range = domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]
        assert np.all(statespace_range / resolution_max <= rep.widths[0]) # widths[0] has `state_space_dims` cols
        assert np.all(rep.widths[0] <= statespace_range / resolution_min)

        phiVecOrigin = rep.phi(np.array([0,0], dtype=np.float), terminal=False)
        assert np.all(phiVecOrigin >= 0) # nonnegative feat func values

        # feature func only dependent on own dimension
        phiVec2 = rep.phi(np.array([0,1], dtype=np.float), terminal=False)

        if normalization:
            assert sum(phiVecOrigin) == 1
            assert sum(phiVec2) == 1

def test_visual():
    """ Test 2-D basis func visualization. """
    kernel = gaussian_kernel
    normalization=False
    domain = InfiniteTrackCartPole.InfTrackCartPole() #2 continuous dims
    discretization = 20 # not used
    num = 1 # number of basis functions to use
    resolution_min=1
    resolution_max=5
    rep = RandomLocalBases(domain, kernel, num, resolution_min,
                           resolution_max,
                           seed=1, normalization=normalization,
                           discretization=discretization)
    rep.plot_2d_feature_centers()


def test_nonparametric_rep():
    """
    For nonparametric representations: test successful kernel function use,
    ensure get expected result.
    """
    for normalization in [False, True]: # verify everything with/out norm

        kernel = gaussian_kernel
        normalization=False
        domain = InfiniteTrackCartPole.InfTrackCartPole() #2 continuous dims
        discretization = 20 # not used
        resolution=2
        # Start by making it impossible to add feats:
        max_similarity = 0
        rep = NonparametricLocalBases(domain, kernel, max_similarity,
                                      resolution,
                                      normalization=normalization,
                                      discretization=discretization)
        assert rep.features_num == 0 # ``num`` feats in each dimension
        origS = np.array([0,0], dtype=np.float)
        s2 = np.array([0,1], dtype=np.float)
        terminal = False # nonterminal states
        a=1 # arbitrary
        rep.pre_discover(origS, terminal, a, s2, terminal)

        # in the first call, automaticlaly add 1 feature since empty phi_s
        # is always < rep.max_similarity.
        # In ALL OTHER cases though, since max_similarity = 0, can never add
        # any more.
        assert rep.features_num == 1

        # Now make it really easy to add feats:
        max_similarity = np.inf
        rep = NonparametricLocalBases(domain, kernel, max_similarity,
                                      resolution,
                                      normalization=normalization,
                                      discretization=discretization)
        assert rep.features_num == 0 # ``num`` feats in each dimension
        origS = np.array([0,0], dtype=np.float)
        s2 = np.array([0,1], dtype=np.float)
        terminal = False # nonterminal states
        a=1 # arbitrary
        rep.pre_discover(origS, terminal, a, s2, terminal)

        # max_similarity == inf means we definitely shouldve added feat for
        # BOTH s and ns:
        assert rep.features_num == 2
        assert np.all(rep.centers[0,:] == origS)
        assert np.all(rep.centers[1,:] == s2)
        statespace_range = domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]
        assert np.all(rep.widths == statespace_range / resolution)

        phiVecOrigin = rep.phi(np.array([0,0], dtype=np.float), terminal=False)
        assert np.all(phiVecOrigin >= 0) # nonnegative feat func values

        # feature func only dependent on own dimension
        phiVec2 = rep.phi(np.array([0,1],dtype=np.float), terminal=False)
        np.all(phiVec2 >= 0)

        if normalization:
            assert sum(phiVecOrigin) == 1
            assert sum(phiVec2) == 1

def test_phi_post_expansion():
    """
    Ensure correct feature is activated for corresponding state, even after
    expansion.  Also tests if weight vector remains aligned with feat vec.

    """
    # TODO - could check to make sure weight vector remains aligned with
    # feat vec, even after expansion
