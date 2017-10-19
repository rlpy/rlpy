from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from rlpy.Representations import IndependentDiscretizationCompactBinary
from rlpy.Domains import GridWorld, InfiniteTrackCartPole, SystemAdministrator
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

def test_number_of_cells():
    """ Ensure create appropriate # of cells (despite ``discretization``) """
    mapDir = os.path.join(__rlpy_location__, "Domains", "GridWorldMaps")
    mapname=os.path.join(mapDir, "4x5.txt") # expect 4*5 = 20 states
    domain = GridWorld(mapname=mapname)
    
    rep = IndependentDiscretizationCompactBinary(domain, discretization=100)
    assert rep.features_num == 9+1
    rep = IndependentDiscretizationCompactBinary(domain, discretization=5)
    assert rep.features_num == 9+1

def test_compact_binary():
    """ Test representation on domain with some binary dimensions """
    mapDir = os.path.join(__rlpy_location__, "Domains", "SystemAdministratorMaps")
    mapname=os.path.join(mapDir, "20MachTutorial.txt") # expect 20+1 = 21 states
    domain = SystemAdministrator(networkmapname=mapname)
    
    rep = IndependentDiscretizationCompactBinary(domain)
    assert rep.features_num == 21
    
    stateVec = np.zeros(20)
    stateVec[0] = 1
    
    phiVec = rep.phi(stateVec, terminal=False)
    
    assert sum(phiVec) == 1
    assert phiVec[0] == 1
