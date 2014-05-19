from rlpy.Representations import Tabular
from rlpy.Domains import GridWorld, InfiniteTrackCartPole
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

def test_number_of_cells():
    """ Ensure create appropriate # of cells (despite ``discretization``) """
    mapDir = os.path.join(__rlpy_location__, "Domains", "GridWorldMaps")
    mapname=os.path.join(mapDir, "4x5.txt") # expect 4*5 = 20 states
    domain = GridWorld(mapname=mapname)
    
    rep = Tabular(domain, discretization=100)
    assert rep.features_num == 20
    rep = Tabular(domain, discretization=5)
    assert rep.features_num == 20
    

def test_phi_cells():
    """ Ensure correct feature is activated for corresponding state """
    mapDir = os.path.join(__rlpy_location__, "Domains", "GridWorldMaps")
    mapname=os.path.join(mapDir, "4x5.txt") # expect 4*5 = 20 states
    domain = GridWorld(mapname=mapname)
    
    # Allow internal represnetation to change -- just make sure each state has
    # a unique id that is consistently activated.
    rep = Tabular(domain)
    seenStates = -1 * np.ones(rep.features_num)
    for r in np.arange(4):
        for c in np.arange(5):
            phiVec = rep.phi(np.array([r,c]), terminal=False)
            assert sum(phiVec) == 1 # only 1 active feature
            activeInd = np.where(phiVec > 0)
            assert seenStates[activeInd] != True # havent seen it before
            seenStates[activeInd] = True
    assert np.all(seenStates == True) # we've covered all states
    
    # Optionally run some trajectories, make sure nothing changed

def test_continuous_discr():
    """ Ensure correct discretization in continuous state spaces """
    # NOTE - if possible, test a domain with mixed discr/continuous
    domain = InfiniteTrackCartPole.InfTrackCartPole() #2 continuous dims
    rep = Tabular(domain, discretization=20)
    assert rep.features_num == 400
    rep = Tabular(domain, discretization=50)
    assert rep.features_num == 2500