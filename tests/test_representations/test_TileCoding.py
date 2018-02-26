from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from rlpy.Representations import TileCoding
from rlpy.Domains import GridWorld, InfiniteTrackCartPole
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

def test_number_of_cells():
    """ Ensure create appropriate # of cells (despite ``discretization``) """
    mapDir = os.path.join(__rlpy_location__, "Domains", "GridWorldMaps")
    mapname=os.path.join(mapDir, "4x5.txt") # expect 4*5 = 20 states
    domain = GridWorld(mapname=mapname)
    
    memory = 30 # Determines number of feats; it is the size of cache
    num_tilings = [2, ] # has 2 tilings
    resolutions = [4, ] # resolution of staterange / 4
    dimensions = [[0,1], ] # tiling over dimensions 0 and 1
    
    
    rep = TileCoding(domain, memory, num_tilings, resolutions, 
                     resolution_matrix=None, dimensions=dimensions,
                     safety="super") # super safety prevents any collisions
    assert rep.features_num == memory
    
    # TODO - test hashing function