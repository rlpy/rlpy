from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from rlpy.Representations import BEBF
from rlpy.Domains import GridWorld, InfiniteTrackCartPole
import numpy as np
from rlpy.Tools import __rlpy_location__
import os

def test_new_feature():
    """
    Test that a valid feature function handle is returned when adding new feat.
    """
    
def test_batch_discover():
    """
    See that new features can be discovered without error
    """
    # Just do an example batch discovery.
