from Domains import Acrobot
from Representations import TileCoding
from Tools import Logger
import numpy as np

def test_tile():
    domain = Acrobot(None)
    resolution_mat = .5 * np.ones((2,4))
    tile_matrix = np.array(np.mat("""
    48 48 48 48;
    1 18 18 18; 18 1 18 18; 18 18 1 18; 18 18 18 1;
    1 1 12 12; 1 12 12 1; 1 12 1 12; 12 1 1 12; 12 1 12 1; 12 12 1 1;
    1 1 1 18; 1 1 18 1; 1 18 1 1; 18 1 1 1"""), dtype="float")
    resolution_mat = tile_matrix
    resolution_mat[resolution_mat == 1] = 0.5
    t = TileCoding(num_tilings=[12, 3, 3, 3, 3]+[2]*6+[3]*4, memory=2000, logger=Logger(),
                    domain=domain, resolution_matrix=resolution_mat)
    for i in np.linspace(-1, 1, 20):
        print i
        a = np.nonzero(t.phi_nonTerminal(np.array([np.pi*i,0.,0.,0.])))[0]
        np.sort(a)
        print a
