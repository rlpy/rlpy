from Tools import *
from Representations import *
from Domains import *

#D = ChainMDP(5)
#R = Tabular(D)
#phi = R.phi(1)
#print R.phi_sa_from_phi_s(phi,1)

S       = sp_matrix(4)
S[1:3,0] = 1
A   = zeros(4)
ind = S.nonzero()[0]
print type(S)
print ind
print S[ind,0]
