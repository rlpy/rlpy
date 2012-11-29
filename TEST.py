from Tools import *
from Representations import *
from Domains import *

#D = ChainMDP(5)
#R = Tabular(D)
#phi = R.phi(1)
#print R.phi_sa_from_phi_s(phi,1)

sA        = sp_matrix(10,dtype=bool)
sA[1:4,0] = 1
sA[8,0]   = 1
sB        = sp_matrix(10,dtype=bool)
sB[2:5,0]  = 2
sB[8:10,0] = 2
sB          = sB*2
A         = arange(10)*10
print sA.astype(int).toarray().flatten()
print sB.astype(int).toarray().flatten()
print sp_dot_sp(sA,sB)
print sA.astype(int).toarray().flatten()
print A.flatten()
print sp_dot_array(sA,A)

print sp_add2_array(sA,A)